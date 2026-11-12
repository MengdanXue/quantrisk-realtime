"""
策略参数优化器测试
Strategy Optimizer Tests

测试:
- 网格搜索优化
- 随机搜索优化
- 遗传算法优化
- 粒子群优化
- 参数规格处理

作者: TraderX Team
创建: 2025-12-24
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import numpy as np

import sys
sys.path.insert(0, 'src')

from quantrisk_realtime.realtime.strategy_optimizer import (
    StrategyOptimizer,
    OptimizationConfig,
    OptimizationMethod,
    OptimizationMetric,
    OptimizationResult,
    ParameterSpec,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    GeneticOptimizer,
    PSOOptimizer,
    create_optimizer,
    create_param_spec,
    quick_optimize,
)


# ============ Test Fixtures ============

@pytest.fixture
def simple_param_specs():
    """简单参数规格"""
    return [
        ParameterSpec('x', 0, 10, step=1, dtype='int'),
        ParameterSpec('y', 0, 1, dtype='float'),
    ]


@pytest.fixture
def strategy_param_specs():
    """策略参数规格"""
    return [
        ParameterSpec('fast_period', 5, 20, step=1, dtype='int'),
        ParameterSpec('slow_period', 20, 50, step=1, dtype='int'),
        ParameterSpec('stop_loss', 0.01, 0.05, dtype='float'),
    ]


def simple_objective(params):
    """简单目标函数: 最大化 -(x-5)^2 - (y-0.5)^2"""
    x = params.get('x', 0)
    y = params.get('y', 0)
    return -((x - 5) ** 2) - ((y - 0.5) ** 2)


def strategy_objective(params):
    """策略目标函数模拟"""
    fast = params.get('fast_period', 10)
    slow = params.get('slow_period', 30)
    stop = params.get('stop_loss', 0.02)

    # 模拟：快线周期太小或太大都不好
    # 慢线比快线大才有效
    # 止损太紧或太松都不好

    if fast >= slow:
        return -1.0  # 无效参数

    score = 1.0
    score -= abs(fast - 10) * 0.05
    score -= abs(slow - 30) * 0.02
    score -= abs(stop - 0.03) * 10

    return {
        'sharpe': score,
        'total_return': score * 0.1,
        'max_drawdown': 0.1 / (score + 1),
    }


# ============ ParameterSpec Tests ============

class TestParameterSpec:
    """参数规格测试"""

    def test_create_parameter_spec(self):
        """测试创建参数规格"""
        spec = ParameterSpec('test', 0, 100, step=10, dtype='int')

        assert spec.name == 'test'
        assert spec.min_value == 0
        assert spec.max_value == 100
        assert spec.step == 10
        assert spec.dtype == 'int'

    def test_get_grid_values_with_step(self):
        """测试带步长的网格值"""
        spec = ParameterSpec('x', 0, 10, step=2, dtype='int')
        values = spec.get_grid_values()

        assert 0 in values
        assert 2 in values
        assert 10 in values
        assert all(isinstance(v, int) for v in values)

    def test_get_grid_values_continuous(self):
        """测试连续参数网格值"""
        spec = ParameterSpec('x', 0, 1, dtype='float')
        values = spec.get_grid_values(n_points=5)

        assert len(values) == 5
        assert values[0] == 0
        assert values[-1] == 1

    def test_sample_random_int(self):
        """测试整数随机采样"""
        spec = ParameterSpec('x', 0, 10, dtype='int')

        for _ in range(10):
            value = spec.sample_random()
            assert isinstance(value, int)
            assert 0 <= value <= 10

    def test_sample_random_float(self):
        """测试浮点随机采样"""
        spec = ParameterSpec('x', 0, 1, dtype='float')

        for _ in range(10):
            value = spec.sample_random()
            assert isinstance(value, float)
            assert 0 <= value <= 1

    def test_sample_random_bool(self):
        """测试布尔随机采样"""
        spec = ParameterSpec('x', 0, 1, dtype='bool')

        values = [spec.sample_random() for _ in range(20)]
        assert True in values or False in values
        assert all(isinstance(v, bool) for v in values)


# ============ OptimizationConfig Tests ============

class TestOptimizationConfig:
    """优化配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = OptimizationConfig()

        assert config.method == OptimizationMethod.GRID
        assert config.metric == OptimizationMetric.SHARPE
        assert config.maximize is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = OptimizationConfig(
            method=OptimizationMethod.GENETIC,
            metric=OptimizationMetric.SORTINO,
            population_size=100,
        )

        assert config.method == OptimizationMethod.GENETIC
        assert config.metric == OptimizationMetric.SORTINO
        assert config.population_size == 100


# ============ GridSearchOptimizer Tests ============

class TestGridSearchOptimizer:
    """网格搜索优化器测试"""

    @pytest.mark.asyncio
    async def test_grid_search_basic(self, simple_param_specs):
        """测试基本网格搜索"""
        config = OptimizationConfig(
            method=OptimizationMethod.GRID,
            grid_points_per_param=5,
        )
        optimizer = GridSearchOptimizer(config)

        result = await optimizer.optimize(simple_param_specs, simple_objective)

        assert result.best_params is not None
        assert 'x' in result.best_params
        assert 'y' in result.best_params
        assert result.iterations > 0

    @pytest.mark.asyncio
    async def test_grid_search_finds_optimum(self, simple_param_specs):
        """测试网格搜索找到最优"""
        config = OptimizationConfig(
            method=OptimizationMethod.GRID,
            grid_points_per_param=11,  # 确保包含x=5
        )
        optimizer = GridSearchOptimizer(config)

        result = await optimizer.optimize(simple_param_specs, simple_objective)

        # 最优应该接近 x=5, y=0.5
        assert result.best_params['x'] == 5
        assert abs(result.best_params['y'] - 0.5) < 0.2


# ============ RandomSearchOptimizer Tests ============

class TestRandomSearchOptimizer:
    """随机搜索优化器测试"""

    @pytest.mark.asyncio
    async def test_random_search_basic(self, simple_param_specs):
        """测试基本随机搜索"""
        config = OptimizationConfig(
            method=OptimizationMethod.RANDOM,
            n_random_samples=20,
        )
        optimizer = RandomSearchOptimizer(config)

        result = await optimizer.optimize(simple_param_specs, simple_objective)

        assert result.best_params is not None
        assert result.iterations <= 20
        assert len(result.convergence_history) > 0

    @pytest.mark.asyncio
    async def test_random_search_early_stopping(self, simple_param_specs):
        """测试随机搜索早停"""
        config = OptimizationConfig(
            method=OptimizationMethod.RANDOM,
            n_random_samples=100,
            early_stopping_rounds=5,
            min_improvement=1.0,  # 很高的阈值，触发早停
        )
        optimizer = RandomSearchOptimizer(config)

        result = await optimizer.optimize(simple_param_specs, simple_objective)

        # 应该早停，迭代次数小于100
        assert result.iterations < 100 or result.early_stopped


# ============ GeneticOptimizer Tests ============

class TestGeneticOptimizer:
    """遗传算法优化器测试"""

    @pytest.mark.asyncio
    async def test_genetic_basic(self, simple_param_specs):
        """测试基本遗传算法"""
        config = OptimizationConfig(
            method=OptimizationMethod.GENETIC,
            population_size=20,
            max_iterations=10,
        )
        optimizer = GeneticOptimizer(config)

        result = await optimizer.optimize(simple_param_specs, simple_objective)

        assert result.best_params is not None
        assert result.method == OptimizationMethod.GENETIC
        assert len(result.convergence_history) > 0

    @pytest.mark.asyncio
    async def test_genetic_convergence(self, simple_param_specs):
        """测试遗传算法收敛"""
        config = OptimizationConfig(
            method=OptimizationMethod.GENETIC,
            population_size=30,
            max_iterations=20,
            random_seed=42,
        )
        optimizer = GeneticOptimizer(config)

        result = await optimizer.optimize(simple_param_specs, simple_objective)

        # 检查收敛趋势（后期分数应该不低于前期）
        history = result.convergence_history
        assert history[-1] >= history[0]


# ============ PSOOptimizer Tests ============

class TestPSOOptimizer:
    """粒子群优化器测试"""

    @pytest.mark.asyncio
    async def test_pso_basic(self, simple_param_specs):
        """测试基本粒子群优化"""
        config = OptimizationConfig(
            method=OptimizationMethod.PSO,
            n_particles=15,
            max_iterations=10,
        )
        optimizer = PSOOptimizer(config)

        result = await optimizer.optimize(simple_param_specs, simple_objective)

        assert result.best_params is not None
        assert result.method == OptimizationMethod.PSO

    @pytest.mark.asyncio
    async def test_pso_finds_optimum(self, simple_param_specs):
        """测试粒子群找到最优"""
        config = OptimizationConfig(
            method=OptimizationMethod.PSO,
            n_particles=30,
            max_iterations=30,
            random_seed=42,
        )
        optimizer = PSOOptimizer(config)

        result = await optimizer.optimize(simple_param_specs, simple_objective)

        # 应该接近最优 x=5, y=0.5
        assert abs(result.best_params['x'] - 5) <= 2
        assert abs(result.best_params['y'] - 0.5) < 0.3


# ============ StrategyOptimizer Tests ============

class TestStrategyOptimizer:
    """策略优化器测试"""

    @pytest.mark.asyncio
    async def test_create_with_grid(self):
        """测试创建网格搜索优化器"""
        optimizer = StrategyOptimizer(method='grid', metric='sharpe')

        assert optimizer.config.method == OptimizationMethod.GRID
        assert optimizer.config.metric == OptimizationMetric.SHARPE

    @pytest.mark.asyncio
    async def test_create_with_genetic(self):
        """测试创建遗传算法优化器"""
        optimizer = StrategyOptimizer(method='genetic', metric='sortino')

        assert optimizer.config.method == OptimizationMethod.GENETIC
        assert optimizer.config.metric == OptimizationMetric.SORTINO

    @pytest.mark.asyncio
    async def test_optimize_strategy_params(self, strategy_param_specs):
        """测试策略参数优化"""
        optimizer = StrategyOptimizer(
            method='random',
            metric='sharpe',
            n_random_samples=20,
        )

        result = await optimizer.optimize(strategy_param_specs, strategy_objective)

        assert result.best_params is not None
        assert 'fast_period' in result.best_params
        assert 'slow_period' in result.best_params
        assert 'stop_loss' in result.best_params

        # 验证参数有效性
        assert result.best_params['fast_period'] < result.best_params['slow_period']

    def test_optimize_sync(self, simple_param_specs):
        """测试同步优化"""
        optimizer = StrategyOptimizer(
            method='random',
            n_random_samples=10,
        )

        result = optimizer.optimize_sync(simple_param_specs, simple_objective)

        assert result.best_params is not None


# ============ Factory Functions Tests ============

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_optimizer(self):
        """测试创建优化器"""
        optimizer = create_optimizer(method='pso', metric='total_return')

        assert optimizer.config.method == OptimizationMethod.PSO
        assert optimizer.config.metric == OptimizationMetric.TOTAL_RETURN

    def test_create_param_spec(self):
        """测试创建参数规格"""
        spec = create_param_spec('test', 0, 100, step=5, dtype='int')

        assert spec.name == 'test'
        assert spec.min_value == 0
        assert spec.max_value == 100
        assert spec.step == 5

    @pytest.mark.asyncio
    async def test_quick_optimize(self, simple_param_specs):
        """测试快速优化"""
        result = await quick_optimize(
            param_specs=simple_param_specs,
            objective_func=simple_objective,
            method='random',
            n_iterations=15,
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None


# ============ OptimizationResult Tests ============

class TestOptimizationResult:
    """优化结果测试"""

    def test_result_to_dict(self):
        """测试结果序列化"""
        result = OptimizationResult(
            best_params={'x': 5, 'y': 0.5},
            best_score=0.95,
            metric=OptimizationMetric.SHARPE,
            method=OptimizationMethod.GENETIC,
            iterations=50,
            duration_seconds=10.5,
        )

        d = result.to_dict()

        assert d['best_params'] == {'x': 5, 'y': 0.5}
        assert d['best_score'] == 0.95
        assert d['metric'] == 'sharpe'
        assert d['method'] == 'genetic'
        assert d['iterations'] == 50


# ============ Edge Cases Tests ============

class TestEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_single_parameter(self):
        """测试单个参数优化"""
        param_specs = [ParameterSpec('x', 0, 10, dtype='float')]

        def objective(params):
            return -(params['x'] - 3) ** 2

        optimizer = create_optimizer(method='random', n_random_samples=20)
        result = await optimizer.optimize(param_specs, objective)

        assert abs(result.best_params['x'] - 3) < 2

    @pytest.mark.asyncio
    async def test_objective_returns_dict(self, simple_param_specs):
        """测试目标函数返回字典"""
        def objective(params):
            score = simple_objective(params)
            return {'sharpe': score, 'sortino': score * 0.9}

        optimizer = create_optimizer(method='random', metric='sharpe', n_random_samples=10)
        result = await optimizer.optimize(simple_param_specs, objective)

        assert result.best_score is not None

    @pytest.mark.asyncio
    async def test_objective_raises_exception(self, simple_param_specs):
        """测试目标函数抛出异常"""
        def bad_objective(params):
            if params['x'] == 5:
                raise ValueError("Test error")
            return simple_objective(params)

        optimizer = create_optimizer(method='random', n_random_samples=10)
        result = await optimizer.optimize(simple_param_specs, bad_objective)

        # 应该仍然返回结果
        assert result.best_params is not None

    @pytest.mark.asyncio
    async def test_minimize_max_drawdown(self, strategy_param_specs):
        """测试最小化最大回撤"""
        optimizer = StrategyOptimizer(
            method='random',
            metric='max_drawdown',
            maximize=False,  # 最小化
            n_random_samples=15,
        )

        result = await optimizer.optimize(strategy_param_specs, strategy_objective)

        assert result.best_params is not None


# ============ Run Tests ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

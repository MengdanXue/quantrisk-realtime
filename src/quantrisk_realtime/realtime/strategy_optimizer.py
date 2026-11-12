"""
策略参数优化器
Strategy Parameter Optimizer

提供多种优化算法用于策略参数调优:
- 网格搜索 (Grid Search)
- 随机搜索 (Random Search)
- 贝叶斯优化 (Bayesian Optimization)
- 遗传算法 (Genetic Algorithm)
- 粒子群优化 (Particle Swarm Optimization)

作者: TraderX Team
创建: 2025-12-24
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
import asyncio
import random
import itertools
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import copy

logger = logging.getLogger(__name__)


class OptimizationMethod(str, Enum):
    """优化方法"""
    GRID = "grid"                   # 网格搜索
    RANDOM = "random"               # 随机搜索
    BAYESIAN = "bayesian"           # 贝叶斯优化
    GENETIC = "genetic"             # 遗传算法
    PSO = "pso"                     # 粒子群优化


class OptimizationMetric(str, Enum):
    """优化目标指标"""
    SHARPE = "sharpe"               # 夏普比率
    SORTINO = "sortino"             # 索提诺比率
    CALMAR = "calmar"               # 卡尔玛比率
    TOTAL_RETURN = "total_return"   # 总收益
    MAX_DRAWDOWN = "max_drawdown"   # 最大回撤（最小化）
    PROFIT_FACTOR = "profit_factor" # 盈亏比
    WIN_RATE = "win_rate"           # 胜率
    CUSTOM = "custom"               # 自定义


@dataclass
class ParameterSpec:
    """参数规格"""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None    # None表示连续
    default: Optional[float] = None
    dtype: str = "float"            # float/int/bool

    def get_grid_values(self, n_points: int = 10) -> List[Any]:
        """获取网格值"""
        if self.step is not None:
            values = np.arange(self.min_value, self.max_value + self.step, self.step)
        else:
            values = np.linspace(self.min_value, self.max_value, n_points)

        if self.dtype == "int":
            values = [int(v) for v in values]
        elif self.dtype == "bool":
            values = [True, False]

        return list(values)

    def sample_random(self) -> Any:
        """随机采样"""
        if self.dtype == "bool":
            return random.choice([True, False])

        value = random.uniform(self.min_value, self.max_value)

        if self.step is not None:
            value = round(value / self.step) * self.step
            value = max(self.min_value, min(self.max_value, value))

        if self.dtype == "int":
            return int(round(value))

        return value


@dataclass
class OptimizationConfig:
    """优化配置"""
    method: OptimizationMethod = OptimizationMethod.GRID
    metric: OptimizationMetric = OptimizationMetric.SHARPE
    maximize: bool = True           # True=最大化, False=最小化

    # 通用配置
    max_iterations: int = 100
    early_stopping_rounds: int = 10
    min_improvement: float = 0.001
    parallel: bool = True
    n_workers: int = 4
    random_seed: Optional[int] = None

    # 网格搜索配置
    grid_points_per_param: int = 10

    # 随机搜索配置
    n_random_samples: int = 50

    # 遗传算法配置
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_count: int = 2

    # PSO配置
    n_particles: int = 30
    inertia_weight: float = 0.7
    cognitive_coef: float = 1.5
    social_coef: float = 1.5

    # 贝叶斯优化配置
    n_initial_points: int = 10
    acquisition_func: str = "ei"    # ei/pi/ucb


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    metric: OptimizationMetric
    method: OptimizationMethod

    # 详细结果
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    duration_seconds: float = 0.0

    # 收敛信息
    convergence_history: List[float] = field(default_factory=list)
    early_stopped: bool = False

    # 参数重要性（如果可用）
    parameter_importance: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'metric': self.metric.value,
            'method': self.method.value,
            'iterations': self.iterations,
            'duration_seconds': self.duration_seconds,
            'early_stopped': self.early_stopped,
            'parameter_importance': self.parameter_importance,
        }


class BaseOptimizer(ABC):
    """优化器基类"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        if config.random_seed:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

    @abstractmethod
    async def optimize(
        self,
        param_specs: List[ParameterSpec],
        objective_func: Callable,
    ) -> OptimizationResult:
        """运行优化"""
        pass

    def _evaluate_objective(
        self,
        params: Dict[str, Any],
        objective_func: Callable,
    ) -> float:
        """评估目标函数"""
        try:
            result = objective_func(params)

            # 如果返回的是dict，提取指标
            if isinstance(result, dict):
                metric_name = self.config.metric.value
                if metric_name in result:
                    score = result[metric_name]
                elif 'score' in result:
                    score = result['score']
                else:
                    score = result.get('total_return', 0)
            else:
                score = float(result)

            # 处理最大回撤（需要反转）
            if self.config.metric == OptimizationMetric.MAX_DRAWDOWN:
                score = -score  # 最大化负回撤 = 最小化回撤

            return score if self.config.maximize else -score

        except Exception as e:
            logger.warning(f"Objective evaluation failed for {params}: {e}")
            return float('-inf') if self.config.maximize else float('inf')


class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""

    async def optimize(
        self,
        param_specs: List[ParameterSpec],
        objective_func: Callable,
    ) -> OptimizationResult:
        start_time = datetime.now()

        # 生成参数网格
        param_grids = {}
        for spec in param_specs:
            param_grids[spec.name] = spec.get_grid_values(self.config.grid_points_per_param)

        # 生成所有组合
        keys = list(param_grids.keys())
        combinations = list(itertools.product(*param_grids.values()))

        logger.info(f"Grid search: {len(combinations)} combinations")

        best_score = float('-inf')
        best_params = {}
        all_results = []
        convergence = []

        # 并行评估
        if self.config.parallel and len(combinations) > 1:
            with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = []
                for combo in combinations:
                    params = dict(zip(keys, combo))
                    futures.append(executor.submit(self._evaluate_objective, params, objective_func))

                for i, (combo, future) in enumerate(zip(combinations, futures)):
                    params = dict(zip(keys, combo))
                    score = future.result()

                    all_results.append({'params': params, 'score': score})

                    if score > best_score:
                        best_score = score
                        best_params = params.copy()

                    convergence.append(best_score)
        else:
            for combo in combinations:
                params = dict(zip(keys, combo))
                score = self._evaluate_objective(params, objective_func)

                all_results.append({'params': params, 'score': score})

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                convergence.append(best_score)

        duration = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score if self.config.maximize else -best_score,
            metric=self.config.metric,
            method=OptimizationMethod.GRID,
            all_results=all_results,
            iterations=len(combinations),
            duration_seconds=duration,
            convergence_history=convergence,
        )


class RandomSearchOptimizer(BaseOptimizer):
    """随机搜索优化器"""

    async def optimize(
        self,
        param_specs: List[ParameterSpec],
        objective_func: Callable,
    ) -> OptimizationResult:
        start_time = datetime.now()

        best_score = float('-inf')
        best_params = {}
        all_results = []
        convergence = []
        no_improvement_count = 0

        for i in range(self.config.n_random_samples):
            # 随机采样参数
            params = {spec.name: spec.sample_random() for spec in param_specs}

            score = self._evaluate_objective(params, objective_func)
            all_results.append({'params': params, 'score': score})

            if score > best_score:
                if best_score != float('-inf'):
                    improvement = (score - best_score) / abs(best_score) if best_score != 0 else float('inf')
                    if improvement < self.config.min_improvement:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0
                best_score = score
                best_params = params.copy()
            else:
                no_improvement_count += 1

            convergence.append(best_score)

            # 早停
            if no_improvement_count >= self.config.early_stopping_rounds:
                logger.info(f"Early stopping at iteration {i+1}")
                break

        duration = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score if self.config.maximize else -best_score,
            metric=self.config.metric,
            method=OptimizationMethod.RANDOM,
            all_results=all_results,
            iterations=len(all_results),
            duration_seconds=duration,
            convergence_history=convergence,
            early_stopped=no_improvement_count >= self.config.early_stopping_rounds,
        )


class GeneticOptimizer(BaseOptimizer):
    """遗传算法优化器"""

    async def optimize(
        self,
        param_specs: List[ParameterSpec],
        objective_func: Callable,
    ) -> OptimizationResult:
        start_time = datetime.now()

        # 初始化种群
        population = self._initialize_population(param_specs)
        fitness_scores = [self._evaluate_objective(ind, objective_func) for ind in population]

        best_score = max(fitness_scores)
        best_idx = fitness_scores.index(best_score)
        best_params = population[best_idx].copy()

        all_results = []
        convergence = [best_score]
        no_improvement_count = 0

        for gen in range(self.config.max_iterations):
            # 选择
            selected = self._selection(population, fitness_scores)

            # 交叉
            offspring = self._crossover(selected, param_specs)

            # 变异
            offspring = self._mutation(offspring, param_specs)

            # 精英保留
            elite_indices = np.argsort(fitness_scores)[-self.config.elitism_count:]
            elites = [population[i] for i in elite_indices]

            # 新种群
            population = elites + offspring[:self.config.population_size - self.config.elitism_count]

            # 评估
            fitness_scores = [self._evaluate_objective(ind, objective_func) for ind in population]

            # 更新最优
            gen_best = max(fitness_scores)
            if gen_best > best_score:
                improvement = (gen_best - best_score) / abs(best_score) if best_score != 0 else float('inf')
                if improvement >= self.config.min_improvement:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                best_score = gen_best
                best_idx = fitness_scores.index(best_score)
                best_params = population[best_idx].copy()
            else:
                no_improvement_count += 1

            convergence.append(best_score)

            # 记录本代最优
            all_results.append({
                'generation': gen + 1,
                'best_score': gen_best,
                'avg_score': np.mean(fitness_scores),
            })

            # 早停
            if no_improvement_count >= self.config.early_stopping_rounds:
                logger.info(f"Early stopping at generation {gen+1}")
                break

        duration = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score if self.config.maximize else -best_score,
            metric=self.config.metric,
            method=OptimizationMethod.GENETIC,
            all_results=all_results,
            iterations=len(convergence) - 1,
            duration_seconds=duration,
            convergence_history=convergence,
            early_stopped=no_improvement_count >= self.config.early_stopping_rounds,
        )

    def _initialize_population(self, param_specs: List[ParameterSpec]) -> List[Dict[str, Any]]:
        """初始化种群"""
        return [
            {spec.name: spec.sample_random() for spec in param_specs}
            for _ in range(self.config.population_size)
        ]

    def _selection(
        self,
        population: List[Dict],
        fitness: List[float],
    ) -> List[Dict]:
        """轮盘赌选择"""
        # 处理负适应度
        min_fitness = min(fitness)
        adjusted = [f - min_fitness + 1e-6 for f in fitness]
        total = sum(adjusted)
        probs = [f / total for f in adjusted]

        selected = []
        for _ in range(len(population)):
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    selected.append(copy.deepcopy(population[i]))
                    break

        return selected

    def _crossover(
        self,
        population: List[Dict],
        param_specs: List[ParameterSpec],
    ) -> List[Dict]:
        """交叉操作"""
        offspring = []

        for i in range(0, len(population) - 1, 2):
            parent1 = population[i]
            parent2 = population[i + 1]

            if random.random() < self.config.crossover_rate:
                # 单点交叉
                child1, child2 = {}, {}
                crossover_point = random.randint(1, len(param_specs) - 1)

                for j, spec in enumerate(param_specs):
                    if j < crossover_point:
                        child1[spec.name] = parent1[spec.name]
                        child2[spec.name] = parent2[spec.name]
                    else:
                        child1[spec.name] = parent2[spec.name]
                        child2[spec.name] = parent1[spec.name]

                offspring.extend([child1, child2])
            else:
                offspring.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])

        return offspring

    def _mutation(
        self,
        population: List[Dict],
        param_specs: List[ParameterSpec],
    ) -> List[Dict]:
        """变异操作"""
        spec_dict = {s.name: s for s in param_specs}

        for individual in population:
            for name, spec in spec_dict.items():
                if random.random() < self.config.mutation_rate:
                    individual[name] = spec.sample_random()

        return population


class PSOOptimizer(BaseOptimizer):
    """粒子群优化器"""

    async def optimize(
        self,
        param_specs: List[ParameterSpec],
        objective_func: Callable,
    ) -> OptimizationResult:
        start_time = datetime.now()

        n_dims = len(param_specs)

        # 初始化粒子
        particles = self._initialize_particles(param_specs)
        velocities = [np.zeros(n_dims) for _ in range(self.config.n_particles)]

        # 个体最优
        personal_best = [p.copy() for p in particles]
        personal_best_scores = [self._evaluate_particle(p, param_specs, objective_func) for p in particles]

        # 全局最优
        global_best_idx = np.argmax(personal_best_scores)
        global_best = particles[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        convergence = [global_best_score]
        all_results = []
        no_improvement_count = 0

        for iteration in range(self.config.max_iterations):
            for i in range(self.config.n_particles):
                # 更新速度
                r1, r2 = random.random(), random.random()
                cognitive = self.config.cognitive_coef * r1 * (personal_best[i] - particles[i])
                social = self.config.social_coef * r2 * (global_best - particles[i])
                velocities[i] = self.config.inertia_weight * velocities[i] + cognitive + social

                # 更新位置
                particles[i] = particles[i] + velocities[i]

                # 边界处理
                for j, spec in enumerate(param_specs):
                    particles[i][j] = np.clip(particles[i][j], spec.min_value, spec.max_value)

                # 评估
                score = self._evaluate_particle(particles[i], param_specs, objective_func)

                # 更新个体最优
                if score > personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = score

                    # 更新全局最优
                    if score > global_best_score:
                        improvement = (score - global_best_score) / abs(global_best_score) if global_best_score != 0 else float('inf')
                        if improvement >= self.config.min_improvement:
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                        global_best = particles[i].copy()
                        global_best_score = score

            convergence.append(global_best_score)
            all_results.append({
                'iteration': iteration + 1,
                'best_score': global_best_score,
                'avg_score': np.mean(personal_best_scores),
            })

            no_improvement_count += 1

            if no_improvement_count >= self.config.early_stopping_rounds:
                logger.info(f"Early stopping at iteration {iteration+1}")
                break

        # 转换为参数字典
        best_params = self._array_to_params(global_best, param_specs)
        duration = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_score=global_best_score if self.config.maximize else -global_best_score,
            metric=self.config.metric,
            method=OptimizationMethod.PSO,
            all_results=all_results,
            iterations=len(convergence) - 1,
            duration_seconds=duration,
            convergence_history=convergence,
            early_stopped=no_improvement_count >= self.config.early_stopping_rounds,
        )

    def _initialize_particles(self, param_specs: List[ParameterSpec]) -> List[np.ndarray]:
        """初始化粒子位置"""
        particles = []
        for _ in range(self.config.n_particles):
            particle = np.array([spec.sample_random() for spec in param_specs])
            particles.append(particle)
        return particles

    def _evaluate_particle(
        self,
        particle: np.ndarray,
        param_specs: List[ParameterSpec],
        objective_func: Callable,
    ) -> float:
        """评估粒子"""
        params = self._array_to_params(particle, param_specs)
        return self._evaluate_objective(params, objective_func)

    def _array_to_params(
        self,
        arr: np.ndarray,
        param_specs: List[ParameterSpec],
    ) -> Dict[str, Any]:
        """数组转参数字典"""
        params = {}
        for i, spec in enumerate(param_specs):
            value = arr[i]
            if spec.dtype == "int":
                value = int(round(value))
            elif spec.dtype == "bool":
                value = value > 0.5
            params[spec.name] = value
        return params


class StrategyOptimizer:
    """
    策略参数优化器

    统一接口封装多种优化算法

    使用方式:
    ```python
    optimizer = StrategyOptimizer(
        method='genetic',
        metric='sharpe',
    )

    # 定义参数空间
    param_specs = [
        ParameterSpec('fast_period', 5, 20, step=1, dtype='int'),
        ParameterSpec('slow_period', 20, 60, step=1, dtype='int'),
        ParameterSpec('stop_loss', 0.01, 0.05),
    ]

    # 运行优化
    result = await optimizer.optimize(
        param_specs=param_specs,
        objective_func=backtest_strategy,
    )
    ```
    """

    def __init__(
        self,
        method: str = 'grid',
        metric: str = 'sharpe',
        maximize: bool = True,
        **kwargs,
    ):
        method_map = {
            'grid': OptimizationMethod.GRID,
            'random': OptimizationMethod.RANDOM,
            'genetic': OptimizationMethod.GENETIC,
            'pso': OptimizationMethod.PSO,
            'bayesian': OptimizationMethod.BAYESIAN,
        }

        metric_map = {
            'sharpe': OptimizationMetric.SHARPE,
            'sortino': OptimizationMetric.SORTINO,
            'calmar': OptimizationMetric.CALMAR,
            'total_return': OptimizationMetric.TOTAL_RETURN,
            'max_drawdown': OptimizationMetric.MAX_DRAWDOWN,
            'profit_factor': OptimizationMetric.PROFIT_FACTOR,
            'win_rate': OptimizationMetric.WIN_RATE,
        }

        self.config = OptimizationConfig(
            method=method_map.get(method, OptimizationMethod.GRID),
            metric=metric_map.get(metric, OptimizationMetric.SHARPE),
            maximize=maximize,
            **{k: v for k, v in kwargs.items() if hasattr(OptimizationConfig, k)},
        )

        self._optimizer = self._create_optimizer()

    def _create_optimizer(self) -> BaseOptimizer:
        """创建优化器实例"""
        if self.config.method == OptimizationMethod.GRID:
            return GridSearchOptimizer(self.config)
        elif self.config.method == OptimizationMethod.RANDOM:
            return RandomSearchOptimizer(self.config)
        elif self.config.method == OptimizationMethod.GENETIC:
            return GeneticOptimizer(self.config)
        elif self.config.method == OptimizationMethod.PSO:
            return PSOOptimizer(self.config)
        else:
            raise ValueError(f"Unsupported method: {self.config.method}")

    async def optimize(
        self,
        param_specs: List[ParameterSpec],
        objective_func: Callable,
    ) -> OptimizationResult:
        """
        运行优化

        Args:
            param_specs: 参数规格列表
            objective_func: 目标函数，接收params dict，返回score或metrics dict

        Returns:
            OptimizationResult
        """
        logger.info(f"Starting optimization with {self.config.method.value} method")
        result = await self._optimizer.optimize(param_specs, objective_func)
        logger.info(f"Optimization complete: best_score={result.best_score:.4f}")
        return result

    def optimize_sync(
        self,
        param_specs: List[ParameterSpec],
        objective_func: Callable,
    ) -> OptimizationResult:
        """同步运行优化"""
        return asyncio.run(self.optimize(param_specs, objective_func))


# ============ 工厂函数 ============

def create_optimizer(
    method: str = 'grid',
    metric: str = 'sharpe',
    **kwargs,
) -> StrategyOptimizer:
    """
    创建策略优化器

    Args:
        method: 优化方法 (grid/random/genetic/pso)
        metric: 优化指标 (sharpe/sortino/total_return/max_drawdown/win_rate)
        **kwargs: 其他配置参数

    Returns:
        StrategyOptimizer实例
    """
    return StrategyOptimizer(method=method, metric=metric, **kwargs)


def create_param_spec(
    name: str,
    min_val: float,
    max_val: float,
    step: Optional[float] = None,
    dtype: str = 'float',
) -> ParameterSpec:
    """创建参数规格"""
    return ParameterSpec(
        name=name,
        min_value=min_val,
        max_value=max_val,
        step=step,
        dtype=dtype,
    )


async def quick_optimize(
    param_specs: List[ParameterSpec],
    objective_func: Callable,
    method: str = 'random',
    metric: str = 'sharpe',
    n_iterations: int = 50,
) -> OptimizationResult:
    """
    快速优化

    Args:
        param_specs: 参数规格
        objective_func: 目标函数
        method: 优化方法
        metric: 优化指标
        n_iterations: 迭代次数

    Returns:
        OptimizationResult
    """
    optimizer = create_optimizer(
        method=method,
        metric=metric,
        n_random_samples=n_iterations,
        max_iterations=n_iterations,
    )
    return await optimizer.optimize(param_specs, objective_func)

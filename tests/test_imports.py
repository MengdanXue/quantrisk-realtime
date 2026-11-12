# Copyright 2026 Mengdan Xue
# SPDX-License-Identifier: Apache-2.0
"""Every module of the package imports with the dev extras installed.

The models (ml extras imported lazily inside functions) and observability packages
came without tests of their own; this checks only that the extraction left no
dangling import, not that they compute anything correctly.
"""

import importlib
import pkgutil

import pytest

import quantrisk_realtime

MODULES = sorted(m.name for m in pkgutil.walk_packages(quantrisk_realtime.__path__, "quantrisk_realtime."))


def test_the_package_has_the_extracted_subpackages():
    top = {name.split(".")[1] for name in MODULES}
    assert top == {"core", "datasources", "exchanges", "models", "observability", "online_learning", "realtime"}


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)

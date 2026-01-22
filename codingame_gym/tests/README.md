# CodingGame Gym Tests

Integration and system tests for the CodingGame Gym Python wrapper.

## ⚠️ Important: JVM Limitation

**Tests that use the gym must be run individually** due to JPype's limitation that the JVM can only be started once per Python process.

### Run tests individually:
```bash
pytest tests/test_reset_consistency.py
pytest tests/test_performance.py::test_quick_performance_sanity  
pytest tests/test_consistency_multi_mode.py
```

### Run all tests (some will be skipped):
```bash
pytest tests/ -v
```
When running all tests together, only the first test that starts the JVM will run; others will be automatically skipped.

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest tests/test_reset_consistency.py
```

### Run only fast tests (skip slow tests):
```bash
pytest -m "not slow"
```

### Run with verbose output:
```bash
pytest -v -s
```

## Test Categories

Tests are organized with pytest markers:

- **`@pytest.mark.integration`** - Integration tests (moderate speed, seconds)
- **`@pytest.mark.slow`** - Long-running tests (minutes)
- **`@pytest.mark.benchmark`** - Performance benchmarks

## Test Files

### `test_reset_consistency.py`
Tests that resetting the environment with the same seed produces identical scores across multiple runs.

**Run:** `pytest tests/test_reset_consistency.py`

### `test_consistency_multi_mode.py`
Comprehensive test that verifies CLI, Python Gym, GameRunner, and CLI concurrent modes all produce identical scores for the same games.

**Run standalone (100 games):** `python tests/test_consistency_multi_mode.py`  
**Run pytest (10 games):** `pytest tests/test_consistency_multi_mode.py`  
**Run custom amount:** `python tests/test_consistency_multi_mode.py 50`

### `test_performance.py`
Performance benchmarks comparing Java and Python gym implementations.

**Run full benchmark:** `pytest tests/test_performance.py -v -s`  
**Run quick sanity:** `pytest tests/test_performance.py::test_quick_performance_sanity`

## Legacy Scripts

These standalone scripts still exist for manual testing:

- `perf_test_python.py` - Python performance test (called by test_performance.py)
- `tests/compare_results.py` - Compare Java/Python results (imported by test_performance.py)
- `run_perf_comparison.sh` - Shell script for full performance comparison workflow

## Configuration

Test settings are in `pytest.ini` at the project root.

"""Pytest configuration for tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--num-games",
        action="store",
        default="100",
        help="Number of games to run in performance test (default: 100)"
    )


@pytest.fixture
def num_games(request):
    """Get the number of games from command-line option."""
    return int(request.config.getoption("--num-games"))

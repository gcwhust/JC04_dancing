"""Share one real source-XML simulation; individual modules reset its state."""
import pytest

from jc04_dancing.simulation import Simulation, build_timeline


@pytest.fixture(scope='session')
def simulation():
    return Simulation(build_timeline('swan_lake'), mode='free')

import pytest
from aiida import orm, load_profile

load_profile()


@pytest.fixture
def LiCoO2():
    a, b, c, d = 1.40803, 0.81293, 4.68453, 1.62585
    cell = [[a, -b, c], [0.0, d, c], [-a, -b, c]]
    sites = [
        ['Co', 'Co', (0, 0, 0)],
        ['O', 'O', (0, 0, 3.6608)],
        ['O', 'O', (0, 0, 10.392)],
        ['Li', 'Li', (0, 0, 7.0268)],
    ]
    structure = orm.StructureData(cell=cell)
    for kind, name, position in sites:
        structure.append_atom(position=position, symbols=kind, name=name)
    return structure


@pytest.fixture
def pw_code():
    return orm.load_code('pw-7.2@localhost')


@pytest.fixture
def hp_code():
    return orm.load_code('hp-7.2@localhost')

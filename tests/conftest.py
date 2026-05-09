import io

import pytest
from aiida import load_profile, orm
from aiida_pseudo.data.pseudo import UpfData
from aiida_pseudo.groups.family import SsspFamily

load_profile()

pytest_plugins = ['aiida.tools.pytest_fixtures']


@pytest.fixture
def test_structure():
    try:
        query = orm.StructureData.collection.query(filters={'extras.formula': 'LiCoO2'})
        return query.first()[0]
    except Exception:
        pass

    a, b, c, d = 1.40803, 0.81293, 4.68453, 1.62585
    cell = [[a, -b, c], [0.0, d, c], [-a, -b, c]]
    sites = [
        ['Co', 'Co', (0, 0, 0)],
        ['O', 'O', (0, 0, 3.6608)],
        ['O', 'O', (0, 0, 10.392)],
        ['Li', 'Li', (0, 0, 7.0268)],
    ]
    structure = orm.StructureData(cell=cell)
    structure.base.extras.set('formula', 'LiCoO2')
    for kind, name, position in sites:
        structure.append_atom(position=position, symbols=kind, name=name)
    structure.store()
    return structure


@pytest.fixture
def pw_code(aiida_code_installed):
    return aiida_code_installed('pw-7.4@localhost')


@pytest.fixture
def hp_code(aiida_code_installed):
    return aiida_code_installed('hp-7.4@localhost')


@pytest.fixture(scope='session')
def generate_upf_data():
    """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""

    def _generate_upf_data(element, filename=None, params=None):
        """Return `UpfData` node.

        NOTE: `params` is used to render the generated UPF unique per pseudo
        """
        extras = '\n\t\t'.join(f'{key}="{value}"' for key, value in (params or {}).items())
        content = f"""
            <UPF version="2.0.1">
                <PP_HEADER
                    element="{element}"
                    z_valence="4.0"
                    {extras}
                />
            </UPF>
        """
        stream = io.BytesIO(content.encode('utf-8'))

        filename = element
        if params:
            filename += f'_{"_".join(params.values())}'
        filename += '.upf'

        return UpfData(stream, filename=filename)

    return _generate_upf_data


ELEMENTS = ['Li', 'Co', 'O']

CUTOFFS = {
    element: {
        'cutoff_wfc': 30.0,
        'cutoff_rho': 240.0,
    }
    for element in ELEMENTS
}

SSSP_VERSION = '1.3'


@pytest.fixture(scope='session', autouse=True)
def sssp(generate_upf_data):
    for functional in ('PBE', 'PBEsol'):
        for accuracy in ('efficiency', 'precision'):
            label = f'SSSP/{SSSP_VERSION}/{functional}/{accuracy}'
            family = SsspFamily(label)
            family.store()
            nodes = []
            for element in ELEMENTS:
                params = {
                    'functional': functional,
                    'accuracy': accuracy,
                }
                filename = f'{element}_{"_".join(params.values())}.upf'
                upf = generate_upf_data(element, filename, params)
                upf.store()
                nodes.append(upf)
            family.add_nodes(nodes)
            family.set_cutoffs(CUTOFFS, accuracy, unit='Ry')

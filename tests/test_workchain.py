import pytest

from aiidalab_qe_hp.model import HpSettingsModel
from aiidalab_qe_hp.workchain import get_builder


@pytest.mark.parametrize('method', ['one-shot', 'self-consistent'])
def test_workchain(test_structure, pw_code, hp_code, method):
    model = HpSettingsModel()
    model.structure_uuid = test_structure.uuid
    model.calculation_type = 'DFT+U+V'
    model.hubbard_u = [['Co', '3d', 3.0]]
    model.hubbard_v = [['Co', '3d', 'O', '2p', 1.0]]
    model.protocol = 'fast'
    model.method = method

    codes = {
        'pw': {
            'code': pw_code,
            'nodes': 2,
            'ntasks_per_node': 3,
            'cpus_per_task': 4,
            'max_wallclock_seconds': 3600,
        },
        'hp': {
            'code': hp_code,
            'nodes': 5,
            'ntasks_per_node': 7,
            'cpus_per_task': 8,
            'max_wallclock_seconds': 3600,
        },
    }

    parameters = {
        'hp': model.get_model_state(),
        'workchain': {
            'protocol': 'fast',
            'relax_type': 'none',
            'electronic_type': 'insulator',
            'spin_type': 'collinear',
        },
        'advanced': {'initial_magnetic_moments': {'Co': 0.0, 'O': 0.0, 'Li': 0.0}},
    }

    builder = get_builder(codes, test_structure, parameters, **{})

    pw_resources = builder.scf.pw.metadata.options.resources
    assert pw_resources['num_machines'] == codes['pw']['nodes']
    assert pw_resources['num_mpiprocs_per_machine'] == codes['pw']['ntasks_per_node']
    assert pw_resources['num_cores_per_mpiproc'] == codes['pw']['cpus_per_task']

    hp_resources = builder.hubbard.hp.metadata.options.resources
    assert hp_resources['num_machines'] == codes['hp']['nodes']
    assert hp_resources['num_mpiprocs_per_machine'] == codes['hp']['ntasks_per_node']
    assert hp_resources['num_cores_per_mpiproc'] == codes['hp']['cpus_per_task']

    if method == 'self-consistent':
        for namespace in ('base_init_relax', 'base_relax'):
            resources = builder.relax[namespace].pw.metadata.options.resources
            assert resources['num_machines'] == codes['pw']['nodes']
            assert resources['num_mpiprocs_per_machine'] == codes['pw']['ntasks_per_node']
            assert resources['num_cores_per_mpiproc'] == codes['pw']['cpus_per_task']

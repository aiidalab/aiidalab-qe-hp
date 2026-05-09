from aiidalab_qe_hp.model import HpSettingsModel
from aiidalab_qe_hp.workchain import get_builder


def test_workchain(test_structure, pw_code, hp_code):
    model = HpSettingsModel()
    model.structure_uuid = test_structure.uuid
    model.calculation_type = 'DFT+U+V'
    model.hubbard_u = [['Co', '3d', 3.0]]
    model.hubbard_v = [['Co', '3d', 'O', '2p', 1.0]]
    model.protocol = 'fast'

    codes = {
        'pw': {
            'code': pw_code,
            'nodes': 1,
            'ntasks_per_node': 1,
            'cpus_per_task': 1,
            'max_wallclock_seconds': 3600,
        },
        'hp': {
            'code': hp_code,
            'nodes': 1,
            'ntasks_per_node': 1,
            'cpus_per_task': 1,
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
    print(builder)

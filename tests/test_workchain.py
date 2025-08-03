def test_workchain(LiCoO2, pw_code, hp_code):
    from aiidalab_qe_hp.workchain import get_builder
    from aiidalab_qe_hp.setting import HPSettingsPanel
    from aiidalab_qe_hp.model import HPSettingsModel
    from aiida.engine import run

    model = HPSettingsModel()
    structure = LiCoO2
    model.input_structure = structure
    model.calculation_type = 'DFT+U+V'
    model.hubbard_u = [['Co', '3d', 3.0]]
    model.hubbard_v = [['Co', '3d', 'O', '2p', 1.0]]
    model.protocol = 'fast'

    codes = {
        'pw': {'code': pw_code,
               'nodes': 1,
               'ntasks_per_node': 1,
               'cpus_per_task': 1,
               'max_wallclock_seconds': 3600
               },

        'hp': {'code': hp_code,
               'nodes': 1,
               'ntasks_per_node': 1,
               'cpus_per_task': 1,
               'max_wallclock_seconds': 3600
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

    builder = get_builder(codes, structure, parameters, **{})
    # run(builder)
    print(builder)

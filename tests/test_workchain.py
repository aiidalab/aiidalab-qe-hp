def test_workchain(LiCoO2, pw_code, hp_code):
    from aiidalab_qe_hp.workchain import get_builder
    from aiidalab_qe_hp.setting import Setting
    from aiida.engine import run

    structure = LiCoO2
    setting = Setting()
    setting.input_structure = LiCoO2
    setting.calculation_type.value = "DFT+U+V"
    setting.hubbard_u_map["Co"][0].value = True
    setting.hubbard_u_map["Co"][2].value = "3d"
    setting.hubbard_u_map["Co"][3].value = 3.0
    setting.hubbard_v_map[("Co", "O")][0].value = True
    setting.hubbard_v_map[("Co", "O")][2].value = "3d"
    setting.hubbard_v_map[("Co", "O")][4].value = "2p"
    setting.hubbard_v_map[("Co", "O")][5].value = 1.0

    codes = {
        "pw": pw_code,
        "hp": hp_code,
    }

    parameters = {
        "hp": setting.get_panel_value(),
        "workchain": {
            "protocol": "fast",
            "relax_type": "none",
            "electronic_type": "insulator",
            "spin_type": "collinear",
        },
        "advanced": {"initial_magnetic_moments": {"Co": 0.0, "O": 0.0, "Li": 0.0}},
    }

    builder = get_builder(codes, structure, parameters, **{})
    # run(builder)
    print(builder)

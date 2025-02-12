def test_setting(LiCoO2):
    from aiidalab_qe_hp.setting import HPSettingsPanel
    from aiidalab_qe_hp.model import HPSettingsModel

    model = HPSettingsModel()
    model.input_structure = LiCoO2
    setting = HPSettingsPanel(model=model)
    setting.calculation_type.value = 'DFT+U+V'
    setting.hubbard_u_map['Co'][0].value = True
    setting.hubbard_u_map['Co'][2].value = '3d'
    setting.hubbard_u_map['Co'][3].value = 3.0
    setting.hubbard_v_map[('Co', 'O')][0].value = True
    setting.hubbard_v_map[('Co', 'O')][2].value = '3d'
    setting.hubbard_v_map[('Co', 'O')][4].value = '2p'
    setting.hubbard_v_map[('Co', 'O')][5].value = 1.0
    parameters = model.get_model_state()
    assert parameters == {
        'method': 'one-shot',
        'qpoints_distance': 1000.0,
        'parallelize_atoms': True,
        'parallelize_qpoints': True,
        'calculation_type': 'DFT+U+V',
        'projector_type': 'ortho-atomic',
        'hubbard_u': [['Co', '3d', 3.0]],
        'hubbard_v': [['Co', '3d', 'O', '2p', 1.0]],
    }
    parameters['hubbard_u'][0][2] = 4.0
    model.set_model_state(parameters)
    assert setting.hubbard_u_map['Co'][3].value == 4.0

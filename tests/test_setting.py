def test_setting(LiCoO2):
    from aiidalab_qe_hp.setting import HPSettingsPanel
    from aiidalab_qe_hp.model import HPSettingsModel

    model = HPSettingsModel()
    model.input_structure = LiCoO2
    structure = LiCoO2
    model.input_structure = structure
    model.calculation_type = 'DFT+U+V'
    model.hubbard_u = [['Co', '3d', 3.0]]
    model.hubbard_v = [['Co', '3d', 'O', '2p', 1.0]]
    model.protocol = 'fast'
    parameters = model.get_model_state()
    setting = HPSettingsPanel(model=model)
    setting._update_hubbard_tables() # Render
    assert parameters == {
        'method': 'one-shot',
        'qpoints_distance': 1.2,
        'parallelize_atoms': True,
        'parallelize_qpoints': True,
        'calculation_type': 'DFT+U+V',
        'projector_type': 'ortho-atomic',
        'hubbard_u': [['Co', '3d', 3.0]],
        'hubbard_v': [['Co', '3d', 'O', '2p', 1.0]],
    }
    parameters['hubbard_u'][0][2] = 4.0
    setting._model.set_model_state(parameters)
    assert model.hubbard_u == [['Co', '3d', 4.0]]

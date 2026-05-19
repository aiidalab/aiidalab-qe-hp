from aiidalab_qe_hp.model import HpSettingsModel
from aiidalab_qe_hp.settings import HpSettingsPanel


def test_settings(test_structure):
    model = HpSettingsModel()
    panel = HpSettingsPanel(model=model)
    panel.render()

    model.structure_uuid = test_structure.uuid
    model.calculation_type = 'DFT+U+V'
    model.hubbard_u = [['Co', '3d', 3.0], ['Li', '3d', 2.0], ['O', '2p', 1.0]]
    model.hubbard_v = [['Co', '3d', 'O', '2p', 1.0], ['Co', '3d', 'Li', '3d', 0.5], ['Li', '3d', 'O', '2p', 0.5]]
    model.protocol = 'fast'

    parameters = model.get_model_state()
    assert parameters == {
        'method': 'one-shot',
        'qpoints_distance': 1.2,
        'parallelize_atoms': True,
        'parallelize_qpoints': True,
        'calculation_type': 'DFT+U+V',
        'projector_type': 'ortho-atomic',
        'hubbard_u': [['Co', '3d', 3.0], ['Li', '3d', 2.0], ['O', '2p', 1.0]],
        'hubbard_v': [['Co', '3d', 'O', '2p', 1.0], ['Co', '3d', 'Li', '3d', 0.5], ['Li', '3d', 'O', '2p', 0.5]],
        'qpoints_override': False,
        'relax_type': 'cell',
    }

    parameters['hubbard_u'][0][2] = 4.0

    model.set_model_state(parameters)
    assert model.hubbard_u == [['Co', '3d', 4.0], ['Li', '3d', 2.0], ['O', '2p', 1.0]]

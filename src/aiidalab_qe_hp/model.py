# hp_model.py
import traitlets as tl
from aiidalab_qe.common.panel import ConfigurationSettingsModel
from aiidalab_qe.common.mixins import HasInputStructure

class HPSettingsModel(ConfigurationSettingsModel, HasInputStructure):
    """Traitlets-based model for the HP plugin settings."""
    title = 'HP Settings'
    identifier = 'hp'

    dependencies = [
        'input_structure',
        'workchain.protocol',
    ]

    # Basic HP traitlets
    method = tl.Unicode(default_value='one-shot')
    calculation_type = tl.Unicode(default_value='DFT+U')
    projector_type = tl.Unicode(default_value='ortho-atomic')
    qpoints_distance = tl.Float(default_value=1.0)
    parallelize_atoms = tl.Bool(default_value=True)
    parallelize_qpoints = tl.Bool(default_value=True)
    protocol = tl.Unicode(allow_none=True)

    # Hubbard U, V will be stored as lists-of-lists or something similar
    # e.g. each entry in hubbard_u might be [kind_name, manifold, U-value],
    # each entry in hubbard_v might be [kind1, manifold1, kind2, manifold2, V-value]
    hubbard_u = tl.List(trait=tl.List(), default_value=[])
    hubbard_v = tl.List(trait=tl.List(), default_value=[])

    def get_model_state(self) -> dict:
        """Return a dictionary capturing the current model state."""
        return {
            'method': self.method,
            'calculation_type': self.calculation_type,
            'projector_type': self.projector_type,
            'qpoints_distance': self.qpoints_distance,
            'parallelize_atoms': self.parallelize_atoms,
            'parallelize_qpoints': self.parallelize_qpoints,
            'hubbard_u': self.hubbard_u,
            'hubbard_v': self.hubbard_v,
        }

    def set_model_state(self, parameters: dict):
        """Set the model state from a given dictionary."""
        self.method = parameters.get('method', 'one-shot')
        self.calculation_type = parameters.get('calculation_type', 'DFT+U')
        self.projector_type = parameters.get('projector_type', 'ortho-atomic')
        self.qpoints_distance = parameters.get('qpoints_distance', 1.0)
        self.parallelize_atoms = parameters.get('parallelize_atoms', True)
        self.parallelize_qpoints = parameters.get('parallelize_qpoints', True)
        self.hubbard_u = parameters.get('hubbard_u', [])
        self.hubbard_v = parameters.get('hubbard_v', [])

    @tl.observe('protocol')
    def _observe_protocol(self, change):
        """When 'protocol' changes, use it to update qpoints_distance etc."""
        if change['new'] is None:
            return
        from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
        parameters = PwBaseWorkChain.get_protocol_inputs(change['new'])
        # Example usage: If kpoints_distance is part of the protocol
        if 'kpoints_distance' in parameters:
            self.qpoints_distance = parameters['kpoints_distance'] * 4

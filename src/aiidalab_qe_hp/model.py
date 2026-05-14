import traitlets as tl
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiidalab_qe.common.mixins import HasInputStructure
from aiidalab_qe.common.panel import PanelModel


class HpSettingsModel(PanelModel, HasInputStructure):
    """Traitlets-based model for the HP plugin settings."""

    title = 'HP Settings'
    identifier = 'hp'

    dependencies = [
        'structure_uuid',
        'workchain.protocol',
    ]

    protocol = tl.Unicode(None, allow_none=True)

    method = tl.Unicode('one-shot')
    calculation_type = tl.Unicode('DFT+U')
    calculation_type_options = tl.List(
        trait=tl.Unicode(),
        default_value=['DFT+U'],
    )
    projector_type = tl.Unicode('ortho-atomic')
    qpoints_distance = tl.Float(1.0)
    qpoints_override = tl.Bool(False)
    parallelize_atoms = tl.Bool(True)
    parallelize_qpoints = tl.Bool(True)
    relax_type = tl.Unicode('cell')

    # Hubbard U, V will be stored as lists-of-lists or something similar
    # e.g. each entry in hubbard_u might be [kind_name, manifold, U-value],
    # each entry in hubbard_v might be [kind1, manifold1, kind2, manifold2, V-value]
    hubbard_u = tl.List(
        trait=tl.List(),
        default_value=[],
    )
    hubbard_v = tl.List(
        trait=tl.List(),
        default_value=[],
    )

    @tl.observe('structure_uuid')
    def update_calculation_type_options_from_structure(self, _):
        num_atoms = len(self.input_structure.sites) if self.has_structure else 0
        options = ['DFT+U', 'DFT+U+V'] if num_atoms > 1 else ['DFT+U']
        self.calculation_type_options = options
        self._initialize_hubbard_lists()

    @tl.observe('protocol')
    def update_qpoints_distance_from_protocol(self, _=None):
        if self.protocol is not None:
            parameters = PwBaseWorkChain.get_protocol_inputs(self.protocol)
            if 'kpoints_distance' in parameters:
                self.qpoints_distance = parameters['kpoints_distance'] * 4

    @tl.observe('qpoints_override')
    def update_qpoints_distance(self, change):
        if not change['new']:
            # Reset to protocol value if override is turned off
            self.update_qpoints_distance_from_protocol()

    def get_model_state(self) -> dict:
        """Return a dictionary capturing the current model state."""
        return {
            'method': self.method,
            'relax_type': self.relax_type,
            'calculation_type': self.calculation_type,
            'projector_type': self.projector_type,
            'qpoints_distance': self.qpoints_distance,
            'qpoints_override': self.qpoints_override,
            'parallelize_atoms': self.parallelize_atoms,
            'parallelize_qpoints': self.parallelize_qpoints,
            'hubbard_u': self._get_active_hubbard_u(),
            'hubbard_v': self._get_active_hubbard_v(),
        }

    def set_model_state(self, parameters: dict):
        """Set the model state from a given dictionary."""
        self.method = parameters.get('method', 'one-shot')
        self.relax_type = parameters.get('relax_type', 'cell')
        self.calculation_type = parameters.get('calculation_type', 'DFT+U')
        self.projector_type = parameters.get('projector_type', 'ortho-atomic')
        self.qpoints_distance = parameters.get('qpoints_distance', 1.0)
        self.qpoints_override = parameters.get('qpoints_override', False)
        self.parallelize_atoms = parameters.get('parallelize_atoms', True)
        self.parallelize_qpoints = parameters.get('parallelize_qpoints', True)
        self.hubbard_u = self._set_active_hubbard_u(parameters.get('hubbard_u', []))
        self.hubbard_v = self._set_active_hubbard_v(parameters.get('hubbard_v', []))

    def _get_active_hubbard_u(self):
        """Return only the active Hubbard U entries (e.g. those with non-empty manifold)."""
        return [entry for entry in self.hubbard_u if entry[1] != '']

    def _get_active_hubbard_v(self):
        """Return only the active Hubbard V entries (e.g. those with non-empty manifolds)."""
        return [entry for entry in self.hubbard_v if entry[1] != '' and entry[3] != '']

    def _set_active_hubbard_u(self, active_u):
        """Update default values with active entries."""
        updated_u = []
        for entry in self.hubbard_u:
            kind_name = entry[0]
            active_entry = next((e for e in active_u if e[0] == kind_name), None)
            if active_entry:
                updated_u.append(active_entry)
            else:
                updated_u.append(entry)
        return updated_u

    def _set_active_hubbard_v(self, active_v):
        """Update default values with active entries."""
        updated_v = []
        for entry in self.hubbard_v:
            kind1, _, kind2, _, _ = entry
            active_entry = next((e for e in active_v if e[0] == kind1 and e[2] == kind2), None)
            if active_entry:
                updated_v.append(active_entry)
            else:
                updated_v.append(entry)
        return updated_v

    def _initialize_hubbard_lists(self):
        """Initialize the Hubbard U and V lists based on the current structure."""
        if not self.has_structure:
            self.hubbard_u = []
            self.hubbard_v = []
            return

        kind_names = self.input_structure.get_kind_names()

        self.hubbard_u = [[kn, '', 0.0] for kn in kind_names]

        self.hubbard_v = [
            [kn1, '', kn2, '', 0.0]
            for i in range(len(kind_names))
            for j in range(i + 1, len(kind_names))
            for kn1, kn2 in [(kind_names[i], kind_names[j])]
        ]

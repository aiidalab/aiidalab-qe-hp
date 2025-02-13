# hp_panel.py
import ipywidgets as ipw
import traitlets as tl
from aiidalab_qe.common.panel import ConfigurationSettingsPanel
from aiida.orm import Float, Bool
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
    create_kpoints_from_distance,
)
from .model import HPSettingsModel  # import the model you just created


class HPSettingsPanel(ConfigurationSettingsPanel[HPSettingsModel]):
    """Panel (view/controller) for HP plugin, built against HPSettingsModel."""

    # Pre-defined HTML help strings:
    one_shot_description = """<div>Single calculation without iterative self-consistent procedure, no structural optimization. </div>"""
    self_consistent_description = """<div>Iterative self-consistent procedure, with structural optimization. </div>"""
    dft_u_description = """<div>Only on-site U Hubbard parameter is computed. </div>"""
    dft_u_v_description = """<div>On-site U and inter-site V Hubbard parameters are computed. </div>"""
    atomic_description = """<div>Non-orthogonalized atomic orbitals. </div>"""
    ortho_atomic_description = """<div>Löwdin-orthogonalized atomic orbitals. </div>"""

    def __init__(self, model: HPSettingsModel, **kwargs):
        super().__init__(model=model, **kwargs)
        self._model = model  # keep a reference

        # 1) Create your widgets.
        self.method = ipw.Dropdown(
            options=['one-shot', 'self-consistent'],
            description='Method:',
            style={'description_width': 'initial'},
        )
        self.method_description = ipw.HTML()

        self.calculation_type = ipw.Dropdown(
            options=['DFT+U', 'DFT+U+V'],
            description='Calculation type:',
            style={'description_width': 'initial'},
        )
        self.calculation_type_description = ipw.HTML()

        self.projector_type = ipw.Dropdown(
            options=['atomic', 'ortho-atomic'],
            description='Hubbard projector type:',
            style={'description_width': 'initial'},
        )
        self.projector_type_description = ipw.HTML()

        self.qpoints_distance = ipw.BoundedFloatText(
            min=0.0,
            step=0.05,
            description='q-points distance (1/Å):',
            style={'description_width': 'initial'},
        )
        self.qpoint_mesh = ipw.HTML()
        self.qpoints_override_prompt = ipw.HTML('<b>&nbsp;&nbsp;Override&nbsp;</b>')
        self.qpoints_override = ipw.Checkbox(
            description='',
            indent=False,
            layout=ipw.Layout(max_width='10%'),
        )

        self.parallelize_atoms = ipw.Checkbox(
            description='Use parallelization over perturbed Hubbard atoms.',
            style={'description_width': 'initial'},
            layout={'width': '600px'},
        )
        self.parallelize_qpoints = ipw.Checkbox(
            description='Use parallelization over q points.',
            style={'description_width': 'initial'},
        )

        # Dynamic U/V table placeholders:
        self.Hubbard_U_title = ipw.HTML(
            """<div style="padding-top: 0px; padding-bottom: 0px">
            <h4>Select atoms for which on-site Hubbard U must be computed</h4></div>"""
        )
        self.hubbard_u = ipw.VBox()
        self.Hubbard_V_title = ipw.HTML(
            """<div style="padding-top: 0px; padding-bottom: 0px">
            <h4>Select couples of atoms for which inter-site Hubbard V must be computed</h4></div>"""
        )
        self.hubbard_v = ipw.VBox()

        # Info/warning area:
        self.Info = ipw.HTML()

        # 2) Link or observe them to the model’s traitlets.
        ipw.link((self._model, 'method'), (self.method, 'value'))
        self.method.observe(self._sync_method_description, 'value')

        ipw.link((self._model, 'calculation_type'), (self.calculation_type, 'value'))
        self.calculation_type.observe(self._sync_calculation_type_description, 'value')

        ipw.link((self._model, 'projector_type'), (self.projector_type, 'value'))
        self.projector_type.observe(self._sync_projector_type_description, 'value')

        ipw.link((self._model, 'qpoints_distance'), (self.qpoints_distance, 'value'))
        self.qpoints_distance.observe(self._on_qpoints_distance_change, 'value')

        ipw.link((self._model, 'parallelize_atoms'), (self.parallelize_atoms, 'value'))
        ipw.link((self._model, 'parallelize_qpoints'), (self.parallelize_qpoints, 'value'))

        # Example of disabling qpoints_distance if not overridden:
        def _toggle_distance(change):
            self.qpoints_distance.disabled = not change['new']
        self.qpoints_override.observe(_toggle_distance, 'value')

        # 3) Observe changes in input structure and re-generate table
        self._model.observe(self._update_hubbard_tables, 'input_structure')
        self._model.observe(self._update_hubbard_tables, 'calculation_type')

        # 4) Arrange them in self.children
        self.children = [
            ipw.HBox([self.method, self.method_description]),
            ipw.HBox([self.calculation_type, self.calculation_type_description]),
            ipw.HBox([self.projector_type, self.projector_type_description]),
            ipw.HBox([
                self.qpoints_distance,
                self.qpoint_mesh,
                self.qpoints_override_prompt,
                self.qpoints_override,
            ]),
            self.parallelize_atoms,
            self.parallelize_qpoints,
            ipw.VBox(layout=ipw.Layout(border='1px solid black')),
            ipw.VBox(children=[self.Hubbard_U_title, self.hubbard_u]),
            ipw.VBox(children=[self.Hubbard_V_title, self.hubbard_v]),
            self.Info,
        ]

        # Mark that we've built everything:
        self.rendered = True

    # Called by `render()`, but we can skip if we already set `self.children` in __init__.
    def render(self):
        """Render the panel (if not rendered)."""
        if self.rendered:
            return
        # If you prefer to build all widgets lazily, do it here.
        self.rendered = True
        self._update_hubbard_tables()

    # Sync the short descriptive text below the dropdowns:
    def _sync_method_description(self, _=None):
        if self.method.value == 'one-shot':
            self.method_description.value = self.one_shot_description
        else:
            self.method_description.value = self.self_consistent_description

    def _sync_calculation_type_description(self, _=None):
        if self.calculation_type.value == 'DFT+U':
            self.calculation_type_description.value = self.dft_u_description
        else:
            self.calculation_type_description.value = self.dft_u_v_description

    def _sync_projector_type_description(self, _=None):
        if self.projector_type.value == 'atomic':
            self.projector_type_description.value = self.atomic_description
        else:
            self.projector_type_description.value = self.ortho_atomic_description

    # Dynamically compute the qpoint mesh upon distance changes
    def _on_qpoints_distance_change(self, change):
        """Update the Q-point mesh info text whenever the distance changes."""
        if not self._model.input_structure:
            return
        if self.qpoints_distance.value > 0:
            mesh = create_kpoints_from_distance.process_class._func(
                self._model.input_structure,
                Float(self.qpoints_distance.value),
                Bool(False),
            )
            self.qpoint_mesh.value = f'Mesh {mesh.get_kpoints_mesh()[0]}'
        else:
            self.qpoint_mesh.value = 'Please select a number > 0.0'

    # Generate or update the “Hubbard U” and “Hubbard V” tables
    def _update_hubbard_tables(self, _=None):
        self._generate_hubbard_u()
        self._generate_hubbard_v()

    def _generate_hubbard_u(self):
        """Example approach: one row per atomic kind."""
        self.hubbard_u.children = []  # Clear old
        structure = self._model.input_structure
        if not structure:
            return

        # Table header:
        header = ipw.HBox([
            ipw.HTML(value=''),
            ipw.HTML(value='Atomic Type', layout=ipw.Layout(width='100px')),
            ipw.HTML(value='Manifold', layout=ipw.Layout(width='100px')),
            ipw.HTML(value='U value (eV)', layout=ipw.Layout(width='100px')),
        ])
        rows = [header]

        # We could store a temporary mapping of widget→(kind_name, manifold_widget, U_widget)
        # so that we can assemble self._model.hubbard_u list automatically
        self._hubbard_u_map = {}

        for kind in structure.kinds:
            kind_name = kind.name
            checkbox = ipw.Checkbox(value=False, indent=False, layout=ipw.Layout(width='30px'))
            name_html = ipw.HTML(value=kind_name, layout=ipw.Layout(width='80px'))
            manifold = ipw.Text(value='', layout=ipw.Layout(width='100px'))
            u_val = ipw.FloatText(value=0.0, layout=ipw.Layout(width='80px'))

            row = ipw.HBox([checkbox, name_html, manifold, u_val])
            rows.append(row)

            # For demonstration, whenever these widgets change, we can reassemble into model.hubbard_u
            def on_change(_=None):
                self._assemble_hubbard_u()
            checkbox.observe(on_change, 'value')
            manifold.observe(on_change, 'value')
            u_val.observe(on_change, 'value')

            self._hubbard_u_map[kind_name] = (checkbox, manifold, u_val)

        self.hubbard_u.children = rows

    def _generate_hubbard_v(self):
        """Example approach: only if calc_type == 'DFT+U+V', build pairwise table."""
        self.hubbard_v.children = []
        structure = self._model.input_structure
        if not structure or self.calculation_type.value != 'DFT+U+V':
            return

        header = ipw.HBox([
            ipw.HTML(value='', layout=ipw.Layout(width='30px')),
            ipw.HTML(value='Type1', layout=ipw.Layout(width='60px')),
            ipw.HTML(value='Manif.1', layout=ipw.Layout(width='80px')),
            ipw.HTML(value='Type2', layout=ipw.Layout(width='60px')),
            ipw.HTML(value='Manif.2', layout=ipw.Layout(width='80px')),
            ipw.HTML(value='V value (eV)', layout=ipw.Layout(width='80px')),
        ])
        rows = [header]
        self._hubbard_v_map = {}

        kind_names = [k.name for k in structure.kinds]
        for i in range(len(kind_names)):
            for j in range(i + 1, len(kind_names)):
                kn1 = kind_names[i]
                kn2 = kind_names[j]
                checkbox = ipw.Checkbox(value=False, indent=False, layout=ipw.Layout(width='30px'))
                manif1 = ipw.Text(value='', layout=ipw.Layout(width='80px'))
                manif2 = ipw.Text(value='', layout=ipw.Layout(width='80px'))
                v_val = ipw.FloatText(value=0.0, layout=ipw.Layout(width='80px'))

                row = ipw.HBox([checkbox,
                                ipw.HTML(kn1, layout=ipw.Layout(width='50px')),
                                manif1,
                                ipw.HTML(kn2, layout=ipw.Layout(width='50px')),
                                manif2,
                                v_val])
                rows.append(row)

                def on_change(_=None):
                    self._assemble_hubbard_v()
                for widget in (checkbox, manif1, manif2, v_val):
                    widget.observe(on_change, 'value')

                self._hubbard_v_map[(kn1, kn2)] = (checkbox, manif1, manif2, v_val)

        self.hubbard_v.children = rows

    def _assemble_hubbard_u(self):
        """Rebuild the model.hubbard_u from the checkboxes/text fields."""
        new_list = []
        for kind_name, (cb, manifold, u_val) in self._hubbard_u_map.items():
            if cb.value:  # only if user has "checked" that they want U for this kind
                new_list.append([kind_name, manifold.value, max(u_val.value, 1e-10)])
        self._model.hubbard_u = new_list

    def _assemble_hubbard_v(self):
        """Rebuild the model.hubbard_v from checkboxes/text fields."""
        new_list = []
        for (kn1, kn2), (cb, man1, man2, v_val) in self._hubbard_v_map.items():
            if cb.value:
                new_list.append([
                    kn1,
                    man1.value,
                    kn2,
                    man2.value,
                    max(v_val.value, 1e-10),
                ])
        self._model.hubbard_v = new_list

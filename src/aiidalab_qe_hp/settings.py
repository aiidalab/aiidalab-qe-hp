import ipywidgets as ipw
from aiida import orm
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
    create_kpoints_from_distance,
)
from aiidalab_qe.common.panel import ConfigurationSettingsPanel

from .model import HpSettingsModel


class HpSettingsPanel(ConfigurationSettingsPanel[HpSettingsModel]):
    """Panel (view/controller) for HP plugin, built against HpSettingsModel."""

    one_shot_description = """
        <div>
            Single calculation without iterative self-consistent procedure, no structural optimization.
        </div>
    """
    self_consistent_description = """
        <div>
            Iterative self-consistent procedure, with structural optimization.
        </div>
    """
    dft_u_description = """
        <div>
            Only on-site U Hubbard parameter is computed.
        </div>
    """
    dft_u_v_description = """
        <div>
            On-site U and inter-site V Hubbard parameters are computed.
        </div>
    """
    atomic_description = """
        <div>
            Non-orthogonalized atomic orbitals.
        </div>
    """
    ortho_atomic_description = """
        <div>
            Löwdin-orthogonalized atomic orbitals.
        </div>
    """

    def __init__(self, model: HpSettingsModel, **kwargs):
        super().__init__(model=model, **kwargs)
        self._model = model  # keep a reference

        self._model.observe(
            self._on_input_structure_change,
            'structure_uuid',
        )
        self._model.observe(
            self._on_protocol_change,
            'protocol',
        )
        self._model.observe(
            self._on_method_change,
            'method',
        )
        self._model.observe(
            self._on_calculation_type_change,
            'calculation_type',
        )
        self._model.observe(
            self._on_projector_type_change,
            'projector_type',
        )
        self._model.observe(
            self._on_qpoints_distance_change,
            'qpoints_distance',
        )

    def render(self):
        if self.rendered:
            return

        label_style = {'description_width': '150px'}
        hbox_layout = {'grid_gap': '4px'}

        self.method = ipw.Dropdown(
            options=['one-shot', 'self-consistent'],
            description='Method:',
            style=label_style,
        )
        ipw.link(
            (self._model, 'method'),
            (self.method, 'value'),
        )
        self.method_description = ipw.HTML()

        self.relax_type = ipw.Dropdown(
            options=['atomic', 'cell'],
            description='Relaxation type:',
            style=label_style,
        )
        ipw.link(
            (self._model, 'relax_type'),
            (self.relax_type, 'value'),
        )
        self.relax_type_description = ipw.HTML("""
            <div>
                Choose between cell relaxation (default) or atomic relaxation.
            </div>
        """)

        self.relax_row = ipw.HBox(
            children=[
                self.relax_type,
                self.relax_type_description,
            ],
            layout={**hbox_layout, 'display': 'none'},
        )

        self.calculation_type = ipw.Dropdown(
            description='Calculation type:',
            style=label_style,
        )
        ipw.dlink(
            (self._model, 'calculation_type_options'),
            (self.calculation_type, 'options'),
        )
        ipw.link(
            (self._model, 'calculation_type'),
            (self.calculation_type, 'value'),
        )
        self.calculation_type_description = ipw.HTML()

        self.projector_type = ipw.Dropdown(
            options=['atomic', 'ortho-atomic'],
            description='Hubbard projector type:',
            style=label_style,
        )
        ipw.link(
            (self._model, 'projector_type'),
            (self.projector_type, 'value'),
        )
        self.projector_type_description = ipw.HTML()

        self.qpoints_distance = ipw.BoundedFloatText(
            min=0.0,
            step=0.05,
            description='q-points distance (1/Å):',
            style=label_style,
        )
        ipw.link(
            (self._model, 'qpoints_distance'),
            (self.qpoints_distance, 'value'),
        )
        ipw.dlink(
            (self._model, 'qpoints_override'),
            (self.qpoints_distance, 'disabled'),
            lambda override: not override,
        )
        self.qpoint_mesh = ipw.HTML()
        self.qpoints_override = ipw.Checkbox(
            description='Override',
            indent=False,
            layout=ipw.Layout(max_width='10%'),
        )
        ipw.link(
            (self._model, 'qpoints_override'),
            (self.qpoints_override, 'value'),
        )

        self.parallelize_atoms = ipw.Checkbox(
            description='Use parallelization over perturbed Hubbard atoms.',
            indent=True,
            style=label_style,
            layout={**hbox_layout, 'width': 'auto'},
        )
        ipw.link(
            (self._model, 'parallelize_atoms'),
            (self.parallelize_atoms, 'value'),
        )

        self.parallelize_qpoints = ipw.Checkbox(
            description='Use parallelization over q points.',
            indent=True,
            style=label_style,
            layout={**hbox_layout, 'width': 'auto'},
        )
        ipw.link(
            (self._model, 'parallelize_qpoints'),
            (self.parallelize_qpoints, 'value'),
        )

        self.Hubbard_U_title = ipw.HTML("""
            <div style="padding-top: 0px; padding-bottom: 0px">
                <h4>Select atoms for which on-site Hubbard U must be computed</h4>
            </div>
        """)
        self.hubbard_u = ipw.VBox()

        self.Hubbard_V_title = ipw.HTML("""
            <div style="padding-top: 0px; padding-bottom: 0px">
                <h4>
                    Select couples of atoms for which inter-site Hubbard V must be computed
                </h4>
            </div>
        """)
        self.hubbard_v = ipw.VBox()

        self.children = [
            ipw.HBox(
                children=[
                    self.method,
                    self.method_description,
                ],
                layout=hbox_layout,
            ),
            self.relax_row,
            ipw.HBox(
                children=[
                    self.calculation_type,
                    self.calculation_type_description,
                ],
                layout=hbox_layout,
            ),
            ipw.HBox(
                children=[
                    self.projector_type,
                    self.projector_type_description,
                ],
                layout=hbox_layout,
            ),
            ipw.HBox(
                children=[
                    self.qpoints_distance,
                    self.qpoint_mesh,
                    self.qpoints_override,
                ],
                layout=hbox_layout,
            ),
            self.parallelize_atoms,
            self.parallelize_qpoints,
            ipw.HTML('<hr>'),
            ipw.VBox(children=[self.hubbard_u]),
            ipw.VBox(children=[self.hubbard_v]),
        ]

        self.rendered = True

        self._toggle_method_description()
        self._toggle_relax_row()
        self._toggle_calculation_type_description()
        self._toggle_projector_type_description()
        self._update_qpoints_mesh()
        self._update_hubbard_tables()

    def _on_input_structure_change(self, _):
        self._update_qpoints_mesh()
        self._update_hubbard_tables()

    def _on_protocol_change(self, _):
        self._model.update(specific='protocol')

    def _on_method_change(self, _):
        if not self.rendered:
            return
        self._toggle_method_description()
        self._toggle_relax_row()

    def _on_calculation_type_change(self, _):
        if not self.rendered:
            return
        self._toggle_calculation_type_description()
        self._update_hubbard_tables()

    def _on_projector_type_change(self, _):
        if not self.rendered:
            return
        self._toggle_projector_type_description()

    def _on_qpoints_distance_change(self, _):
        self._update_qpoints_mesh()

    def _toggle_method_description(self):
        if self._model.method == 'one-shot':
            self.method_description.value = self.one_shot_description
        else:
            self.method_description.value = self.self_consistent_description

    def _toggle_relax_row(self):
        if self._model.method == 'one-shot':
            self.relax_row.layout.display = 'none'
        else:
            self.relax_row.layout.display = 'flex'

    def _toggle_calculation_type_description(self):
        if self._model.calculation_type == 'DFT+U':
            self.calculation_type_description.value = self.dft_u_description
        else:
            self.calculation_type_description.value = self.dft_u_v_description

    def _toggle_projector_type_description(self):
        if self._model.projector_type == 'atomic':
            self.projector_type_description.value = self.atomic_description
        else:
            self.projector_type_description.value = self.ortho_atomic_description

    def _update_qpoints_mesh(self):
        if not (self.rendered and self._model.has_structure):
            return
        if self._model.qpoints_distance > 0:
            mesh = create_kpoints_from_distance.process_class._func(
                self._model.input_structure,
                orm.Float(self._model.qpoints_distance),
                orm.Bool(False),
            )
            self.qpoint_mesh.value = f'Mesh {mesh.get_kpoints_mesh()[0]}'
        else:
            self.qpoint_mesh.value = 'Please select a number > 0.0'

    def _update_hubbard_tables(self):
        self._generate_hubbard_u()
        self._generate_hubbard_v()

    def _generate_hubbard_u(self):
        if not self.rendered:
            return

        structure = self._model.input_structure
        if not structure:
            self.hubbard_u.children = []
            return

        header = ipw.HBox(
            children=[
                ipw.HTML('', layout=ipw.Layout(width='30px')),
                ipw.HTML('Type', layout=ipw.Layout(width='50px')),
                ipw.HTML('Manifold', layout=ipw.Layout(width='80px')),
                ipw.HTML('U value (eV)', layout=ipw.Layout(width='80px')),
            ]
        )
        rows = [header]

        self._hubbard_u_map = {}

        for kind in structure.kinds:
            kind_name = kind.name
            checkbox = ipw.Checkbox(
                False,
                indent=False,
                layout=ipw.Layout(width='30px'),
            )
            manifold = ipw.Text('', layout=ipw.Layout(width='80px'))
            u_val = ipw.FloatText(0.0, layout=ipw.Layout(width='80px'))

            row = ipw.HBox(
                children=[
                    checkbox,
                    ipw.HTML(kind_name, layout=ipw.Layout(width='50px')),
                    manifold,
                    u_val,
                ]
            )
            rows.append(row)

            def on_change(_):
                self._assemble_hubbard_u()

            checkbox.observe(on_change, 'value')
            manifold.observe(on_change, 'value')
            u_val.observe(on_change, 'value')

            self._hubbard_u_map[kind_name] = (checkbox, manifold, u_val)

        self.hubbard_u.children = [
            self.Hubbard_U_title,
            *rows,
        ]

    def _generate_hubbard_v(self):
        if not self.rendered:
            return

        structure = self._model.input_structure
        if not structure or self._model.calculation_type != 'DFT+U+V':
            self.hubbard_v.children = []
            return

        header = ipw.HBox(
            children=[
                ipw.HTML('', layout=ipw.Layout(width='30px')),
                ipw.HTML('Type 1', layout=ipw.Layout(width='50px')),
                ipw.HTML('Manifold 1', layout=ipw.Layout(width='80px')),
                ipw.HTML('', layout=ipw.Layout(width='4px')),
                ipw.HTML('Type 2', layout=ipw.Layout(width='50px')),
                ipw.HTML('Manifold 2', layout=ipw.Layout(width='80px')),
                ipw.HTML('V value (eV)', layout=ipw.Layout(width='80px')),
            ]
        )
        rows = [header]

        self._hubbard_v_map = {}

        kind_names = [k.name for k in structure.kinds]
        for i in range(len(kind_names)):
            for j in range(i + 1, len(kind_names)):
                kind_name1 = kind_names[i]
                kind_name2 = kind_names[j]
                checkbox = ipw.Checkbox(
                    False,
                    indent=False,
                    layout=ipw.Layout(width='30px'),
                )
                manifold1 = ipw.Text('', layout=ipw.Layout(width='80px'))
                manifold2 = ipw.Text('', layout=ipw.Layout(width='80px'))
                v_val = ipw.FloatText(0.0, layout=ipw.Layout(width='80px'))

                row = ipw.HBox(
                    children=[
                        checkbox,
                        ipw.HTML(kind_name1, layout=ipw.Layout(width='50px')),
                        manifold1,
                        ipw.HTML('', layout=ipw.Layout(width='4px')),
                        ipw.HTML(kind_name2, layout=ipw.Layout(width='50px')),
                        manifold2,
                        v_val,
                    ]
                )
                rows.append(row)

                def on_change(_):
                    self._assemble_hubbard_v()

                for widget in (checkbox, manifold1, manifold2, v_val):
                    widget.observe(on_change, 'value')

                self._hubbard_v_map[(kind_name1, kind_name2)] = (
                    checkbox,
                    manifold1,
                    manifold2,
                    v_val,
                )

        self.hubbard_v.children = [
            self.Hubbard_V_title,
            *rows,
        ]

    def _assemble_hubbard_u(self):
        """Rebuild the model.hubbard_u from the checkboxes/text fields."""
        new_list = []
        for kind_name, (
            checkbox,
            manifold,
            u_val,
        ) in self._hubbard_u_map.items():
            if checkbox.value:
                new_list.append([kind_name, manifold.value, max(u_val.value, 1e-10)])
        self._model.hubbard_u = new_list

    def _assemble_hubbard_v(self):
        """Rebuild the model.hubbard_v from checkboxes/text fields."""
        new_list = []
        for (
            kind_name1,
            kind_name2,
        ), (
            checkbox,
            manifold1,
            manifold2,
            v_val,
        ) in self._hubbard_v_map.items():
            if checkbox.value:
                new_list.append(
                    [
                        kind_name1,
                        manifold1.value,
                        kind_name2,
                        manifold2.value,
                        max(v_val.value, 1e-10),
                    ]
                )
        self._model.hubbard_v = new_list

# hp_results_panel.py

import ipywidgets as ipw
import numpy as np
from aiidalab_qe.common.panel import ResultsPanel
from weas_widget import WeasWidget

# Suppose you have your own custom table widget:
from table_widget import TableWidget
from .model import HpResultsModel

class HpResultsPanel(ResultsPanel[HpResultsModel]):
    """The 'View/Controller' for displaying HP results.
    """

    def _render(self):
        self._model.fetch_result()
        self.hubbard_structure = self._model.hubbard_structure

        self.result_table = TableWidget()
        self.result_table.from_data(
            self._model.table_data['data'],
            columns=self._model.table_data['columns']
        )
        self.result_table.observe(self.on_single_row_select, 'selectedRowId')

        guiConfig = {
            'components': {
                'enabled': True,
                'atomsControl': True,
                'buttons': True},
            'buttons': {
                'enabled': True,
                'fullscreen': True,
                'download': True,
                'measurement': True,
            },
        }
        self.structure_view = WeasWidget(guiConfig=guiConfig)
        self.structure_view_ready = False

        table_help = ipw.HTML(
            """
            <div style='margin: 10px 0;'>
                <h4 style='margin-bottom: 5px; color: #3178C6;'>Result</h4>
            </div>
            """,
            layout=ipw.Layout(margin='0 0 20px 0'),
        )
        structure_help = ipw.HTML(
            """
            <div style='margin: 10px 0;'>
                <h4 style='margin-bottom: 5px; color: #3178C6;'>Structure</h4>
                <p style='margin: 5px 0; font-size: 14px;'>
                    Click on the row above to highlight the specific atoms pair
                    for which the inter-site Hubbard V is being calculated.
                </p>
                <p style='margin: 5px 0; font-size: 14px; color: #555;'>
                    <i>Note:</i> The index in the structure view is one smaller
                    than the value in the table.
                </p>
            </div>
            """,
            layout=ipw.Layout(margin='0 0 20px 0'),
        )

        self._update_structure(self.hubbard_structure)
        self.output = ipw.HTML('HP results are ready.')

        self.children = [
            ipw.VBox(
                children=[
                    ipw.VBox([table_help, self.result_table]),
                    ipw.VBox([structure_help, self.structure_view]),
                    self.output,
                ],
                layout=ipw.Layout(justify_content='space-between', margin='10px'),
            )
        ]

        self.rendered = True

    def on_single_row_select(self, change):
        """Highlight the corresponding atoms in the 3D viewer.
        """
        if change['new'] is not None:
            row_index = int(change['new'])
            atom_index_i = self.result_table.data[row_index]['atom_index_i']
            atom_index_j = self.result_table.data[row_index]['atom_index_j']
            self.structure_view.avr.selected_atoms_indices = [atom_index_i - 1, atom_index_j - 1]

            # Reposition the camera:
            self.structure_view.camera.look_at = self.hubbard_structure.sites[
                atom_index_i - 1
            ].position

            # If this is the first time, trigger a resize event in the viewer
            if not self.structure_view_ready:
                self.structure_view._widget.send_js_task(
                    {'name': 'tjs.onWindowResize', 'kwargs': {}}
                )
                self.structure_view._widget.send_js_task(
                    {
                        'name': 'tjs.updateCameraAndControls',
                        'kwargs': {'direction': [0, -100, 0]},
                    }
                )
                self.structure_view_ready = True

    def _update_structure(self, hubbard_structure):
        """
        Build a large supercell around the original structure for
        better visualization, and load it into the 3D viewer.
        """
        atoms0 = hubbard_structure.get_ase()
        atoms = atoms0.copy()
        # Translate 1 unit cell in +x,+y,+z
        atoms.translate(np.dot([1, 1, 1], atoms.cell))

        # Now replicate the structure in a 3x3x3 grid:
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    atoms_copy = atoms0.copy()
                    atoms_copy.translate(np.dot([i + 1, j + 1, k + 1], atoms0.cell))
                    atoms.extend(atoms_copy)

        # Expand the cell to 3× the original in each lattice direction
        atoms.cell = np.array([3 * atoms.cell[c] for c in range(3)])

        self.structure_view.from_ase(atoms)
        self.structure_view.avr.model_style = 1
        self.structure_view.avr.color_type = 'VESTA'
        self.structure_view.avr.atom_label_type = 'Index'

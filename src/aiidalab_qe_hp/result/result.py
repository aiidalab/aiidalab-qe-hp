# hp_results_panel.py

import ipywidgets as ipw
import numpy as np
from aiidalab_qe.common.panel import ResultsPanel
from weas_widget import WeasWidget

# Suppose you have your own custom table widget:
from table_widget import TableWidget
from .model import HpResultsModel

class HpResultsPanel(ResultsPanel[HpResultsModel]):
    """
    The 'View/Controller' for displaying HPC results.

    In MVC terms:
    - This panel is the View + minimal Controller logic.
    - The HPC data (structure, table rows) is in HpResultsModel.
    """

    def _render(self):
        """
        Render the HPC result panel UI elements. Called automatically
        when the parent sets .render() or displays the widget.
        """
        # 1) Fetch the HPC results from the model
        self._model.fetch_result()
        # The HPC structure and table_data are now stored in the model traitlets.
        self.hubbard_structure = self._model.hubbard_structure

        # 2) Build the table widget
        self.result_table = TableWidget()
        self.result_table.data = self._model.table_data
        self.result_table.observe(self._on_table_row_clicked, 'row_index')

        # 3) Build the 3D structure viewer
        guiConfig = {
            'enabled': True,
            'components': {'atomsControl': True, 'buttons': True},
            'buttons': {
                'fullscreen': True,
                'download': True,
                'measurement': True,
            },
        }
        self.structure_view = WeasWidget(guiConfig=guiConfig)
        self.structure_view_ready = False

        # 4) Build help text for table and structure
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

        # 5) Build and display the supercell in the viewer
        self._update_structure(self.hubbard_structure)

        # 6) Arrange everything
        self.children = [
            ipw.VBox(
                children=[
                    ipw.VBox([table_help, self.result_table]),
                    ipw.VBox([structure_help, self.structure_view]),
                ],
                layout=ipw.Layout(justify_content='space-between', margin='10px'),
            )
        ]

        self.rendered = True

    def _on_table_row_clicked(self, change):
        """
        Callback when the user selects a row in the HPC results table,
        so we highlight the corresponding atoms in the 3D viewer.
        """
        if change['new'] is not None:
            row_index = change['new']
            # HPC table_data: row 0 is a header, so actual data starts at row 1
            # The columns 3 and 4 are "Index (I)" and "Index (J)"
            # We want the newly selected row, plus 1 offset for the header
            selected_atoms_indices = [
                idx - 1 for idx in self.result_table.data[row_index + 1][3:5]
            ]
            self.structure_view.avr.selected_atoms_indices = selected_atoms_indices

            # Reposition the camera:
            self.structure_view.camera.look_at = self.hubbard_structure.sites[
                selected_atoms_indices[0]
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

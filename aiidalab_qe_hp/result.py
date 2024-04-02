import ipywidgets as ipw
from aiidalab_qe.common.panel import ResultPanel


class Result(ResultPanel):
    title = "HP"
    workchain_labels = ["hp"]

    def __init__(self, node=None, **kwargs):
        super().__init__(node=node, **kwargs)
        self.summary_view = ipw.HTML()

    def _update_view(self):
        if "one_shot" in self.node.inputs.hp:
            hubbard_structure = self.node.outputs.hp.one_shot.hubbard_structure
        else:
            hubbard_structure = self.node.outputs.hp.scf_hubbard.hubbard_structure
        self._generate_table(hubbard_structure)
        self.children = [
            ipw.HBox(
                children=[self.summary_view],
                layout=ipw.Layout(justify_content="space-between", margin="10px"),
            ),
        ]

    def _generate_table(self, hubbard_structure):
        # Start of the HTML string for the table
        html_str = """<div class="custom-table" style="padding-top: 0px; padding-bottom: 0px">
                    <h4>Habbard parameter table</h4>
                    <style>
                        .custom-table table, .custom-table th, .custom-table td {
                            border: 1px solid black;
                            border-collapse: collapse;
                            text-align: left;
                            padding: 8px;
                        }
                        .custom-table th, .custom-table td {
                            min-width: 60px;
                            word-wrap: break-word;
                        }
                        .custom-table table {
                            width: 100%;
                            font-size: 1.2em;
                        }
                    </style>
                    <table>
                        <tr>
                            <th>Kind</th>
                            <th>Manifold</th>
                            <th>Neighbour kind</th>
                            <th>Neighbour manifold</th>
                            <th>Value (eV)</th>
                            <th>Hubbard type</th>
                        </tr>"""

        hubbard = hubbard_structure.hubbard.dict()["parameters"]
        for site in hubbard:
            kind = hubbard_structure.sites[site["atom_index"]].kind_name
            neighbour_kind = hubbard_structure.sites[site["neighbour_index"]].kind_name
            html_str += f"""<tr>
                                <td>{kind}</td>
                                <td>{site['atom_manifold']}</td>
                                <td>{neighbour_kind}</td>
                                <td>{site['neighbour_manifold']}</td>
                                <td>{site['value']}</td>
                                <td>{site['hubbard_type']}</td>
                            </tr>"""

        # Closing the table and div tags
        html_str += "</table></div>"
        self.summary_view = ipw.HTML(html_str)

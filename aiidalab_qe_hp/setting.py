# -*- coding: utf-8 -*-
"""Panel for HP plugin.

"""
import ipywidgets as ipw
import traitlets as tl
from aiida.orm import StructureData

from aiidalab_qe.common.panel import Panel


class Setting(Panel):
    title = "HP Settings"
    identifier = "hp"
    input_structure = tl.Instance(StructureData, allow_none=True)
    protocol = tl.Unicode(allow_none=True)

    Hubbard_U_title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Hubbard U</h4></div>"""
    )
    Hubbard_V_title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Hubbard V</h4></div>"""
    )

    def __init__(self, **kwargs):
        # method to calculate the Hubbard U and V values: One-shot or Self-consistent
        self.method = ipw.Dropdown(
            options=["one-shot", "self-consistent"],
            value="one-shot",
            description="Method:",
            disabled=False,
            style={"description_width": "initial"},
        )
        # type of calculation: DFT+U (default) or DFT+U+V
        self.calculation_type = ipw.Dropdown(
            options=["DFT+U", "DFT+U+V"],
            value="DFT+U",
            description="Calculation type:",
            disabled=False,
            style={"description_width": "initial"},
        )
        # type of Hubbard projectors: atomic or ortho-atomic (default)
        self.projector_type = ipw.Dropdown(
            options=["atomic", "ortho-atomic"],
            value="ortho-atomic",
            description="Projector type:",
            disabled=False,
            style={"description_width": "initial"},
        )

        self.hubbard_u = ipw.VBox()
        self.hubbard_v = ipw.VBox()
        self.qpoints_distance = ipw.FloatText(
            value=1000,
            description="qpoints_distance:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.parallelize_atoms = ipw.Checkbox(
            description="parallelize_atoms: ",
            indent=False,
            value=True,
        )
        self.parallelize_qpoints = ipw.Checkbox(
            description="parallelize_qpoints: ",
            indent=False,
            value=True,
        )

        self.calculation_type.observe(self._generate_hubbard_table, names="value")

        self.children = [
            self.method,
            self.calculation_type,
            self.projector_type,
            self.qpoints_distance,
            self.parallelize_atoms,
            self.parallelize_qpoints,
            self.Hubbard_U_title,
            self.hubbard_u,
            self.Hubbard_V_title,
            self.hubbard_v,
        ]
        super().__init__(**kwargs)

    def get_panel_value(self):
        """Return a dictionary with the input parameters for the plugin."""
        hubbard_u = [
            [data[1].value, data[2].value, data[3].value]
            for data in self.hubbard_u_map.values()
            if data[0].value
        ]
        hubbard_v = [
            [data[1].value, data[2].value, data[3].value, data[4].value, data[5].value]
            for data in self.hubbard_v_map.values()
            if data[0].value
        ]
        parameters = {
            "method": self.method.value,
            "qpoints_distance": self.qpoints_distance.value,
            "parallelize_atoms": self.parallelize_atoms.value,
            "parallelize_qpoints": self.parallelize_qpoints.value,
            "calculation_type": self.calculation_type.value,
            "projector_type": self.projector_type.value,
            "hubbard_u": hubbard_u,
            "hubbard_v": hubbard_v,
        }
        return parameters

    def set_panel_value(self, input_dict):
        """Load a dictionary with the input parameters for the plugin."""
        self.method.value = input_dict.get("method", "one-shot")
        self.qpoints_distance.value = input_dict.get("qpoints_distance", 10000)
        self.parallelize_atoms.value = input_dict.get("parallelize_atoms", True)
        self.parallelize_qpoints.value = input_dict.get("parallelize_qpoints", True)
        self.calculation_type.value = input_dict.get("calculation_type", "DFT+U")
        self.projector_type.value = input_dict.get("projector_type", "ortho-atomic")
        hubbard_u = input_dict.get("hubbard_u", [])
        for data in hubbard_u:
            checkbox, _, manifold, U = self.hubbard_u_map[data[0]]
            checkbox.value = True
            manifold.value = data[1]
            U.value = data[2]

        hubbard_v = input_dict.get("hubbard_v", [])
        for data in hubbard_v:
            checkbox, _, manifold1, _, manifold2, V = self.hubbard_v_map[
                (data[0], data[2])
            ]
            checkbox.value = True
            manifold1.value = data[1]
            manifold2.value = data[3]
            V.value = data[4]

    @tl.observe("input_structure")
    def _update_structure(self, _=None):
        self._generate_hubbard_table()

    def _generate_hubbard_table(self, _=None):
        self._generate_u()
        self._generate_v()

    def _generate_u(self):
        """Generate a table with the U values for the selected structure.
        One row for each kind in the structure.
        Each row includes: kind symbol, manifold, U value."""
        print("generate_u")
        if self.input_structure is None:
            return
        structure = self.input_structure
        kind_list = [Kind.symbol for Kind in structure.kinds]
        rows = []
        self.hubbard_u_map = {}
        for kind_symbol in kind_list:
            checkbox = ipw.Checkbox(
                value=False,
                description="",
                indent=False,
                layout=ipw.Layout(width="20px"),
            )
            symbol = ipw.HTML(value=kind_symbol)
            manifold = ipw.Text(value="", placeholder="", description="Manifold")
            u = ipw.FloatText(value=1e-10, description="U", min=1e10, max=100)
            rows.append(ipw.HBox([checkbox, symbol, manifold, u]))
            self.hubbard_u_map[kind_symbol] = [checkbox, symbol, manifold, u]
        self.hubbard_u.children = rows

    def _generate_v(self):
        """Generate a table with the V values for the selected structure.
        One row for each kind pair in the structure.
        Each row includes: kind symbol1, manifold1, kind symbol2, manifold2, V value."""
        print("generate_v")
        self.hubbard_v.children = []
        self.hubbard_v_map = {}
        if self.input_structure is None or self.calculation_type.value != "DFT+U+V":
            print("input_structure is: ", self.input_structure)
            print("calculation_type is: ", self.calculation_type.value)
            print("skip generate_v")
            return
        structure = self.input_structure
        kind_list = [Kind.symbol for Kind in structure.kinds]
        rows = []
        for i in range(len(kind_list)):
            for j in range(i + 1, len(kind_list)):
                kind_symbol1 = kind_list[i]
                kind_symbol2 = kind_list[j]
                checkbox = ipw.Checkbox(
                    value=False,
                    description="",
                    indent=False,
                    layout=ipw.Layout(width="20px"),
                )
                symbol1 = ipw.HTML(value=kind_symbol1)
                manifold1 = ipw.Text(value="", placeholder="", description="Manifold")
                symbol2 = ipw.HTML(value=kind_symbol2)
                manifold2 = ipw.Text(value="", placeholder="", description="Manifold")
                v = ipw.FloatText(value=1e-10, description="V", min=1e10, max=100)
                rows.append(
                    ipw.HBox([checkbox, symbol1, manifold1, symbol2, manifold2, v])
                )
                self.hubbard_v_map[(kind_symbol1, kind_symbol2)] = [
                    checkbox,
                    symbol1,
                    manifold1,
                    symbol2,
                    manifold2,
                    v,
                ]
        print("hubbard_v_map: ", self.hubbard_v_map)
        self.hubbard_v.children = rows

    def reset(self):
        """Reset the panel to its initial state."""
        self.input_structure = None
        self.hubbard_u.children = []

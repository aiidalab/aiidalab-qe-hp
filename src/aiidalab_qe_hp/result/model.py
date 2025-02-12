# hp_results_model.py

import time
import numpy as np
import traitlets as tl
from aiida import orm
from aiida_quantumespresso.utils.hubbard import get_supercell_atomic_index
from aiidalab_qe.common.panel import ResultsModel


class HpResultsModel(ResultsModel):
    """Traitlets-based model holding the HP calculation results."""

    title = 'HP'
    identifier = 'hp'

    # The final structure containing the Hubbard parameters.
    hubbard_structure = tl.Instance(orm.StructureData, allow_none=True)

    # Data for displaying in the HPC result table, i.e. a list of rows.
    # Typically the first row might be a header, the rest are data.
    table_data = tl.List(allow_none=True, default_value=[])

    def fetch_result(self):
        """
        Fetch the HPC results from the process node and populate the traitlets.
        """
        process = self.fetch_process_node()
        # The original code checks 'relax' in the inputs to decide:
        if 'relax' not in process.inputs.hp:
            self.hubbard_structure = process.outputs.hp.hubbard_structure
        else:
            self.hubbard_structure = process.outputs.hp.hubbard_structure

        # Build the table data from the HPC results:
        self.table_data = self._generate_table_data(self.hubbard_structure)

    def _generate_table_data(self, structure: orm.StructureData) -> list:
        """
        Build a 2D list (header + rows) describing the final Hubbard parameters.
        """
        data = [
            [
                'Hubbard parameters',
                'Atoms (I)',
                'Atoms (J)',
                'Index (I)',
                'Index (J)',
                'Value (eV)',
                'Translation vector',
                'Distance (Å)',
            ]
        ]

        hubbard = structure.hubbard.dict()['parameters']
        natoms = len(structure.sites)

        for site in hubbard:
            kind_i = structure.sites[site['atom_index']].kind_name
            kind_j = structure.sites[site['neighbour_index']].kind_name

            index_i = site['atom_index'] + 1
            # Use the utility to get the supercell index:
            index_j = (
                get_supercell_atomic_index(
                    site['neighbour_index'], natoms, site['translation']
                )
                + 1
            )

            # Compute distance including the supercell translation:
            distance = np.linalg.norm(
                structure.sites[site['neighbour_index']].position
                + np.dot(site['translation'], structure.cell)
                - structure.sites[site['atom_index']].position
            )
            distance = round(distance, 2)

            value = round(site['value'], 2)

            data.append(
                [
                    site['hubbard_type'],
                    f"{kind_i}-{site['atom_manifold']}",
                    f"{kind_j}-{site['neighbour_manifold']}",
                    index_i,
                    index_j,
                    value,
                    site['translation'],
                    distance,
                ]
            )

        return data

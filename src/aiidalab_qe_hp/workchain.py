from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_hubbard.workflows.hubbard import SelfConsistentHubbardWorkChain
from aiida import orm
from aiidalab_qe.utils import (
    enable_pencil_decomposition,
    set_component_resources,
)


PROTOCOL_MAP_U = {'fast': 1.0, 'balanced': 0.5, 'stringent': 0.1}

PROTOCOL_MAP_V = {
    'fast': 0.5,
    'balanced': 0.1,
    'stringent': 0.02,
}


def check_codes(pw_code, hp_code):
    """Check that the codes are installed on the same computer."""
    if (
        not any(
            [
                pw_code is None,
                hp_code is None,
            ]
        )
        and len(
            set(
                (
                    pw_code.computer.pk,
                    hp_code.computer.pk,
                )
            )
        )
        != 1
    ):
        raise ValueError(
            'All selected codes must be installed on the same computer. This is because the '
            'HP calculations rely on large files that are not retrieved by AiiDA.'
        )


def update_resources(builder, codes):
    set_component_resources(builder.relax.base.pw, codes.get('pw'))
    set_component_resources(builder.scf.pw, codes.get('pw'))
    set_component_resources(builder.hubbard.hp, codes.get('hp'))


def get_builder(codes, structure, parameters, **kwargs):
    pw_code = codes.get('pw')['code']
    hp_code = codes.get('hp')['code']
    check_codes(pw_code, hp_code)
    protocol = parameters['workchain']['protocol']
    # generate Hubbard structure
    hubbard_structure = HubbardStructureData.from_structure(structure)
    hubbard_u = parameters['hp'].pop('hubbard_u')
    hubbard_v = parameters['hp'].pop('hubbard_v')
    for data in hubbard_u:
        hubbard_structure.initialize_onsites_hubbard(*data)
    for data in hubbard_v:
        hubbard_structure.initialize_intersites_hubbard(*data)
    # print(HubbardUtils(hubbard_structure).get_hubbard_card())
    hubbard = parameters.get('hp', {})
    parallelize_atoms = hubbard.get('parallelize_atoms', False)
    parallelize_qpoints = hubbard.get('parallelize_qpoints', False)
    overrides = {
        'tolerance_onsite': orm.Float(PROTOCOL_MAP_U[protocol]),
        'tolerance_intersite': orm.Float(PROTOCOL_MAP_V[protocol]),
        'hubbard': {
            'parallelize_atoms': orm.Bool(parallelize_atoms),
            'parallelize_qpoints': orm.Bool(parallelize_qpoints),
            'qpoints_distance': orm.Float(hubbard.get('qpoints_distance', 1)),
        },
    }
    builder = SelfConsistentHubbardWorkChain.get_builder_from_protocol(
        pw_code=pw_code,
        hp_code=hp_code,  # modify here if you downloaded the notebook
        hubbard_structure=hubbard_structure,
        protocol=protocol,
        overrides=overrides,
        electronic_type=ElectronicType(parameters['workchain']['electronic_type']),
        spin_type=SpinType(parameters['workchain']['spin_type']),
        initial_magnetic_moments=parameters['advanced']['initial_magnetic_moments'],
        **kwargs,
    )
    # update resources
    update_resources(builder, codes)
    method = parameters['hp'].pop('method')
    if method == 'one-shot':
        builder.max_iterations = orm.Int(1)
        builder.meta_convergence = orm.Bool(False)
        builder.pop('relax', None)

    builder.pop('clean_workdir', None)

    return builder


workchain_and_builder = {
    'workchain': SelfConsistentHubbardWorkChain,
    'exclude': ('structure',),
    'get_builder': get_builder,
}

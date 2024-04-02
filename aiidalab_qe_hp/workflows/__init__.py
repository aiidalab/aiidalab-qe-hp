# -*- coding: utf-8 -*-
"""Work chain to run a Quantum ESPRESSO hp.x calculation."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, if_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_quantumespresso_hp.workflows.hubbard import SelfConsistentHubbardWorkChain
from aiida_quantumespresso_hp.workflows.hp.main import HpWorkChain
from aiida_quantumespresso.utils.mapping import prepare_process_inputs



PROTOCOL_MAP_U = {
    "fast": 1.0,
    "moderate": 0.5,
    "precise": 0.1
}

PROTOCOL_MAP_V = {
    "fast": 0.5,
    "moderate": 0.1,
    "precise": 0.02,
}



class QeappHpWorkChain(WorkChain, ProtocolMixin):
    """Work chain to run a Quantum ESPRESSO hp.x calculation.
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='scf',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` of the `scf` calculation.',
                'required': False,
                'populate_defaults': False,
            }
        )
        spec.expose_inputs(
            HpWorkChain,
            exclude=('clean_workdir', 'hp.parent_scf'),
            namespace='one_shot',
            namespace_options={
                'help': 'Inputs for the `HpWorkChain` of the `one-shot` calculation.',
                'required': False,
                'populate_defaults': False,
            }
        )
        spec.expose_inputs(
            SelfConsistentHubbardWorkChain,
            exclude=('clean_workdir', ),
            namespace='scf_hubbard',
            namespace_options={
                'help': 'Inputs for the `HpWorkChain` of the `one-shot` calculation.',
                'required': False,
                'populate_defaults': False,
            }
        )
        spec.input("method", valid_type=orm.Str, help="The method to use to compute the Hubbard parameters.",
                   default=lambda: orm.Str("one-shot"))
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')

        spec.outline(
            if_(cls.method_is_one_shot)(
                if_(cls.should_run_scf)(
                    cls.run_scf,
                    cls.inspect_scf,
                ),
                cls.run_one_shot,
            ).else_(
                cls.run_self_consistent,
            ),
            cls.results,
        )
        spec.expose_outputs(HpWorkChain, namespace='one_shot', namespace_options={'required': False})
        spec.expose_outputs(SelfConsistentHubbardWorkChain, namespace='scf_hubbard', namespace_options={'required': False})
        spec.exit_code(200, 'ERROR_SUB_PROCESS_FAILED_SCF', message='the scf sub process failed')
        spec.exit_code(201, 'ERROR_SUB_PROCESS_FAILED_ONE_SHOT', message='the one-shot sub process failed')
        spec.exit_code(202, 'ERROR_SUB_PROCESS_FAILED_SELF_CONSISTENT', message='the self-consistent sub process failed')


    @classmethod
    def get_builder_from_protocol(cls, pw_code, hp_code, hubbard_structure=None,
                                  protocol=None, method="one-shot",
                                  parallelize_atoms=False,
                                  parallelize_qpoints=False,
                                  parent_scf_folder=None,
                                  qpoints_distance=1000,
                                  overrides=None,
                                  options=None,
                                  **kwargs):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.hp`` plugin.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param parent_scf_folder: the parent ``RemoteData`` of the respective SCF calcualtion.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        builder = cls.get_builder()
        builder.method = orm.Str(method)
        overrides = overrides or {}

        if method == "one-shot":
            overrides_one_shot = overrides.get("one_shot", {})
            overrides_one_shot.update({
                "parallelize_atoms":parallelize_atoms,
                "parallelize_qpoints":parallelize_qpoints,
                "qpoints_distance": qpoints_distance,
                })
            overrides_one_shot.setdefault("hp", {})
            overrides_one_shot["hp"].setdefault("hubbard_structure", hubbard_structure)
            builder_one_shot = HpWorkChain.get_builder_from_protocol(
                code=hp_code,
                protocol="fast",
                overrides=overrides_one_shot,
            )
            if parent_scf_folder:
                builder_one_shot.hp.parent_scf = parent_scf_folder
            else:
                builder_scf = PwBaseWorkChain.get_builder_from_protocol(
                    pw_code, hubbard_structure, protocol, overrides=overrides.get('scf', None),
                    **kwargs
                )
                builder.scf = builder_scf
            builder.one_shot = builder_one_shot
            builder.pop("scf_hubbard", None)
        elif method == "self-consistent":
            overrides_scf_hubbard = overrides.get("scf_hubbard", {})
            overrides_scf_hubbard.update({
                "tolerance_onsite": PROTOCOL_MAP_U[protocol],
                "tolerance_intersite": PROTOCOL_MAP_V[protocol],
            })
            overrides_scf_hubbard.setdefault("hubbard", {})
            overrides_scf_hubbard["hubbard"].update({
                "parallelize_atoms": parallelize_atoms,
                "parallelize_qpoints": parallelize_qpoints,
                "qpoints_distance": qpoints_distance,
            })
            builder_scf_hubbard = SelfConsistentHubbardWorkChain.get_builder_from_protocol(
                pw_code=pw_code,
                hp_code=hp_code,
                hubbard_structure=hubbard_structure,
                protocol=protocol,
                overrides=overrides_scf_hubbard,
                **kwargs,
            )
            builder.scf_hubbard = builder_scf_hubbard
            builder.pop("one_shot", None)
            builder.pop("scf", None)

        builder.pop("clean_workdir", None)

        return builder

    def method_is_one_shot(self):
        """Return whether the work chain should run a one-shot calculation."""
        return self.inputs.method.value == "one-shot"

    def should_run_scf(self):
        """Return whether the work chain should run an SCF calculation."""
        return 'scf' in self.inputs

    def run_scf(self):
        """Run an SCF calculation, to generate the wavefunction."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, 'scf'))

        inputs.metadata.call_link_label = 'scf'
        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

        future = self.submit(PwBaseWorkChain, **inputs)

        self.report(f'launching SCF PwBaseWorkChain<{future.pk}>')

        return ToContext(workchain_scf=future)

    def inspect_scf(self):
        """Verify that the SCF calculation finished successfully."""
        workchain = self.ctx.workchain_scf
        if not workchain.is_finished_ok:
            self.report(f'SCF PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.scf_parent_folder = workchain.outputs.remote_folder

    def run_one_shot(self):
        """Run the HpWorkChain restarting from the last completed scf calculation."""

        inputs = AttributeDict(self.exposed_inputs(HpWorkChain, namespace='one_shot'))
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.hp.parent_scf = self.ctx.scf_parent_folder
        inputs.metadata.call_link_label = f"one_shot"

        future = self.submit(HpWorkChain, **inputs)

        self.report(f'launching HpWorkChain<{future.pk}> for the one_shot calculation')
        return ToContext(workchain_one_shot=future)

    def run_self_consistent(self):
        """Run the SelfConsistentHubbardWorkChain."""
        inputs = AttributeDict(self.exposed_inputs(SelfConsistentHubbardWorkChain, namespace='scf_hubbard'))
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.metadata.call_link_label = 'self_consistent'

        future = self.submit(SelfConsistentHubbardWorkChain, **inputs)

        self.report(f'launching SelfConsistentHubbardWorkChain<{future.pk}> for the self_consistent calculation')
        return ToContext(workchain_scf_hubbard=future)

    def results(self):
        """Retrieve the results from the completed sub workchain."""
        if 'one_shot' in self.inputs:
            self.out_many(self.exposed_outputs(self.ctx.workchain_one_shot, HpWorkChain, namespace='one_shot'))
        if 'scf_hubbard' in self.inputs:
            self.out_many(self.exposed_outputs(self.ctx.workchain_scf_hubbard, SelfConsistentHubbardWorkChain, namespace='scf_hubbard'))

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")

"""Panel for hp plugin."""

from aiidalab_qe.common.code.model import CodeModel, PwCodeModel
from aiidalab_qe.common.panel import (
    PluginResourceSettingsModel,
    PluginResourceSettingsPanel,
)


class ResourceSettingsModel(PluginResourceSettingsModel):
    """Model for the hp code setting plugin."""

    title = 'hp'
    identifier = 'hp'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_models(
            {
                'pw': PwCodeModel(
                    name='pw.x',
                    description='pw.x',
                    default_calc_job_plugin='quantumespresso.pw',
                ),
                'hp': CodeModel(
                    name='hp.x',
                    description='hp.x',
                    default_calc_job_plugin='quantumespresso.hp',
                ),
            }
        )


class ResourceSettingsPanel(
    PluginResourceSettingsPanel[ResourceSettingsModel],
):
    """Panel for configuring the hp plugin."""

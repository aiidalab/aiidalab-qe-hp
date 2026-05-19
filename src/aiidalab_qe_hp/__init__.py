"""AiiDALab plugin for Quantum ESPRESSO Hubbard parameters (HP) calculations."""

from aiidalab_qe.common.panel import PluginOutline

from .model import HpSettingsModel
from .resources import HpResourceSettingsModel, HpResourceSettingsPanel
from .results import HpResultsModel, HpResultsPanel
from .settings import HpSettingsPanel
from .workchain import workchain_and_builder

__version__ = '0.1.5'


class HpPluginOutline(PluginOutline):
    title = 'Hubbard parameter (HP)'
    help = """"""


hp = {
    'outline': HpPluginOutline,
    'configuration': {
        'panel': HpSettingsPanel,
        'model': HpSettingsModel,
    },
    'resources': {
        'panel': HpResourceSettingsPanel,
        'model': HpResourceSettingsModel,
    },
    'workchain': workchain_and_builder,
    'result': {
        'panel': HpResultsPanel,
        'model': HpResultsModel,
    },
}

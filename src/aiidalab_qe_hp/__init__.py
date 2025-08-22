"""Web GUI for Quantum ESPRESSO and hp calculations in AiiDA."""
from aiidalab_qe.common.panel import PluginOutline

from .model import HPSettingsModel
from .resources import ResourceSettingsModel, ResourceSettingsPanel
from .setting import HPSettingsPanel
from .workchain import workchain_and_builder
from .result import HpResultsPanel, HpResultsModel


__version__ = '0.1.3'



class PluginOutline(PluginOutline):
    title = 'Hubbard parameter (HP)'
    help = """"""


hp = {
    'outline': PluginOutline,
    'configuration': {
        'panel': HPSettingsPanel,
        'model': HPSettingsModel,
    },
    'resources': {
        'panel': ResourceSettingsPanel,
        'model': ResourceSettingsModel,
    },
    'workchain': workchain_and_builder,
    'result': {
        'panel': HpResultsPanel,
        'model': HpResultsModel,
    },
}

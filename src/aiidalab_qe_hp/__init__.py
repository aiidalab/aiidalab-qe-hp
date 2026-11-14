"""AiiDALab QE plugin for computing Hubbard parameters (HP)."""

from aiidalab_qe.plugins import QeAppPlugin

__version__ = '0.1.8'


class HpPlugin(QeAppPlugin):
    @property
    def outline(self) -> str:
        return 'Hubbard parameter (HP)'

    def configuration(self):
        from .model import HpSettingsModel
        from .settings import HpSettingsPanel

        return {
            'panel': HpSettingsPanel,
            'model': HpSettingsModel,
        }

    def resources(self):
        from .resources import HpResourceSettingsModel, HpResourceSettingsPanel

        return {
            'panel': HpResourceSettingsPanel,
            'model': HpResourceSettingsModel,
        }

    def workchain(self):
        from .workchain import workchain_and_builder

        return workchain_and_builder

    def result(self):
        from .results import HpResultsModel, HpResultsPanel

        return {
            'panel': HpResultsPanel,
            'model': HpResultsModel,
        }

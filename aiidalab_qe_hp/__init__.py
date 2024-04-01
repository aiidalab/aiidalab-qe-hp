from aiidalab_qe.common.panel import OutlinePanel
from aiidalab_widgets_base import ComputationalResourcesWidget

from .result import Result
from .setting import Setting
from .workchain import workchain_and_builder

__version__ = "0.1.0"


class HpOutline(OutlinePanel):
    title = "Habbard parameter (HP)"
    help = """"""


hp_code = ComputationalResourcesWidget(
    description="hp.x",
    default_calc_job_plugin="quantumespresso.hp",
)


hp = {
    "outline": HpOutline,
    "setting": Setting,
    "code": {"hp": hp_code},
    "result": Result,
    "workchain": workchain_and_builder,
}

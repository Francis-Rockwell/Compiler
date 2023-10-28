from typing import Optional
from utils.label.funclabel import FuncLabel
from utils.tac.temp import Temp

"""
SubroutineInfo: collect some info when selecting instr which will be used in SubroutineEmitter
"""


class SubroutineInfo:
    def __init__(
        self, funcLabel: FuncLabel, more_size: int, params: list[Temp]
    ) -> None:
        self.funcLabel = funcLabel
        self.params = params
        self.more_size = more_size

    def __str__(self) -> str:
        return "funcLabel: {}".format(
            self.funcLabel.name,
        )

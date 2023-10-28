from typing import Any, Optional, Union

from .tacfunc import TACFunc

from .tacvar import TACVar


# A TAC program consists of several TAC functions.
class TACProg:
    def __init__(self, funcs: list[TACFunc], vars: list[TACVar]) -> None:
        self.funcs = funcs
        self.vars = vars

    def printTo(self) -> None:
        for var in self.vars:
            var.printTo()
        for func in self.funcs:
            func.printTo()

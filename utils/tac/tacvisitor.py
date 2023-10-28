from __future__ import annotations

from .tacinstr import *


class TACVisitor:
    def visitOther(self, instr: TACInstr) -> None:
        pass

    def visitAssign(self, instr: Assign) -> None:
        self.visitOther(instr)

    def visitLoadImm4(self, instr: LoadImm4) -> None:
        self.visitOther(instr)

    def visitUnary(self, instr: Unary) -> None:
        self.visitOther(instr)

    def visitBinary(self, instr: Binary) -> None:
        self.visitOther(instr)

    def visitBranch(self, instr: Branch) -> None:
        self.visitOther(instr)

    def visitCondBranch(self, instr: CondBranch) -> None:
        self.visitOther(instr)

    def visitReturn(self, instr: Return) -> None:
        self.visitOther(instr)

    def visitMemo(self, instr: Memo) -> None:
        self.visitOther(instr)

    def visitMark(self, instr: Mark) -> None:
        self.visitOther(instr)

    def visitCall(self, instr: Call) -> None:
        self.visitOther(instr)

    def visitLoadSymbol(self, instr: Load_Symbol) -> None:
        self.visitOther(instr)

    def visitLoad(self, instr: Load) -> None:
        self.visitOther(instr)

    def visitStore(self, instr: Store) -> None:
        self.visitOther(instr)

    def visitAlloc(self, instr: Alloc) -> None:
        self.visitOther(instr)

    def visitGetAddr(self, instr: GetAddr) -> None:
        self.visitOther(instr)

    def visitLoadFromStack(self, instr: LOAD_FROM_STACK) -> None:
        self.visitOther(instr)

    def visitStoreToStack(self, instr: STORE_TO_STACK) -> None:
        self.visitOther(instr)

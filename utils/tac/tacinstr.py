from enum import Enum, auto, unique
from typing import Any, Optional, Union

from utils.label.label import Label
from utils.tac.nativeinstr import NativeInstr
from utils.tac.reg import Reg
from utils.tac.tacop import InstrKind

from .tacop import *
from .tacvisitor import TACVisitor
from .temp import Temp


class TACInstr:
    def __init__(
        self,
        kind: InstrKind,
        dsts: list[Temp],
        srcs: list[Temp],
        label: Optional[Label],
    ) -> None:
        self.kind = kind
        self.dsts = dsts.copy()
        self.srcs = srcs.copy()
        self.label = label

    def getRead(self) -> list[int]:
        return [src.index for src in self.srcs]

    def getWritten(self) -> list[int]:
        return [dst.index for dst in self.dsts]

    def isLabel(self) -> bool:
        return self.kind is InstrKind.LABEL

    def isSequential(self) -> bool:
        return self.kind == InstrKind.SEQ

    def isReturn(self) -> bool:
        return self.kind == InstrKind.RET

    def toNative(self, dstRegs: list[Reg], srcRegs: list[Reg]) -> NativeInstr:
        oldDsts = dstRegs
        oldSrcs = srcRegs
        self.dsts = dstRegs
        self.srcs = srcRegs
        instrString = self.__str__()
        newInstr = NativeInstr(self.kind, dstRegs, srcRegs, self.label, instrString)
        self.dsts = oldDsts
        self.srcs = oldSrcs
        return newInstr

    def accept(self, v: TACVisitor) -> None:
        pass


# Assignment instruction.
class Assign(TACInstr):
    def __init__(self, dst: Temp, src: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [src], None)
        self.dst = dst
        self.src = src

    def __str__(self) -> str:
        return "%s = %s" % (self.dst, self.src)

    def accept(self, v: TACVisitor) -> None:
        v.visitAssign(self)


# Loading an immediate 32-bit constant.
class LoadImm4(TACInstr):
    def __init__(self, dst: Temp, value: int) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.dst = dst
        self.value = value

    def __str__(self) -> str:
        return "%s = %d" % (self.dst, self.value)

    def accept(self, v: TACVisitor) -> None:
        v.visitLoadImm4(self)


# Unary operations.
class Unary(TACInstr):
    def __init__(self, op: TacUnaryOp, dst: Temp, operand: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [operand], None)
        self.op = op
        self.dst = dst
        self.operand = operand

    def __str__(self) -> str:
        return "%s = %s %s" % (
            self.dst,
            str(self.op),
            self.operand,
        )

    def accept(self, v: TACVisitor) -> None:
        v.visitUnary(self)


# Binary Operations.
class Binary(TACInstr):
    def __init__(self, op: TacBinaryOp, dst: Temp, lhs: Temp, rhs: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [lhs, rhs], None)
        self.op = op
        self.dst = dst
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        opStr = {
            TacBinaryOp.ADD: "+",
            TacBinaryOp.SUB: "-",
            TacBinaryOp.MUL: "*",
            TacBinaryOp.DIV: "/",
            TacBinaryOp.MOD: "%",
            TacBinaryOp.LOR: "||",
            TacBinaryOp.LAND: "&&",
            TacBinaryOp.EQ: "==",
            TacBinaryOp.NE: "!=",
            TacBinaryOp.LT: "<",
            TacBinaryOp.GT: ">",
            TacBinaryOp.LE: "<=",
            TacBinaryOp.GE: ">=",
        }[self.op]
        return "%s = (%s %s %s)" % (self.dst, self.lhs, opStr, self.rhs)

    def accept(self, v: TACVisitor) -> None:
        v.visitBinary(self)


# Branching instruction.
class Branch(TACInstr):
    def __init__(self, target: Label) -> None:
        super().__init__(InstrKind.JMP, [], [], target)
        self.target = target

    def __str__(self) -> str:
        return "branch %s" % str(self.target)

    def accept(self, v: TACVisitor) -> None:
        v.visitBranch(self)


# Branching with conditions.
class CondBranch(TACInstr):
    def __init__(self, op: CondBranchOp, cond: Temp, target: Label) -> None:
        super().__init__(InstrKind.COND_JMP, [], [cond], target)
        self.op = op
        self.cond = cond
        self.target = target

    def __str__(self) -> str:
        return "if (%s %s) branch %s" % (
            self.cond,
            "== 0" if self.op == CondBranchOp.BEQ else "!= 0",
            str(self.target),
        )

    def accept(self, v: TACVisitor) -> None:
        v.visitCondBranch(self)


# Return instruction.
class Return(TACInstr):
    def __init__(self, value: Optional[Temp]) -> None:
        if value is None:
            super().__init__(InstrKind.RET, [], [], None)
        else:
            super().__init__(InstrKind.RET, [], [value], None)
        self.value = value

    def __str__(self) -> str:
        return "return" if (self.value is None) else ("return " + str(self.value))

    def accept(self, v: TACVisitor) -> None:
        v.visitReturn(self)


# Annotation (used for debugging).
class Memo(TACInstr):
    def __init__(self, msg: str) -> None:
        super().__init__(InstrKind.SEQ, [], [], None)
        self.msg = msg

    def __str__(self) -> str:
        return "memo '%s'" % self.msg

    def accept(self, v: TACVisitor) -> None:
        v.visitMemo(self)


# Label (function entry or branching target).
class Mark(TACInstr):
    def __init__(self, label: Label) -> None:
        super().__init__(InstrKind.LABEL, [], [], label)

    def __str__(self) -> str:
        return "%s:" % str(self.label)

    def accept(self, v: TACVisitor) -> None:
        v.visitMark(self)


class Call(TACInstr):
    def __init__(self, dst: Temp, src: [Temp], label: Label) -> None:
        super().__init__(InstrKind.Call, [dst], src, label)
        self.dst = dst
        self.srcs = src
        self.label = label

    def __str__(self) -> str:
        args = "("
        for arg in self.srcs[:-1]:
            args = args + str(arg) + ", "
        args = args + ("" if len(self.srcs) == 0 else str(self.srcs[-1])) + ")"

        return str(self.dst) + " = Call " + str(self.label) + args

    def accept(self, v: TACVisitor) -> None:
        v.visitCall(self)


class Load_Symbol(TACInstr):
    def __init__(self, dst: Temp, name: str) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.dst = dst
        self.name = name

    def __str__(self) -> str:
        return str(self.dst) + " = LOAD SYMBOL " + self.name

    def accept(self, v: TACVisitor) -> None:
        v.visitLoadSymbol(self)


class Load(TACInstr):
    def __init__(self, dst: Temp, src: str, offset: int) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.src = src
        self.dst = dst
        self.offset = offset

    def __str__(self) -> str:
        return str(self.dst) + " = LOAD " + str(self.src) + ", " + str(self.offset)

    def accept(self, v: TACVisitor) -> None:
        v.visitLoad(self)


class Store(TACInstr):
    def __init__(self, dst: Temp, src: Temp, offset: int) -> None:
        super().__init__(InstrKind.SEQ, [Temp], [src], None)
        self.src = src
        self.dst = dst
        self.offset = offset

    def __str__(self) -> str:
        return "STORE " + str(self.src) + " " + str(self.dst) + ", " + str(self.offset)

    def accept(self, v: TACVisitor) -> None:
        v.visitStore(self)


class Alloc(TACInstr):
    def __init__(self, size: int) -> None:
        super().__init__(InstrKind.SEQ, [], [], None)
        self.size = size

    def __str__(self) -> str:
        return "ALLOC ON STACK " + str(self.size)

    def accept(self, v: TACVisitor) -> None:
        v.visitAlloc(self)


class GetAddr(TACInstr):
    def __init__(self, dst: Temp) -> None:
        super().__init__(InstrKind.SEQ, [], [], None)
        self.dst = dst

    def __str__(self) -> str:
        return "STORE SP IN " + str(self.dst)

    def accept(self, v: TACVisitor) -> None:
        v.visitGetAddr(self)


class LOAD_FROM_STACK(TACInstr):
    def __init__(self, src: Temp, dst: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [src], None)
        self.dst = dst
        self.src = src

    def __str__(self) -> str:
        return "LOAD FROM " + str(self.src) + " TO " + str(self.dst)

    def accept(self, v: TACVisitor) -> None:
        v.visitLoadFromStack(self)


class STORE_TO_STACK(TACInstr):
    def __init__(self, src: Temp, dst: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [src], None)
        self.src = src
        self.dst = dst

    def __str__(self) -> str:
        return "STORE FROM " + str(self.src) + " TO " + str(self.dst)

    def accept(self, v: TACVisitor) -> None:
        v.visitStoreToStack(self)

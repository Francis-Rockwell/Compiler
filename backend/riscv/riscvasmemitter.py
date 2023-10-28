from typing import Sequence, Tuple

from backend.asmemitter import AsmEmitter
from utils.error import IllegalArgumentException
from utils.label.blocklabel import BlockLabel
from utils.label.funclabel import FuncLabel
from utils.label.label import Label, LabelKind
from utils.riscv import Riscv, RvBinaryOp, RvUnaryOp
from utils.tac.reg import Reg
from utils.tac.tacfunc import TACFunc
from utils.tac.tacvar import TACVar
from utils.tac.tacinstr import *
from utils.tac.tacinstr import Alloc, GetAddr, Call
from utils.tac.tacvisitor import TACVisitor

from ..subroutineemitter import SubroutineEmitter
from ..subroutineinfo import SubroutineInfo

"""
RiscvAsmEmitter: an AsmEmitter for RiscV
"""


class RiscvAsmEmitter(AsmEmitter):
    def __init__(
        self,
        global_vars: list[TACVar],
        allocatableRegs: list[Reg],
        callerSaveRegs: list[Reg],
    ) -> None:
        super().__init__(allocatableRegs, callerSaveRegs)

        # the start of the asm code
        # int step10, you need to add the declaration of global var here
        self.printer.println(".bss")
        for var in global_vars:
            if not bool(var.init):
                self.printer.println(".global " + var.name)
                self.printer.printLabel(Label(LabelKind.VAR, var.name))
                self.printer.println(".space " + str(var.size))
                self.printer.println("")
        self.printer.println("")

        self.printer.println(".data")
        for var in global_vars:
            if bool(var.init):
                self.printer.println(".global " + var.name)
                self.printer.printLabel(Label(LabelKind.VAR, var.name))
                for init in var.init:
                    self.printer.println(".word " + str(init))
                for i in range(int(var.size / 4) - len(var.init)):
                    self.printer.println(".word " + str(0))
                self.printer.println("")
        self.printer.println("")

        self.printer.println(".text")
        self.printer.println(".global main")
        self.printer.println("")

        self.printer.printLabel(Label(FuncLabel, "fill_n"))

        self.printer.printComment("start of prologue")

        self.printer.println("addi sp, sp, -52")
        self.printer.println("sw ra, 44(sp)")

        self.printer.printComment("end of prologue")
        self.printer.println("")

        self.printer.printComment("start of body")

        self.printer.println("li t0, 0")
        self.printer.println("mv t1, t0")
        self.printer.println("sw t1, 48(sp)")

        self.printer.printLabel(Label(BlockLabel, "_fill_n_L1"))

        self.printer.println("lw t0, 48(sp)")
        self.printer.println("lw t1, 56(sp)")
        self.printer.println("slt t2, t0, t1")
        self.printer.println("sw t1, 4(sp)")
        self.printer.println("sw t0, 48(sp)")
        self.printer.println("beq x0, t2, _fill_n_L3")
        self.printer.println("li t0, 0")
        self.printer.println("li t1, 0")
        self.printer.println("li t2, 4")
        self.printer.println("lw t3, 48(sp)")
        self.printer.println("mul t4, t2, t3")
        self.printer.println("add t2, t1, t4")
        self.printer.println("mv t1, t2")
        self.printer.println("lw t2, 52(sp)")
        self.printer.println("add t4, t2, t1")
        self.printer.println("sw t0, 0(t4)")
        self.printer.println("sw t2, 0(sp)")
        self.printer.println("sw t3, 48(sp)")

        self.printer.printLabel(Label(BlockLabel, "_fill_n_L2"))

        self.printer.println("li t0, 1")
        self.printer.println("lw t1, 48(sp)")
        self.printer.println("add t2, t1, t0")
        self.printer.println("mv t1, t2")
        self.printer.println("sw t1, 48(sp)")
        self.printer.println("j _fill_n_L1")

        self.printer.printLabel(Label(BlockLabel, "_fill_n_L3"))

        self.printer.println("li a0, 0")
        self.printer.println("j fill_n_exit")

        self.printer.printComment("end of body")
        self.printer.println("")

        self.printer.printLabel(Label(BlockLabel, "fill_n_exit"))

        self.printer.printComment("start of epilogue")

        self.printer.println("lw ra, 44(sp)")
        self.printer.println("addi sp, sp, 52")

        self.printer.printComment("end of epilogue")
        self.printer.println("")

        self.printer.println("ret")
        self.printer.println("")

    # transform tac instrs to RiscV instrs
    # collect some info which is saved in SubroutineInfo for SubroutineEmitter
    def selectInstr(self, func: TACFunc) -> tuple[list[str], SubroutineInfo]:
        selector: RiscvAsmEmitter.RiscvInstrSelector = (
            RiscvAsmEmitter.RiscvInstrSelector(func.entry)
        )
        for instr in func.getInstrSeq():
            instr.accept(selector)

        info = SubroutineInfo(func.entry, selector.more_size, func.params)

        return (selector.seq, info)

    # use info to construct a RiscvSubroutineEmitter
    def emitSubroutine(self, info: SubroutineInfo):
        return RiscvSubroutineEmitter(self, info)

    # return all the string stored in asmcodeprinter
    def emitEnd(self):
        return self.printer.close()

    class RiscvInstrSelector(TACVisitor):
        def __init__(self, entry: Label) -> None:
            self.entry = entry
            self.seq = []
            self.more_size = 0
            self.last_size = 0

        def visitOther(self, instr: TACInstr) -> None:
            raise NotImplementedError(
                "RiscvInstrSelector visit{} not implemented".format(
                    type(instr).__name__
                )
            )

        # in step11, you need to think about how to deal with globalTemp in almost all the visit functions.
        def visitReturn(self, instr: Return) -> None:
            if instr.value is not None:
                self.seq.append(Riscv.Move(Riscv.A0, instr.value))
            else:
                self.seq.append(Riscv.LoadImm(Riscv.A0, 0))
            self.seq.append(Riscv.JumpToEpilogue(self.entry))

        def visitMark(self, instr: Mark) -> None:
            self.seq.append(Riscv.RiscvLabel(instr.label))

        def visitLoadImm4(self, instr: LoadImm4) -> None:
            self.seq.append(Riscv.LoadImm(instr.dst, instr.value))

        def visitUnary(self, instr: Unary) -> None:
            op = {
                TacUnaryOp.NEG: RvUnaryOp.NEG,
                TacUnaryOp.BITNOT: RvUnaryOp.NOT,
                TacUnaryOp.LOGICNOT: RvUnaryOp.SEQZ
                # You can add unary operations here.
            }[instr.op]
            self.seq.append(Riscv.Unary(op, instr.dst, instr.operand))

        def visitBinary(self, instr: Binary) -> None:
            """
            For different tac operation, you should translate it to different RiscV code
            A tac operation may need more than one RiscV instruction
            """
            if instr.op == TacBinaryOp.LOR:
                self.seq.append(
                    Riscv.Binary(RvBinaryOp.OR, instr.dst, instr.lhs, instr.rhs)
                )
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.LAND:
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.lhs))
                self.seq.append(
                    Riscv.Binary(RvBinaryOp.SUB, instr.dst, Riscv.ZERO, instr.dst)
                )
                self.seq.append(
                    Riscv.Binary(RvBinaryOp.AND, instr.dst, instr.dst, instr.rhs)
                )
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.EQ:
                self.seq.append(
                    Riscv.Binary(RvBinaryOp.SUB, instr.dst, instr.lhs, instr.rhs)
                )
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.NE:
                self.seq.append(
                    Riscv.Binary(RvBinaryOp.SUB, instr.dst, instr.lhs, instr.rhs)
                )
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.LE:
                self.seq.append(
                    Riscv.Binary(RvBinaryOp.SGT, instr.dst, instr.lhs, instr.rhs)
                )
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.GE:
                self.seq.append(
                    Riscv.Binary(RvBinaryOp.SLT, instr.dst, instr.lhs, instr.rhs)
                )
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            else:
                op = {
                    TacBinaryOp.ADD: RvBinaryOp.ADD,
                    TacBinaryOp.SUB: RvBinaryOp.SUB,
                    TacBinaryOp.MUL: RvBinaryOp.MUL,
                    TacBinaryOp.DIV: RvBinaryOp.DIV,
                    TacBinaryOp.MOD: RvBinaryOp.REM,
                    TacBinaryOp.LT: RvBinaryOp.SLT,
                    TacBinaryOp.GT: RvBinaryOp.SGT
                    # You can add binary operations here.
                }[instr.op]
                self.seq.append(Riscv.Binary(op, instr.dst, instr.lhs, instr.rhs))

        def visitCondBranch(self, instr: CondBranch) -> None:
            self.seq.append(Riscv.Branch(instr.cond, instr.label))

        def visitBranch(self, instr: Branch) -> None:
            self.seq.append(Riscv.Jump(instr.target))

        # in step9, you need to think about how to pass the parameters and how to store and restore callerSave regs
        # in step11, you need to think about how to store the array

        def visitAssign(self, instr: Assign) -> None:
            self.seq.append(Riscv.Move(instr.dst, instr.src))

        def visitCall(self, instr: Call) -> None:
            self.seq.append(Riscv.Call(instr.label, instr.srcs))
            self.seq.append(Riscv.Move(instr.dst, Riscv.A0))

        def visitLoadSymbol(self, instr: Load_Symbol) -> None:
            self.seq.append(Riscv.La(instr.dst, instr.name))

        def visitLoad(self, instr: Load) -> None:
            self.seq.append(Riscv.Load(instr.dst, instr.src, instr.offset))

        def visitStore(self, instr: Store) -> None:
            self.seq.append(Riscv.Store(instr.dst, instr.src, instr.offset))

        def visitAlloc(self, instr: Alloc) -> None:
            self.more_size += instr.size

        def visitGetAddr(self, instr: GetAddr) -> None:
            self.seq.append(Riscv.GetAddr(instr.dst, self.last_size))
            self.last_size = self.more_size

        def visitStoreToStack(self, instr: STORE_TO_STACK) -> None:
            self.seq.append(Riscv.Store(instr.dst, instr.src, 0))

        def visitLoadFromStack(self, instr: LOAD_FROM_STACK) -> None:
            self.seq.append(Riscv.Load(instr.dst, instr.src, 0))


"""
RiscvAsmEmitter: an SubroutineEmitter for RiscV
"""


class RiscvSubroutineEmitter(SubroutineEmitter):
    def __init__(self, emitter: RiscvAsmEmitter, info: SubroutineInfo) -> None:
        super().__init__(emitter, info)

        # + 4 is for the RA reg
        self.nextLocalOffset = 4 * len(Riscv.CalleeSaved) + 4 + self.info.more_size

        # the buf which stored all the NativeInstrs in this function
        self.buf: list[NativeInstr] = []

        # from temp to int
        # record where a temp is stored in the stack
        self.offsets = {}

        self.params = {}
        for order, temp in enumerate(info.params):
            self.params[temp.index] = 4 * order

        self.loaded_param = 0
        # self.numArgs = info.numArgs

        self.printer.printLabel(info.funcLabel)

        # in step9, step11 you can compute the offset of local array and parameters here

    def emitComment(self, comment: str) -> None:
        # you can add some log here to help you debug
        pass

    # store some temp to stack
    # usually happen when reaching the end of a basicblock
    # in step9, you need to think about the fuction parameters here
    def emitStoreToStack(self, src: Reg) -> None:
        if src.temp.index not in self.offsets:
            self.offsets[src.temp.index] = self.nextLocalOffset
            self.nextLocalOffset += 4
        self.buf.append(
            Riscv.NativeStoreWord(src, Riscv.SP, self.offsets[src.temp.index])
        )

    # load some temp from stack
    # usually happen when using a temp which is stored to stack before
    # in step9, you need to think about the fuction parameters here
    def emitLoadFromStack(self, dst: Reg, src: Temp):
        if src.index not in self.offsets:
            self.offsets[src.index] = self.params[src.index]
            self.buf.append(
                Riscv.NativeLoadWord(dst, Riscv.SP, self.offsets[src.index], True)
            )
        else:
            self.buf.append(
                Riscv.NativeLoadWord(dst, Riscv.SP, self.offsets[src.index], False)
            )

    # add a NativeInstr to buf
    # when calling the fuction emitEnd, all the instr in buf will be transformed to RiscV code
    def emitNative(self, instr: NativeInstr):
        self.buf.append(instr)

    def emitLabel(self, label: Label):
        self.buf.append(Riscv.RiscvLabel(label).toNative([], []))

    def emitEnd(self):
        self.printer.printComment("start of prologue")
        self.printer.printInstr(Riscv.SPAdd(-self.nextLocalOffset))

        # in step9, you need to think about how to store RA here
        # you can get some ideas from how to save CalleeSaved regs
        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeStoreWord(
                        Riscv.CalleeSaved[i], Riscv.SP, 4 * i + self.info.more_size
                    )
                )

        self.printer.printInstr(
            Riscv.NativeStoreWord(
                Riscv.RA, Riscv.SP, 4 * len(Riscv.CalleeSaved) + self.info.more_size
            )
        )

        self.printer.printComment("end of prologue")
        self.printer.println("")

        self.printer.printComment("start of body")

        # in step9, you need to think about how to pass the parameters here
        # you can use the stack or regs

        # using asmcodeprinter to output the RiscV code
        for instr in self.buf:
            if isinstance(instr, Riscv.NativeLoadWord) and instr.tag:
                instr.offset += self.nextLocalOffset
            self.printer.printInstr(instr)

        self.printer.printComment("end of body")
        self.printer.println("")

        self.printer.printLabel(
            Label(LabelKind.TEMP, self.info.funcLabel.name + Riscv.EPILOGUE_SUFFIX)
        )
        self.printer.printComment("start of epilogue")

        self.printer.printInstr(
            Riscv.NativeLoadWord(
                Riscv.RA, Riscv.SP, 4 * len(Riscv.CalleeSaved) + self.info.more_size
            )
        )
        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeLoadWord(
                        Riscv.CalleeSaved[i], Riscv.SP, 4 * i + self.info.more_size
                    )
                )

        self.printer.printInstr(Riscv.SPAdd(self.nextLocalOffset))
        self.printer.printComment("end of epilogue")
        self.printer.println("")

        self.printer.printInstr(Riscv.NativeReturn())
        self.printer.println("")

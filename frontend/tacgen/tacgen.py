from frontend.ast.node import T, Optional
from frontend.ast.tree import T, ArgList, Call, Function, Optional, ParamList
from frontend.ast import node
from frontend.ast.tree import *
from frontend.ast.visitor import T, Visitor
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from utils.label.blocklabel import BlockLabel
from utils.label.funclabel import FuncLabel
from utils.tac import tacop
from utils.tac.temp import Temp
from utils.tac.tacinstr import *
from utils.tac.tacfunc import TACFunc
from utils.tac.tacvar import TACVar
from utils.tac.tacprog import TACProg
from utils.tac.tacvisitor import TACVisitor
from utils.error import *


"""
The TAC generation phase: translate the abstract syntax tree into three-address code.
"""


class LabelManager:
    """
    A global label manager (just a counter).
    We use this to create unique (block) labels accross functions.
    """

    def __init__(self):
        self.nextTempLabelId = 0

    def freshLabel(self) -> BlockLabel:
        self.nextTempLabelId += 1
        return BlockLabel(str(self.nextTempLabelId))


class TACFuncEmitter(TACVisitor):
    """
    Translates a minidecaf (AST) function into low-level TAC function.
    """

    def __init__(
        self,
        entry: FuncLabel,
        numArgs: int,
        labelManager: LabelManager,
    ) -> None:
        self.labelManager = labelManager
        self.func = TACFunc(entry, numArgs)
        self.visitLabel(entry)
        self.nextTempId = 0

        self.continueLabelStack = []
        self.breakLabelStack = []

    # To get a fresh new temporary variable.
    def freshTemp(self) -> Temp:
        temp = Temp(self.nextTempId)
        self.nextTempId += 1
        return temp

    # To get a fresh new label (for jumping and branching, etc).
    def freshLabel(self) -> Label:
        return self.labelManager.freshLabel()

    # To count how many temporary variables have been used.
    def getUsedTemp(self) -> int:
        return self.nextTempId

    # In fact, the following methods can be named 'appendXXX' rather than 'visitXXX'.
    # E.g., by calling 'visitAssignment', you add an assignment instruction at the end of current function.
    def visitAssignment(self, dst: Temp, src: Temp) -> Temp:
        self.func.add(Assign(dst, src))
        return src

    def visitLoad(self, value: Union[int, str]) -> Temp:
        temp = self.freshTemp()
        self.func.add(LoadImm4(temp, value))
        return temp

    def visitUnary(self, op: UnaryOp, operand: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Unary(op, temp, operand))
        return temp

    def visitUnarySelf(self, op: UnaryOp, operand: Temp) -> None:
        self.func.add(Unary(op, operand, operand))

    def visitBinary(self, op: BinaryOp, lhs: Temp, rhs: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Binary(op, temp, lhs, rhs))
        return temp

    def visitBinarySelf(self, op: BinaryOp, lhs: Temp, rhs: Temp) -> None:
        self.func.add(Binary(op, lhs, lhs, rhs))

    def visitBranch(self, target: Label) -> None:
        self.func.add(Branch(target))

    def visitCondBranch(self, op: CondBranchOp, cond: Temp, target: Label) -> None:
        self.func.add(CondBranch(op, cond, target))

    def visitReturn(self, value: Optional[Temp]) -> None:
        self.func.add(Return(value))

    def visitLabel(self, label: Label) -> None:
        self.func.add(Mark(label))

    def visitMemo(self, content: str) -> None:
        self.func.add(Memo(content))

    def visitRaw(self, instr: TACInstr) -> None:
        self.func.add(instr)

    def visitEnd(self) -> TACFunc:
        if (len(self.func.instrSeq) == 0) or (not self.func.instrSeq[-1].isReturn()):
            self.func.add(Return(None))
        self.func.tempUsed = self.getUsedTemp()
        return self.func

    # To open a new loop (for break/continue statements)
    def openLoop(self, breakLabel: Label, continueLabel: Label) -> None:
        self.breakLabelStack.append(breakLabel)
        self.continueLabelStack.append(continueLabel)

    # To close the current loop.
    def closeLoop(self) -> None:
        self.breakLabelStack.pop()
        self.continueLabelStack.pop()

    # To get the label for 'break' in the current loop.
    def getBreakLabel(self) -> Label:
        return self.breakLabelStack[-1]

    # To get the label for 'continue' in the current loop.
    def getContinueLabel(self) -> Label:
        return self.continueLabelStack[-1]

    def visitCall(self, args: [Temp], func_name: Label) -> Temp:
        temp = self.freshTemp()
        self.func.add(Call(temp, args, func_name))
        return temp

    def visitLoadGlobalVar(self, name: str) -> Temp:
        dst_src = self.freshTemp()
        self.func.add(Load_Symbol(dst_src, name))
        dst = self.freshTemp()
        self.func.add(Load(dst, dst_src, 0))
        return dst

    def visitStoreGlobalVar(self, src: Temp, name: str) -> Temp:
        dst = self.freshTemp()
        self.func.add(Load_Symbol(dst, name))
        self.func.add(Store(dst, src, 0))
        return src

    def visitArrayDeclaration(self, dst: Temp, size: int) -> None:
        self.func.add(Alloc(size))
        self.func.add(GetAddr(dst))

    def visitLoadFromStack(self, src: Temp, offset: Temp) -> Temp:
        dst = self.freshTemp()
        temp = self.freshTemp()
        self.func.add(Binary(TacBinaryOp.ADD, temp, src, offset))
        self.func.add(LOAD_FROM_STACK(temp, dst))
        return dst

    def visitStoreToStack(self, dst: Temp, src: Temp, offset: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Binary(TacBinaryOp.ADD, temp, dst, offset))
        self.func.add(STORE_TO_STACK(src, temp))
        return src

    def visitGlobalArray(self, name: str) -> Temp:
        dst = self.freshTemp()
        self.func.add(Load_Symbol(dst, name))
        return dst

    def visitArrayInitZero(self, addr: Temp, size: int) -> Temp:
        dst = self.freshTemp()
        size_temp = self.visitLoad(size)
        self.func.add(Call(dst, [addr, size_temp], Label(FuncLabel, "fill_n")))
        return dst


class TACGen(Visitor[TACFuncEmitter, None]):
    # Entry of this phase
    def transform(self, program: Program) -> TACProg:
        labelManager = LabelManager()
        tacFuncs = []
        tacVars = []
        self.func_labels = {}
        self.label_to_type = {}

        for item in program:
            if isinstance(item, Function):
                # in step9, you need to use real parameter count
                funcName = item.ident.value
                astFunc = item
                self.func_labels[funcName] = FuncLabel(funcName)
                emitter = TACFuncEmitter(
                    self.func_labels[funcName], len(astFunc.params), labelManager
                )
                astFunc.params.accept(self, emitter)
                params = []
                isArray = []
                for param in astFunc.params:
                    params.append(param.getattr("symbol").Temp)
                    isArray.append(param.isArray())
                self.label_to_type[funcName] = isArray
                emitter.func.params = params

                astFunc.body.accept(self, emitter)
                tacFuncs.append(emitter.visitEnd())
            if isinstance(item, Declaration):
                item.getattr("symbol").isGlobal = True
                if bool(item.dim):
                    size = 4
                    item.getattr("symbol").dim = []
                    for dim in item.dim:
                        if dim.value:
                            item.getattr("symbol").dim.append(dim.value)
                            size *= dim.value
                        else:
                            raise DecafBadArraySizeError
                    item.getattr("symbol").size = size
                init_list = []
                if bool(item.init_list):
                    for init in item.init_list:
                        init_list.append(init.value)
                    tacVars.append(
                        TACVar(item.ident.value, init_list, item.getattr("symbol").size)
                    )
                elif bool(item.init_expr):
                    if not isinstance(item.init_expr, IntLiteral):
                        raise DecafGlobalVarBadInitValueError(item.ident.value)
                    tacVars.append(
                        TACVar(
                            item.ident.value,
                            [item.init_expr.value],
                            item.getattr("symbol").size,
                        )
                    )
                else:
                    tacVars.append(
                        TACVar(item.ident.value, None, item.getattr("symbol").size)
                    )

        return TACProg(tacFuncs, tacVars)

    def visitParamList(self, param_list: ParamList, mv: TACFuncEmitter) -> None:
        for param in param_list:
            param.accept(self, mv)
            param.getattr("symbol").Temp = mv.freshTemp()

    def visitArgList(self, arg_list: ArgList, mv: TACFuncEmitter) -> [Temp]:
        args = []
        for arg in arg_list:
            arg.accept(self, mv)
            args.append(arg.getattr("val"))
        arg_list.setattr("val", args)

    def visitCall(self, call: Call, mv: TACFuncEmitter) -> None:
        call.args.accept(self, mv)
        should_be_array = self.label_to_type[call.ident.value]
        for order, arg in enumerate(call.args):
            if isinstance(arg, Identifier) and (
                arg.getattr("symbol").isArray() != should_be_array[order]
            ):
                raise DecafTypeMismatchError
        call.setattr(
            "val",
            mv.visitCall(
                call.args.getattr("val"),
                self.func_labels[call.ident.value],
            ),
        )

    def visitBlock(self, block: Block, mv: TACFuncEmitter, func=False) -> None:
        for child in block:
            child.accept(self, mv)

    def visitReturn(self, stmt: Return, mv: TACFuncEmitter) -> None:
        stmt.expr.accept(self, mv)
        if isinstance(stmt.expr, Subscription) and not stmt.expr.legal():
            raise DecafTypeMismatchError
        mv.visitReturn(stmt.expr.getattr("val"))

    def visitBreak(self, stmt: Break, mv: TACFuncEmitter) -> None:
        mv.visitBranch(mv.getBreakLabel())

    def visitContinue(self, stmt: Continue, mv: TACFuncEmitter) -> None:
        mv.visitBranch(mv.getContinueLabel())

    def visitIdentifier(self, ident: Identifier, mv: TACFuncEmitter) -> None:
        """
        1. Set the 'val' attribute of ident as the temp variable of the 'symbol' attribute of ident.
        """
        if ident.getattr("symbol").isGlobal:
            if not ident.left:
                if ident.getattr("symbol").isArray():
                    ident.setattr("val", mv.visitGlobalArray(ident.value))
                else:
                    ident.setattr("val", mv.visitLoadGlobalVar(ident.value))
        else:
            ident.setattr("val", ident.getattr("symbol").Temp)

    def visitDeclaration(self, decl: Declaration, mv: TACFuncEmitter) -> None:
        """
        1. Get the 'symbol' attribute of decl.
        2. Use mv.freshTemp to get a new temp variable for this symbol.
        3. If the declaration has an initial value, use mv.visitAssignment to set it.
        """
        size = 4
        if bool(decl.dim):
            decl.getattr("symbol").dim = []
            for dim in decl.dim:
                if dim.value:
                    dim.accept(self, mv)
                    decl.getattr("symbol").dim.append(dim.getattr("val"))
                    size *= dim.value
                else:
                    raise DecafBadArraySizeError

        decl.getattr("symbol").Temp = mv.freshTemp()
        if bool(decl.init_expr):
            decl.init_expr.accept(self, mv)
            mv.visitAssignment(
                decl.getattr("symbol").Temp, decl.init_expr.getattr("val")
            )

        if (
            isinstance(decl.init_expr, Identifier)
            and decl.init_expr.getattr("symbol").isArray()
        ):
            raise DecafTypeMismatchError

        if bool(decl.dim):
            mv.visitArrayDeclaration(decl.getattr("symbol").Temp, size)

        if bool(decl.init_list):
            mv.visitArrayInitZero(decl.getattr("symbol").Temp, int(size / 4))
            for order, init in enumerate(decl.init_list):
                init.accept(self, mv)
                offset_temp = mv.visitLoad(4 * order)
                mv.visitStoreToStack(
                    decl.getattr("symbol").Temp, init.getattr("val"), offset_temp
                )

    def visitAssignment(self, expr: Assignment, mv: TACFuncEmitter) -> None:
        """
        1. Visit the right hand side of expr, and get the temp variable of left hand side.
        2. Use mv.visitAssignment to emit an assignment instruction.
        3. Set the 'val' attribute of expr as the value of assignment instruction.
        """
        expr.rhs.accept(self, mv)
        expr.lhs.left = True
        expr.lhs.accept(self, mv)

        if (
            isinstance(expr.lhs, Identifier)
            and isinstance(expr.rhs, Identifier)
            and (
                expr.lhs.getattr("symbol").isArray()
                != expr.rhs.getattr("symbol").isArray()
            )
        ):
            raise DecafTypeMismatchError

        if (
            isinstance(expr.lhs, Identifier) and expr.lhs.getattr("symbol").isArray()
        ) or (
            isinstance(expr.rhs, Identifier) and expr.rhs.getattr("symbol").isArray()
        ):
            raise DecafTypeMismatchError
        if (isinstance(expr.lhs, Subscription) and not expr.lhs.legal()) or (
            isinstance(expr.rhs, Subscription) and not expr.rhs.legal()
        ):
            raise DecafTypeMismatchError

        if isinstance(expr.lhs, Identifier) and expr.lhs.getattr("symbol").isGlobal:
            expr.setattr(
                "val",
                mv.visitStoreGlobalVar(expr.rhs.getattr("val"), expr.lhs.value),
            )
        elif isinstance(expr.lhs, Subscription):
            expr.setattr(
                "val",
                mv.visitStoreToStack(
                    expr.lhs.getattr("address")[0],
                    expr.rhs.getattr("val"),
                    expr.lhs.getattr("address")[1],
                ),
            )
        else:
            expr.setattr(
                "val",
                mv.visitAssignment(expr.lhs.getattr("val"), expr.rhs.getattr("val")),
            )

    def visitIf(self, stmt: If, mv: TACFuncEmitter) -> None:
        stmt.cond.accept(self, mv)

        if stmt.otherwise is NULL:
            skipLabel = mv.freshLabel()
            mv.visitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitLabel(skipLabel)
        else:
            skipLabel = mv.freshLabel()
            exitLabel = mv.freshLabel()
            mv.visitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitBranch(exitLabel)
            mv.visitLabel(skipLabel)
            stmt.otherwise.accept(self, mv)
            mv.visitLabel(exitLabel)

    def visitWhile(self, stmt: While, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitDowhile(self, stmt: Dowhile, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()

        mv.openLoop(breakLabel, loopLabel)
        mv.visitLabel(loopLabel)
        stmt.body.accept(self, mv)

        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        mv.visitBranch(loopLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitFor(self, stmt: For, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        stmt.init.accept(self, mv)
        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        stmt.update.accept(self, mv)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitUnary(self, expr: Unary, mv: TACFuncEmitter) -> None:
        expr.operand.accept(self, mv)
        if (
            isinstance(expr.operand, Identifier)
            and expr.operand.getattr("symbol").isArray()
        ):
            raise DecafTypeMismatchError

        if isinstance(expr.operand, Subscription) and not expr.operand.legal():
            raise DecafTypeMismatchError

        op = {
            node.UnaryOp.Neg: tacop.TacUnaryOp.NEG,
            node.UnaryOp.BitNot: tacop.TacUnaryOp.BITNOT,
            node.UnaryOp.LogicNot: tacop.TacUnaryOp.LOGICNOT
            # You can add unary operations here.
        }[expr.op]
        expr.setattr("val", mv.visitUnary(op, expr.operand.getattr("val")))

    def visitBinary(self, expr: Binary, mv: TACFuncEmitter) -> None:
        expr.lhs.accept(self, mv)
        expr.rhs.accept(self, mv)
        if (
            isinstance(expr.lhs, Identifier) and expr.lhs.getattr("symbol").isArray()
        ) or (
            isinstance(expr.rhs, Identifier) and expr.rhs.getattr("symbol").isArray()
        ):
            raise DecafTypeMismatchError
        if (isinstance(expr.lhs, Subscription) and not expr.lhs.legal()) or (
            isinstance(expr.rhs, Subscription) and not expr.rhs.legal()
        ):
            raise DecafTypeMismatchError

        op = {
            node.BinaryOp.Add: tacop.TacBinaryOp.ADD,
            node.BinaryOp.Sub: tacop.TacBinaryOp.SUB,
            node.BinaryOp.Mul: tacop.TacBinaryOp.MUL,
            node.BinaryOp.Div: tacop.TacBinaryOp.DIV,
            node.BinaryOp.Mod: tacop.TacBinaryOp.MOD,
            node.BinaryOp.LogicOr: tacop.TacBinaryOp.LOR,
            node.BinaryOp.LogicAnd: tacop.TacBinaryOp.LAND,
            node.BinaryOp.EQ: tacop.TacBinaryOp.EQ,
            node.BinaryOp.NE: tacop.TacBinaryOp.NE,
            node.BinaryOp.LT: tacop.TacBinaryOp.LT,
            node.BinaryOp.GT: tacop.TacBinaryOp.GT,
            node.BinaryOp.LE: tacop.TacBinaryOp.LE,
            node.BinaryOp.GE: tacop.TacBinaryOp.GE
            # You can add binary operations here.
        }[expr.op]
        expr.setattr(
            "val", mv.visitBinary(op, expr.lhs.getattr("val"), expr.rhs.getattr("val"))
        )

    def visitCondExpr(self, expr: ConditionExpression, mv: TACFuncEmitter) -> None:
        """
        1. Refer to the implementation of visitIf and visitBinary.
        """
        expr.cond.accept(self, mv)

        skipLabel = mv.freshLabel()
        exitLabel = mv.freshLabel()
        expr.setattr("val", mv.freshTemp())
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, expr.cond.getattr("val"), skipLabel)
        expr.then.accept(self, mv)
        mv.visitAssignment(expr.getattr("val"), expr.then.getattr("val"))
        mv.visitBranch(exitLabel)
        mv.visitLabel(skipLabel)
        expr.otherwise.accept(self, mv)
        mv.visitAssignment(expr.getattr("val"), expr.otherwise.getattr("val"))
        mv.visitLabel(exitLabel)

    def visitIntLiteral(self, expr: IntLiteral, mv: TACFuncEmitter, L=False) -> None:
        expr.setattr("val", mv.visitLoad(expr.value))

    def visitSubscription(self, subscription: Subscription, mv: TACFuncEmitter) -> None:
        subscription.index.accept(self, mv)
        subscription.base.accept(self, mv)
        dim = []
        index = []
        address: Temp
        if isinstance(subscription.base, Identifier):
            index = [subscription.index.getattr("val")]
            if subscription.base.getattr("symbol").isGlobal:
                address = subscription.base.getattr("val")
                for dimmy in subscription.base.getattr("symbol").dim:
                    dim.append(mv.visitLoad(int(dimmy)))
            else:
                dim = subscription.base.getattr("symbol").dim
                address = subscription.base.getattr("symbol").Temp
            subscription.setattr("info", (dim, index, address))
        elif isinstance(subscription.base, Subscription):
            dim = subscription.base.getattr("info")[0]
            index = subscription.base.getattr("info")[1] + [
                subscription.index.getattr("val")
            ]
            address = subscription.base.getattr("info")[2]
            subscription.setattr("info", (dim, index, address))
        if len(dim) == len(index):
            reverse_dim = dim[::-1]
            reverse_index = index[::-1]
            offset = mv.visitLoad(0)
            weight = mv.visitLoad(4)
            for i in range(len(dim)):
                temp = mv.visitBinary(TacBinaryOp.MUL, weight, reverse_index[i])
                mv.visitAssignment(
                    offset, mv.visitBinary(TacBinaryOp.ADD, offset, temp)
                )
                if i != len(dim) - 1:
                    mv.visitAssignment(
                        weight, mv.visitBinary(TacBinaryOp.MUL, weight, reverse_dim[i])
                    )

            if subscription.left:
                subscription.setattr("address", (address, offset))
            else:
                subscription.setattr("val", mv.visitLoadFromStack(address, offset))

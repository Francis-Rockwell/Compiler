from typing import Protocol, TypeVar, cast

from frontend.ast.node import T, Node, NullType, Optional
from frontend.ast.tree import *
from frontend.ast.tree import T, Optional, ParamList
from frontend.ast.visitor import T, RecursiveVisitor, Visitor
from frontend.scope.globalscope import GlobalScope
from frontend.scope.scope import Scope, ScopeKind
from frontend.scope.scopestack import ScopeStack
from frontend.symbol.funcsymbol import FuncSymbol
from frontend.symbol.symbol import Symbol
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from frontend.type.type import DecafType
from utils.error import *
from utils.riscv import MAX_INT

"""
The namer phase: resolve all symbols defined in the abstract 
syntax tree and store them in symbol tables (i.e. scopes).
"""


class Namer(Visitor[ScopeStack, None]):
    def __init__(self) -> None:
        self.scopestack = ScopeStack()
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        # Global scope. You don't have to consider it until Step 6.
        program.globalScope = GlobalScope
        ctx = Scope(program.globalScope)
        ctx.kind = ScopeKind.GLOBAL
        self.scopestack.push(ctx)
        program.accept(self, ctx)
        self.scopestack.pop()
        return program

    def visitProgram(self, program: Program, ctx: Scope) -> None:
        # Check if the 'main' function is missing
        if not program.hasMainFunc():
            raise DecafNoMainFuncError
        for item in program:
            if isinstance(item, Function):
                program.globalScope.define(item.ident)
                for symbol in ctx.symbols.values():
                    if symbol.name == item.ident.value:
                        raise DecafRedefinedFuncError(item.ident.value)

                ctx.declare(
                    FuncSymbol(item.ident.value, item.ret_t, ctx, len(item.params))
                )

                item.setattr("symbol", ctx.get(item.ident.value))
                item_ctx = Scope(ScopeKind.LOCAL)
                self.scopestack.push(item_ctx)
                item.accept(self, item_ctx)
                self.scopestack.pop()

            elif isinstance(item, Declaration):
                item.accept(self, ctx)

    def visitFunction(self, func: Function, ctx: Scope) -> None:
        func.params.accept(self, ctx)
        func.body.accept(self, ctx, True)

    def visitParamList(self, param_list: ParamList, ctx: Scope) -> None:
        for param in param_list.params().values():
            param.accept(self, ctx)

    def visitParam(self, param: Param, ctx: Scope) -> None:
        if bool(ctx.lookup(param.ident.value)):
            raise DecafRedefinedVarError(param.ident.value)
        else:
            if param.isArray():
                var = VarSymbol(param.ident.value, param.var_t.type)
                var.dim = param.dim
                ctx.declare(var)
            else:
                ctx.declare(VarSymbol(param.ident.value, param.var_t.type))

        param.setattr("symbol", ctx.get(param.ident.value))

    def visitCall(self, call: Call, ctx: Scope) -> None:
        for arg in call.args:
            arg.accept(self, ctx)

        for context in self.scopestack:
            opt_symbol = context.lookup(call.ident.value)
            if bool(opt_symbol) and not opt_symbol.isFunc:
                raise DecafVarShadowFuncError(call.ident.value)

        found = False
        for symbol in self.scopestack.global_scope().symbols.values():
            if symbol.name == call.ident.value and symbol.isFunc:
                call.ident.setattr(
                    "symbol", self.scopestack.global_scope().get(call.ident.value)
                )
                if len(call.args) != symbol.params_num:
                    raise DecafWrongArgsNumError(call.ident.value)
                found = True
                break
        if not found:
            raise DecafUndefinedFuncError(call.ident.value)

    def visitBlock(self, block: Block, ctx: Scope, func=False) -> None:
        if func:
            for child in block:
                child.accept(self, ctx)
        else:
            newctx = Scope(ScopeKind.LOCAL)
            self.scopestack.push(newctx)
            for child in block:
                child.accept(self, newctx)
            self.scopestack.pop()

    def visitReturn(self, stmt: Return, ctx: Scope) -> None:
        stmt.expr.accept(self, ctx)

    def visitFor(self, stmt: For, ctx: Scope) -> None:
        """
        1. Open a local scope for stmt.init.
        2. Visit stmt.init, stmt.cond, stmt.update.
        3. Open a loop in ctx (for validity checking of break/continue)
        4. Visit body of the loop.
        5. Close the loop and the local scope.
        """
        for_ctx = Scope(ScopeKind.LOCAL)
        self.scopestack.push(for_ctx)
        for_ctx.loop = True
        stmt.init.accept(self, for_ctx)
        stmt.cond.accept(self, for_ctx)
        stmt.update.accept(self, for_ctx)
        stmt.body.accept(self, for_ctx)
        self.scopestack.pop()

    def visitIf(self, stmt: If, ctx: Scope) -> None:
        stmt.cond.accept(self, ctx)
        stmt.then.accept(self, ctx)

        # check if the else branch exists
        if not stmt.otherwise is NULL:
            stmt.otherwise.accept(self, ctx)

    def visitWhile(self, stmt: While, ctx: Scope) -> None:
        ctx.loop = True
        stmt.cond.accept(self, ctx)
        stmt.body.accept(self, ctx)
        ctx.loop = False

    def visitDowhile(self, stmt: Dowhile, ctx: Scope) -> None:
        ctx.loop = True
        stmt.cond.accept(self, ctx)
        stmt.body.accept(self, ctx)
        ctx.loop = False

    def visitBreak(self, stmt: Break, ctx: Scope) -> None:
        """
        You need to check if it is currently within the loop.
        To do this, you may need to check 'visitWhile'.

        if not in a loop:
            raise DecafBreakOutsideLoopError()
        """
        loop = False
        for context in self.scopestack:
            if context.loop:
                loop = True
                break
        if not loop:
            raise DecafBreakOutsideLoopError

    def visitContinue(self, stmt: Continue, ctx: Scope) -> None:
        """
        1. Refer to the implementation of visitBreak.
        """
        loop = False
        for context in self.scopestack:
            if context.loop:
                loop = True
                break
        if not loop:
            raise DecafBreakOutsideLoopError

    def visitDeclaration(self, decl: Declaration, ctx: Scope) -> None:
        """
        1. Use ctx.lookup to find if a variable with the same name has been declared.
        2. If not, build a new VarSymbol, and put it into the current scope using ctx.declare.
        3. Set the 'symbol' attribute of decl.
        4. If there is an initial value, visit it.
        """
        if bool(ctx.lookup(decl.ident.value)):
            raise DecafRedefinedVarError(decl.ident.value)
        else:
            ctx.declare(VarSymbol(decl.ident.value, decl.var_t.type))

        decl.setattr("symbol", ctx.get(decl.ident.value))

        if bool(decl.init_expr):
            decl.init_expr.accept(self, ctx)

        if bool(decl.init_list):
            for init in decl.init_list:
                init.accept(self, ctx)

    def visitAssignment(self, expr: Assignment, ctx: Scope) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.lhs.accept(self, ctx)
        if not isinstance(expr.lhs, Identifier) and not isinstance(
            expr.lhs, Subscription
        ):
            raise DecafBadAssignTypeError
        expr.rhs.accept(self, ctx)

    def visitUnary(self, expr: Unary, ctx: Scope) -> None:
        expr.operand.accept(self, ctx)

    def visitBinary(self, expr: Binary, ctx: Scope) -> None:
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)

    def visitCondExpr(self, expr: ConditionExpression, ctx: Scope) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.cond.accept(self, ctx)
        expr.then.accept(self, ctx)
        expr.otherwise.accept(self, ctx)

    def visitIdentifier(self, ident: Identifier, ctx: Scope) -> None:
        """
        1. Use ctx.lookup to find the symbol corresponding to ident.
        2. If it has not been declared, raise a DecafUndefinedVarError.
        3. Set the 'symbol' attribute of ident.
        """
        found = False
        for context in self.scopestack:
            if bool(context.lookup(ident.value)):
                ident.setattr("symbol", context.get(ident.value))
                found = True
                break
        if not found:
            raise DecafUndefinedVarError(ident.value)

    def visitIntLiteral(self, expr: IntLiteral, ctx: Scope) -> None:
        value = expr.value
        if value > MAX_INT:
            raise DecafBadIntValueError(value)

    def visitSubscription(self, subscription: Subscription, ctx: Scope) -> None:
        subscription.base.accept(self, ctx)
        subscription.index.accept(self, ctx)

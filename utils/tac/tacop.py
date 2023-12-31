from enum import Enum, auto, unique


# Kinds of instructions.
@unique
class InstrKind(Enum):
    # Labels.
    LABEL = auto()
    # Sequential instructions (unary operations, binary operations, etc).
    SEQ = auto()
    # Branching instructions.
    JMP = auto()
    # Branching with conditions.
    COND_JMP = auto()
    # Return instruction.
    RET = auto()
    # Call function
    Call = auto()


# Kinds of unary operations.
@unique
class TacUnaryOp(Enum):
    NEG = auto()
    BITNOT = auto()
    LOGICNOT = auto()

    def __str__(self) -> str:
        return ["-", "~", "!"][self.value - 1]


# Kinds of binary operations.
@unique
class TacBinaryOp(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    LOR = auto()
    LAND = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()


# Kinds of branching with conditions.
@unique
class CondBranchOp(Enum):
    BEQ = auto()
    BNE = auto()

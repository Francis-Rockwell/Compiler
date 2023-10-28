from typing import Optional

from frontend.symbol.symbol import Symbol

from .scope import Scope


class ScopeStack:
    def __init__(self) -> None:
        self.stack = []

    def is_empty(self) -> bool:
        if len(self.stack) == 0:
            return True
        else:
            return False

    def size(self) -> int:
        return len(self.stack)

    def push(self, scope: Scope) -> None:
        self.stack.append(scope)

    def pop(self) -> Optional[Scope]:
        if self.is_empty():
            return None
        else:
            return self.stack.pop()

    # 用于返回全局作用域
    def global_scope(self) -> Optional[Scope]:
        if self.is_empty():
            return None
        else:
            return self.stack[0]

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count + self.size() > 0:
            self.count = self.count - 1
            return self.stack[self.count]
        else:
            raise StopIteration

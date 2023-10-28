from typing import Optional


class TACVar:
    def __init__(self, name: str, init: Optional[list[int]], size=4) -> None:
        self.name = name
        self.init = init
        self.size = size

    def printTo(self) -> None:
        print("Global " + self.name + " = " + str(self.init))

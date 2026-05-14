from __future__ import annotations


class KillSwitch:
    def __init__(self) -> None:
        self.engaged: bool = False
        self.reason: str = ""

    def engage(self, reason: str) -> None:
        if not self.engaged:
            self.engaged = True
            self.reason = reason

    def reset(self) -> None:
        self.engaged = False
        self.reason = ""

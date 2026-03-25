class KillSwitch:
    def __init__(self) -> None:
        self._engaged = False
        self._reason = ""

    def engage(self, reason: str) -> None:
        self._engaged = True
        self._reason = reason

    def reset(self) -> None:
        self._engaged = False
        self._reason = ""

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def reason(self) -> str:
        return self._reason


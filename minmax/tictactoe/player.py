class Player:
    def __init__(self, ai: bool, xPlayer : bool, name: str):
        self._ai = ai
        self._sign = 'X' if xPlayer else 'O'
        self._name = name
    
    def getAi(self):
        return self._ai
    
    def getSign(self):
        return self._sign

    def getName(self):
        return self._name
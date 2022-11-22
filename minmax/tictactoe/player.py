class Player:
    def __init__(self, ai: bool, maxPlayer : bool, name: str, searchDepth : int):
        self._ai = ai
        self._maxPlayer = maxPlayer
        self._sign = 1 if maxPlayer else -1
        self._name = name
        self._searchDepth = searchDepth
    
    def getAi(self):
        return self._ai
    
    def getSign(self):
        return self._sign

    def getName(self):
        return self._name
    
    def isMaxPlayer(self):
        return self._maxPlayer
    
    def getSearchDepth(self):
        return self._searchDepth
from .board import Board

class Minimax:
    def __init__(self, searchDepth) -> None:
        self._searchDepth = searchDepth

    def minmax(self, searchDepth : int, board : Board, isMaxPlayer : bool):
        boardValue = board.boardState()
        if board.isTerminalState() or searchDepth == 0:
            return boardValue
        successors = board.generateSuccessors(isMaxPlayer)
        
        values = []
        for successor in successors:
            values.append(self.minmax(searchDepth - 1, successor, not isMaxPlayer))
        
        if isMaxPlayer:
            finalValue = sorted(values, reverse=True)[0]
        else:
            finalValue = sorted(values, reverse=False)[0]

        if searchDepth != self._searchDepth:
            return finalValue
        else: 
            valueIndex = values.index(finalValue)
            return board.getCoords(valueIndex)

from itertools import chain
import copy

class Board:
    def __init__(self):
        self._board = [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],]
    
    def __str__(self):
        boardStr = ""
        intToSign = {-1:'O', 0:' ', 1: 'X'}
        for row in self._board:
            boardStr += "|"
            for element in row:
                boardStr += f" {intToSign[element]} |"
            boardStr += "\n"
        return boardStr
    
    def changeSquare(self, y, x, sign):
        if y not in {0, 1, 2} or x not in {0, 1, 2} or self._board[y][x] != 0:
            raise IndexError()
        else:
            self._board[y][x] = sign
    
    def isTerminalState(self):
        if self.checkWinCondition(True) or self.checkWinCondition(False):
            return True
        flattenedBoard = list(chain.from_iterable(self._board))
        # check for draw
        for square in flattenedBoard:
            if square == 0:
                return False
        return True

    def checkWinCondition(self, maxPlayer):
        table = [1,1,1] if maxPlayer else [-1,-1,-1]

        for i in range(3):
            if [column[i] for column in self._board] == table:
                return True
            if self._board[i] == table:
                return True

        if [self._board[i][i] for i in range(3)] == table:
            return True
        
        if [self._board[i][2 - i] for i in range(3)] == table:
            return True
        
        return False
    
    def boardState(self):
        stateValue = 0
        squareValues = [3,2,3,2,4,2,3,2,3]
        flattenedBoard = list(chain.from_iterable(self._board))
        maxPlayerWins = self.checkWinCondition(True)
        minPlayerWins = self.checkWinCondition(False)

        if maxPlayerWins:
            stateValue = 5
            for square in flattenedBoard:
                stateValue += 1 if square == 0 else 0
        elif minPlayerWins:
            stateValue = -5
            for square in flattenedBoard:
                stateValue -= 1 if square == 0 else 0
        else: # nobody has won yet
            for i in range(len(flattenedBoard)):
                stateValue += squareValues[i] * flattenedBoard[i]
        return stateValue
    
    def generateSuccessors(self, maxPlayer : bool):
        sign = 1 if maxPlayer else -1
        successors = []
        for i in range(3):
            for j in range(3):
                if self._board[i][j] == 0:
                    newBoard = copy.deepcopy(self)
                    newBoard._board[i][j] = sign
                    successors.append(newBoard)
        return successors

    def nthEmptySpace(self, n):
        for i in range(3):
            for j in range(3):
                if self._board[i][j] == 0 and n == 0:
                    return (i, j) # y, x
                elif self._board[i][j] == 0 and n != 0:
                    n -= 1

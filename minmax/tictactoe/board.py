class Board:
    def __init__(self):
        self._board = [[' ', ' ', ' '],
                       [' ', ' ', ' '],
                       [' ', ' ', ' ']]
    
    def __str__(self):
        boardStr = ""
        for row in self._board:
            boardStr += "|"
            for element in row:
                boardStr += f" {element} |"
            boardStr += "\n"
        return boardStr
    
    def changeSquare(self, y, x, sign):
        if y not in {0, 1, 2} or x not in {0, 1, 2} or self._board[y][x] != ' ':
            raise IndexError()
        else:
            self._board[y][x] = sign
    
    def checkWinCondition(self, sign):
        table = ['X', 'X', 'X'] if sign == 'X' else ['O', 'O', 'O']

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

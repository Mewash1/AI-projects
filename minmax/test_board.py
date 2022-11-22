from tictactoe.board import Board

class TestBoard(Board):
    def __init__(self):
        super().__init__()
    
    def overwriteBoard(self, newBoard):
        self._board = newBoard

def test_board_simple():
    board = TestBoard()
    board.changeSquare(0, 0, 'X')
    assert board.boardState(True) == 3

def test_board_max_wins():
    board = TestBoard()
    newBoard = [['X', 'X', 'X'],
                [' ', ' ', ' '],
                [' ', ' ', ' ']]
    board.overwriteBoard(newBoard)
    assert board.boardState(True) == 100
    assert board.boardState(False) == -100

def test_board_max_and_min():
    board = TestBoard()
    newBoard = [[' ', 'X', 'X'],
                ['O', ' ', ' '],
                [' ', ' ', ' ']]
    board.overwriteBoard(newBoard)
    assert board.boardState(True) == 3
    assert board.boardState(False) == -3

def test_board_full():
    board = TestBoard()
    newBoard = [['O', 'X', 'O'],
                ['X', 'O', 'X'],
                ['O', 'X', 'X']]
    board.overwriteBoard(newBoard)
    assert board.boardState(True) == -100
    assert board.boardState(False) == 100

def test_generate_successors():
    board = TestBoard()
    successors = board.generateSuccessors(True)
    newSuccessors = successors[0][0].generateSuccessors(False)
    pass
from .board import Board
from .player import Player
from .minimax import Minimax
import click

class Game:
    def __init__(self, player1 : Player, player2 : Player):
        self._board = Board()
        self._maxPlayer = player1 if player1.isMaxPlayer() else player2
        self._minPlayer = player2 if player1.isMaxPlayer() else player1
        if self._maxPlayer == self._minPlayer:
            raise ValueError("Both players cannot be maxPlayer")

        self._currentPlayer = self._maxPlayer
        self._maxPlayer_minimax = Minimax(self._maxPlayer.getSearchDepth())
        self._minPlayer_minimax = Minimax(self._minPlayer.getSearchDepth())
        self._currentPlayer_minimax = self._maxPlayer_minimax
        self._moves = []    

    def runGame(self, returnMoves : bool = False):
        for _ in range(9):
            self._gameTurn()
            
            if self._board.checkWinCondition(self._currentPlayer.isMaxPlayer()):

                print(self._board)
                print(f"{self._currentPlayer.getName()} has won the game!")

                if returnMoves:
                    return self._moves
                else:
                    return 'X' if self._currentPlayer.isMaxPlayer() else 'O'

            self._switchPlayer()
        print(self._board)
        print("The game has ended in a draw!")

        return 'draw' if not returnMoves else self._moves

    def _switchPlayer(self):
        if self._currentPlayer == self._maxPlayer:
            self._currentPlayer = self._minPlayer
            self._currentPlayer_minimax = self._minPlayer_minimax
        else:
            self._currentPlayer = self._maxPlayer
            self._currentPlayer_minimax = self._maxPlayer_minimax

    def _gameTurn(self):
        if self._currentPlayer.getAi():
           self._aiTurn() 
        else:
            self._humanTurn()
    
    def _aiTurn(self):
        print(self._board)
        y, x = self._currentPlayer_minimax.minimax(self._currentPlayer.getSearchDepth(), self._board, self._currentPlayer.isMaxPlayer())
        self._board.changeSquare(y, x, self._currentPlayer.getSign())
        self._moves.append((x,y))

    def _humanTurn(self):
        badInput = True
        while badInput:
            print(self._board)
            badInput = False
            try:
                x, y = input("Choose x and y coordinates for your move: ").split()
                try:
                    self._board.changeSquare(int(y), int(x), self._currentPlayer.getSign())
                except IndexError:
                    print("Wrong coords: try again")
                    badInput = True
            except ValueError:
                print("wrong input: try again")
                badInput = True

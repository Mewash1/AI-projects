from .board import Board
import random
import click

class Game:
    def __init__(self, player1, player2):
        self._board = Board()
        self._player1 = player1
        self._player2 = player2
        self._currentPlayer = self._player1
        self._turns = 0
    
    def showBoard(self):
        print(self._board)
    
    def switchPlayer(self):
        if self._currentPlayer == self._player1:
            self._currentPlayer = self._player2
        else:
            self._currentPlayer = self._player1

    def runGame(self):
        for _ in range(9):
            self.gameTurn()
            
            if self._board.checkWinCondition(self._currentPlayer.getSign()):
                click.clear()
                print(self._board)
                print(f"{self._currentPlayer.getName()} has won the game!")
                return 0

            self.switchPlayer()
        click.clear()
        print(self._board)
        print("The game has ended in a draw!")
        return 1

    def gameTurn(self):
        if self._currentPlayer.getAi():
           self.aiTurn() 
        else:
            self.humanTurn()
            
    def aiTurn(self):
        badInput = True
        while badInput:
            badInput = False
            x, y = random.choice([0,1,2]), random.choice([0,1,2])
            try:
                self._board.changeSquare(y, x, self._currentPlayer.getSign())
            except IndexError:
                badInput = True

    def humanTurn(self):
        badInput = True
        while badInput:
            click.clear()
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

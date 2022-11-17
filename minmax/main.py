from tictactoe.game import Game, Player

if __name__ == "__main__":
    player1, player2 = Player(False, True, "Human"), Player(True, False, "Ai")
    game = Game(player1, player2)
    game.runGame()
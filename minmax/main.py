from tictactoe.game import Game
from tictactoe.player import Player

if __name__ == "__main__":
    depthsScores = []
    for searchDepth in [9]:
        player1 = Player(maxPlayer=True, ai=False, name="Max", searchDepth=searchDepth), 
        player2 = Player(maxPlayer=False, ai=False, name="Min", searchDepth=searchDepth)

        game = Game(player1, player2, searchDepth)
        depthsScores.append((searchDepth, game.runGame(False, True)))
    print(depthsScores)


# [(1, 1), (2, 2), (3, 1), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)]
# [(1, 1), (2, 2), (3, 1), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)]
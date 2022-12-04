from tictactoe.game import Game
from tictactoe.player import Player
import pandas as pd
import dataframe_image as dfi

def runTests():
    n = 9
    depthsScores = [[] for _ in range(n)]
    for searchDepthMax in range (1, n+1):
        for searchDepthMin in range(1, n+1):
            player1 = Player(maxPlayer=True, ai=True, name="Max", searchDepth=searchDepthMax)
            player2 = Player(maxPlayer=False, ai=True, name="Min", searchDepth=searchDepthMin)
            game = Game(player1, player2)
            depthsScores[searchDepthMax - 1].append(game.runGame(False, False))
    df = pd.DataFrame(depthsScores, range(1, n+1), range(1, n+1))
    dfi.export(df,"results.png")

def play():
    player1 = Player(maxPlayer=True, ai=True, name="Max", searchDepth=9)
    player2 = Player(maxPlayer=False, ai=False, name="Min", searchDepth=9)
    game = Game(player1, player2)
    game.runGame()

if __name__ == "__main__":
    play()

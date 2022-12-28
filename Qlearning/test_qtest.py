import random

def test_first():
    def pickMove(moves):
        maxMove = max(moves)
        goodMoves = []
        for i, move in enumerate(moves):
            if move == maxMove:
                goodMoves.append(i)
        return random.choice(goodMoves)
    
    moves = [1,5,4]
    for i in range(10):
        print(pickMove(moves))

if __name__ == "__main__":
    test_first()
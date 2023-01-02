from Qlearn import Q_learning
import gym
import matplotlib.pyplot as plt

def test_boltzmann():
    envi = gym.make('Taxi-v3')
    
    episodes = 100
    moves = 50
    test_moves = 50
    learning_rates = [0.1,0.3,0.5,0.7,1,5]
    Q_learn = Q_learning(initial_value=0, learning_rate=0.5, env=envi, discount_factor=0.1, from_jason=False, tau=2)
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax.set_xlabel("Episodes trained")
    ax.set_ylabel("Positive results")
    for learning_rate in learning_rates:
        results, episodesList = [], []
        Q_learn.learning_rate = learning_rate
        for i in range(50):
            Q_learn.trainBoltzmann(episodes=episodes, turns=moves)
            miniresults = Q_learn.test(episodes=episodes, turns=test_moves, useBoltzmann=True)
            results.append(miniresults["boltzmann"]["positive"])
            episodesList.append((i+1)*episodes)
        Q_learn.resetTable()
        ax.plot(episodesList, results, label=f"learning rate = {learning_rate}")
        ax.legend()
    fig.savefig("graphs/boltzmann", dpi=400)
    envi.close()

def test_greedy():
    envi = gym.make('Taxi-v3')
    
    episodes = 100
    moves = 50
    test_moves = 50
    learning_rates = [0.1,0.3,0.5,0.7,1,5]
    Q_learn = Q_learning(initial_value=0, learning_rate=0.5, env=envi, discount_factor=0.1, from_jason=False, tau=2)
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax.set_xlabel("Episodes trained")
    ax.set_ylabel("Positive results")
    for learning_rate in learning_rates:
        results, episodesList = [], []
        Q_learn.learning_rate = learning_rate
        for i in range(50):
            Q_learn.trainGreedy(episodes=episodes, turns=moves)
            miniresults = Q_learn.test(episodes=episodes, turns=test_moves, useGreedy=True)
            results.append(miniresults["greedy"]["positive"])
            episodesList.append((i+1)*episodes)
        Q_learn.resetTable()
        ax.plot(episodesList, results, label=f"learning rate = {learning_rate}")
        ax.legend()
    fig.savefig("graphs/greedy1", dpi=400)
    envi.close()

if __name__ == "__main__":
    test_boltzmann()
    #test_greedy()
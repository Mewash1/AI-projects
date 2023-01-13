from Qlearn import Q_learning
import gym
import matplotlib.pyplot as plt

def test_greedy():
    envi = gym.make('Taxi-v3')
    
    episodes = 100
    moves = 50
    test_moves = 50
    learning_rates = [0.1,0.3,0.5,0.7,1,5]
    Q_learn = Q_learning(initial_value=0, learning_rate=0.5, env=envi, discount_factor=0.9, from_jason=False, tau=0.0000000000001)
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax.set_xlabel("Episodes trained")
    ax.set_ylabel("Positive results")
    for learning_rate in learning_rates:
        results, episodesList = [], []
        Q_learn.learning_rate = learning_rate
        for i in range(50):
            Q_learn.train(episodes=episodes, turns=moves, strategy="greedy")
            miniresults = Q_learn.test(episodes=episodes, turns=test_moves, useGreedy=True)
            results.append(miniresults["greedy"]["positive"])
            episodesList.append((i+1)*episodes)
        Q_learn.resetTable()
        ax.plot(episodesList, results, label=f"learning rate = {learning_rate}")
        ax.legend()
    ax.set_title("Greedy strategy")
    fig.savefig("graphs/greedy1", dpi=400)
    envi.close()

def test_epsilon_greedy():
    envi = gym.make('Taxi-v3')
    
    episodes = 100
    moves = 50
    test_moves = 50
    learning_rates = [0.1,0.3,0.5,0.7,1,5]
    epsilons = [0.3, 0.5, 0.7]
    Q_learn = Q_learning(initial_value=0, learning_rate=0.5, env=envi, discount_factor=0.9, from_jason=False, epsilon=1)
    for epsilon in epsilons:
        Q_learn.epsilon = epsilon
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(20)
        ax.set_xlabel("Episodes trained")
        ax.set_ylabel("Positive results")
        for learning_rate in learning_rates:
            results, episodesList = [], []
            Q_learn.learning_rate = learning_rate
            for i in range(50):
                Q_learn.train(episodes=episodes, turns=moves, strategy="epsilonGreedy")
                miniresults = Q_learn.test(episodes=episodes, turns=test_moves, useEpsilonGreedy=True)
                results.append(miniresults["epsilonGreedy"]["positive"])
                episodesList.append((i+1)*episodes)
            Q_learn.resetTable()
            ax.plot(episodesList, results, label=f"learning rate = {learning_rate}")
            ax.legend()
        ax.set_title(f"Epsilon-greedy strategy, epsilon={epsilon}")
        fig.savefig(f"graphs/epsilon_greedy,epsilon={epsilon}".replace('.',','), dpi=400)
    envi.close()

def test_boltzmann():
    envi = gym.make('Taxi-v3')
    
    episodes = 100
    moves = 50
    test_moves = 50
    learning_rates = [0.1,0.3,0.5,0.7,1,5]
    taus = [0.1, 1, 10]
    Q_learn = Q_learning(initial_value=0, learning_rate=0.5, env=envi, discount_factor=0.9, from_jason=False, tau=2)

    for tau in taus:
        Q_learn.tau = tau;
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(20)
        ax.set_xlabel("Episodes trained")
        ax.set_ylabel("Positive results")
        for learning_rate in learning_rates:
            results, episodesList = [], []
            Q_learn.learning_rate = learning_rate
            for i in range(50):
                Q_learn.train(episodes=episodes, turns=moves, strategy="boltzmann")
                miniresults = Q_learn.test(episodes=episodes, turns=test_moves, useBoltzmann=True)
                results.append(miniresults["boltzmann"]["positive"])
                episodesList.append((i+1)*episodes)
            Q_learn.resetTable()
            ax.plot(episodesList, results, label=f"learning rate = {learning_rate}")
            ax.legend()
        ax.set_title(f"Boltzmann strategy, tau={tau}")
        fig.savefig(f"graphs/boltzmann,tau={tau}".replace('.', ','), dpi=400)
    envi.close()

def test_count():
    envi = gym.make('Taxi-v3')
    
    episodes = 100
    moves = 50
    test_moves = 50
    learning_rates = [0.1,0.3,0.5,0.7,1,5]
    Q_learn = Q_learning(initial_value=0, learning_rate=0.5, env=envi, discount_factor=0.9, from_jason=False)
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax.set_xlabel("Episodes trained")
    ax.set_ylabel("Positive results")
    for learning_rate in learning_rates:
        results, episodesList = [], []
        Q_learn.learning_rate = learning_rate
        for i in range(50):
            Q_learn.train(episodes=episodes, turns=moves, strategy="count")
            miniresults = Q_learn.test(episodes=episodes, turns=test_moves, useCount=True)
            results.append(miniresults["count"]["positive"])
            episodesList.append((i+1)*episodes)
        Q_learn.resetTable()
        ax.plot(episodesList, results, label=f"learning rate = {learning_rate}")
        ax.legend()
    ax.set_title("Counting strategy")
    fig.savefig("graphs/count", dpi=400)
    envi.close()

if __name__ == "__main__":
    test_boltzmann()
    #test_greedy()
    test_epsilon_greedy()
    test_count()
    '''envi = gym.make('Taxi-v3')
    Q_learn = Q_learning(initial_value=0, learning_rate=0.5, env=envi, discount_factor=0.9, tau=0.1)
    Q_learn.train(5000, 100, "boltzmann")
    Q_learn.saveTable()'''
    
    '''envi2 = gym.make('Taxi-v3', render_mode="human")
    Q_learn = Q_learning(initial_value=0, learning_rate=0.5, env=envi2, discount_factor=0.9, from_jason=True, tau=0.1)
    Q_learn.env = envi2
    Q_learn.episodeTest(50, Q_learn.boltzmannChoice)'''
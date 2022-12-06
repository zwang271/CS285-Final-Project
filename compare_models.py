import numpy as np
import argparse
import multiprocessing
from time import time

num_iter = 5_000_000
def simulate_game(queue):
    result = np.zeros(num_iter)
    for i in range(num_iter):
        result[i] = np.random.rand()
    queue.put(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_games', default='20', type=int)

    # parser.add_argument('--protagonist', default='rand',
    #                     choices=['rand', 'greedy', 'maximin', 'human', 'mcts'])
    # parser.add_argument('--opponent', default='rand',
    #                     choices=['rand', 'greedy', 'maximin', 'human', 'mcts'])

    parser = parser.parse_args()
    args = vars(parser)


    start = time()
    q = multiprocessing.Queue()
    processes = []
    results = []

    for i in range(args['num_games']):
        p = multiprocessing.Process(target=simulate_game, args=(q,))
        processes.append(p)
        p.start()

    for p in processes:
        ret = q.get()
        results.append(ret)
    
    for p in processes:
        p.join
    results = np.array(results)
    print(results.shape)
    end = time()
    print(end - start)


    start = time()
    results = []
    for i in range(args['num_games']):
        result = np.zeros(num_iter)
        for i in range(num_iter):
            result[i] = np.random.rand()
        results.append(result)
    results = np.array(results)
    print(results.shape)
    end = time()
    print(end - start)
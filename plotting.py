import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == '__main__':
    TRAIN = './trained_models/'

    # i = int(sys.argv[1])
    # loss_i = np.load(TRAIN + 'loss' + str(i) + '.npy')
    # x = range(1, loss_i.shape[0] + 1)
    # plt.plot(x, loss_i)

    # plt.xlabel('training iteration')
    # plt.ylabel('training loss')
    # # show the plot
    # plt.show()
    # quit()



    # PATH = TRAIN + 'board4_mcts-iter500_num-games10/'
    # title = '500 MCTS iterations, 10 games per training loop'

    # PATH = TRAIN + 'board4_mcts-iter1000_num-games64/'
    # title = '1000 MCTS iterations, 64 games per training loop'

    PATH = TRAIN + 'board4_mcts-iter200_targetQ/'
    title = 'Target Q: 200 MCTS iterations, 64 games per training loop'

    rand_results = np.load(PATH + 'rand_results.npy')
    maximin_results = np.load(PATH + 'maximin_results.npy')

    # calculate win rates by averaging the results over the training iterations
    rand_win_rate = np.mean(rand_results, axis=1)
    maximin_win_rate = np.mean(maximin_results, axis=1)

    # create a range for the x-axis representing the training iterations
    x = range(1, rand_results.shape[0] + 1)

    # plot the win rates against the training iterations
    plt.plot(x, rand_win_rate, label='random policy')
    plt.plot(x, maximin_win_rate, label='maximin policy')

    # add labels and a legend
    plt.xlabel('training iteration')
    plt.ylabel('win rate')
    plt.title(title)
    plt.legend()

    # show the plot
    plt.show()
    
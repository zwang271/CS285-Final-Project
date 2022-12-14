import argparse
from mcts_utils import simulate_mcts
from time import time, sleep
from policies.model import Othello_QNet
import torch
from tqdm import tqdm

import gym
from game_utils import * 
import numpy as np
from policies import mcts
from policies.mcts import copy_env, ReplayBuffer, BLACK_DISK, NO_DISK, WHITE_DISK


# ARGS
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--board_size', type=int, default=4, \
        help='Size of the board')
    parser.add_argument('--seed', type=int, default=0, \
        help='Seed for random number generation')
    parser.add_argument('--mcts_iterations', type=int, default=1000, \
        help='Number of MCTS iterations')
    parser.add_argument('--mcts_c', type=int, default=1, \
        help='Value of exploration parameter c in mcts')    
    parser.add_argument('--initial_rand_steps', type=int, default=0, \
        help='Number of initial random steps')
    parser.add_argument('--num_disk_as_reward', type=bool, \
        default=False, help='Whether to use the number of disks as the reward')
    parser.add_argument('--maximin_depth', type=int, default=3, \
        help='Maximin search depth')
    parser.add_argument('--swap_colors', type=bool, default=False, \
        help='Whether to swap the colors for the maximin search')
    parser.add_argument('--prompt_to_step', type=bool, default=False, \
        help='Whether to prompt the user to step through each move')
    parser.add_argument('--print_tree', type=bool, default=False, \
        help='Whether to print the tree of MCTS iterations')
    parser.add_argument('--render_board', type=bool, default=False, \
        help='Whether to render the board')
    parser.add_argument('--num_games', type=int, default=10, \
        help='How many games to play each time mcts is simulated')
    parser.add_argument('--mute', type=bool, default=True, \
        help='Whether mute Othello game environment from printing game outcomes')
    parser.add_argument('--debug', type=bool, default=False, \
        help='Whether to print debug messages')

    args = vars(parser.parse_args())

    # Test
    board_size = args['board_size']
    seed = args['seed']
    mcts_iterations = args['mcts_iterations']
    mcts_c = args['mcts_c']
    initial_rand_steps = args['initial_rand_steps']
    num_disk_as_reward = args['num_disk_as_reward']
    maximin_depth = args['maximin_depth']
    swap_colors = args['swap_colors']
    prompt_to_step = args['prompt_to_step']
    print_tree = args['print_tree']
    render_board = args['render_board']
    debug = args['debug']
    num_games = args['num_games']


    # env = othello.OthelloBaseEnv(board_size=board_size, num_disk_as_reward=num_disk_as_reward)
    # env.reset()
    # rotate90_map = env.rotate90_map()
    # flip_map = env.flip_map()
    # env.step(env.possible_moves[0])
    # move = env.possible_moves[1]

    # for i in range(4):
    #     env.rotate90()
    #     move = rotate90_map[move]

    #     env.flip()
    #     move = flip_map[move]
  
    #     env.flip()
    #     move = flip_map[move]

    PATH = './trained_models/'

    start = time()
    Q_net = Othello_QNet(args['board_size'])

    # data_buffer = simulate_mcts(args, Q_net)
    # quit()

    # Save initial Q_net
    torch.save(Q_net.state_dict(), PATH + 
        'board' + str(board_size) + '_' +
        'mcts-iter'+ str(mcts_iterations) + '_' +
        'iter0'
        )

    # training the Q_network 
    sgd_iter = 1000
    batch_size = 1000
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(Q_net.parameters(), lr=0.01, weight_decay=0.01)
    for i in tqdm(range(30)):
        data_buffer = simulate_mcts(args, Q_net)

        for j in range(sgd_iter):
            # Load a batch of data
            ob, ac, va = data_buffer.sample(batch_size)

            ac = torch.from_numpy(ac).float()
            va = torch.from_numpy(va).float()
            ob = torch.from_numpy(ob).float()
            pred_va, pred_ac = Q_net(ob)
            pred_va = torch.flatten(pred_va)

            loss_va = mse_loss(va, pred_va)
            loss_ac = mse_loss(ac, pred_ac)
            loss = loss_va + loss_ac

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.save(Q_net.state_dict(), PATH + 
            'board' + str(board_size) + 
            'mcts-iter'+ str(mcts_iterations) + 
            'iter' + str(i+1)
            )

import argparse
from mcts_utils import simulate_mcts
from time import time, sleep
from policies.model import Othello_QNet
import torch
from tqdm import tqdm
from compare_models import *

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
    parser.add_argument('--maximin_depth', type=int, default=5, \
        help='Maximin search depth')
    parser.add_argument('--swap_colors', type=bool, default=False, \
        help='Whether to swap the colors for the maximin search')
    parser.add_argument('--prompt_to_step', type=bool, default=False, \
        help='Whether to prompt the user to step through each move')
    parser.add_argument('--print_tree', type=bool, default=False, \
        help='Whether to print the tree of MCTS iterations')
    parser.add_argument('--render_board', type=bool, default=False, \
        help='Whether to render the board')
    parser.add_argument('--num_games', type=int, default=16, \
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

    PATH = './trained_models/'
    Q_net = Othello_QNet(args['board_size'])
    target_Q_net = None

    # Save initial Q_net
    Q_net_iter = 0
    torch.save(Q_net.state_dict(), PATH + 
                'board' + str(board_size) + '_' +
                'mcts-iter'+ str(mcts_iterations) + '_' +
                'targetQdeep' + '_' + 
                'correctCopying' + 
                'iter' + str(Q_net_iter)
                )

    # training the Q_network 
    sgd_iter = 1000
    batch_size = 1024
    mse_loss = torch.nn.MSELoss()
    data_buffer = ReplayBuffer()
    for i in range(50):
        print(f'Loading buffer for iteration {i}')
        data_buffer.merge(simulate_mcts(args, target_Q_net))
        data_buffer.shuffle()
        # data_buffer = simulate_mcts(args, target_Q_net)

        print(f'Training on buffer for iteration {i},\
             buffer size = {len(data_buffer.buffer)}')

        optimizer = torch.optim.Adam(Q_net.parameters(), lr=0.01, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        loss_i = []
        for j in tqdm(range(sgd_iter)):
            # Load a batch of data
            ob, ac, va = data_buffer.sample(batch_size)

            ac = torch.from_numpy(ac).float()
            va = torch.from_numpy(va).float()
            ob = torch.from_numpy(ob).float()

            # Debug info
            # print(ac.shape, va.shape, ob.shape)
            # sleep(0.5)

            pred_va, pred_ac = Q_net(ob)
            pred_va = torch.squeeze(torch.flatten(pred_va))

            if ac.shape != pred_ac.shape: 
                print("skipped a loop iteration due to mismatch in ac and pred_ac shape")
                print('ac:', ac.shape, '\npred_ac:', pred_ac.shape)
                continue

            loss_va = mse_loss(va, pred_va)
            loss_ac = mse_loss(ac, pred_ac)
            loss = loss_va + loss_ac

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_i.append(loss.item())
        
        loss_i = np.array(loss_i)
        np.save(PATH + 'loss' + str(i), loss_i)
        
        # if Q_net_iter == 0:
        #     Q_net_iter += 1
        #     torch.save(Q_net.state_dict(), PATH + 
        #         'board' + str(board_size) + '_' +
        #         'mcts-iter'+ str(mcts_iterations) + '_' +
        #         'targetQ' + '_' + 
        #         'iter' + str(Q_net_iter)
        #         )
        #     # Update the target Q_net 
        #     target_Q_net = Q_net
        #     # Reset the data buffer
        #     data_buffer.reset()
        #     print(f'Target Q_net has been updated! On Q_net_iter = {Q_net_iter}')
        # else:
        results = compare_mcts(args, Q_net, opponent='mcts', opponent_Q_net=target_Q_net)
        winrate = np.average(np.array(results))
        print(f'\ncurrent Q_net has {winrate} against target Q_net.')
        if winrate > 0.55:
            Q_net_iter += 1
            torch.save(Q_net.state_dict(), PATH + 
                'board' + str(board_size) + '_' +
                'mcts-iter'+ str(mcts_iterations) + '_' +
                'targetQdeep' + '_' + 
                'correctCopying' + 
                'iter' + str(Q_net_iter)
                )
            # Update the target Q_net 
            if target_Q_net is None:
                target_Q_net = Othello_QNet(board_size=board_size)
            target_Q_net.load_state_dict(Q_net.state_dict())
            # Reset the data buffer
            data_buffer.reset()
            print(f'Target Q_net has been updated! On Q_net_iter = {Q_net_iter}')

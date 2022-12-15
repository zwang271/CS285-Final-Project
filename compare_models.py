import numpy as np
import argparse
import multiprocessing
from time import time
import gym
from game_utils import * 
from policies import mcts
from policies.mcts import copy_env, ReplayBuffer, BLACK_DISK, NO_DISK, WHITE_DISK
from tqdm import tqdm
from policies.model import Othello_QNet
import torch


def simulate_game(queue, args, Q_net, opponent, protagonist_plays_white=True, opponent_Q_net=None):
    '''
    appends 0 to Queue if protagonist MCTS augmented with Q_net lost
    otherwise appends a 1 to Queue if protagonist won
    '''
    # Parse args
    board_size = args['board_size']
    seed = args['seed']
    mcts_iterations = args['mcts_iterations']
    mcts_c = args['mcts_c']
    initial_rand_steps = args['initial_rand_steps']
    num_disk_as_reward = args['num_disk_as_reward']
    maximin_depth = args['maximin_depth']
    prompt_to_step = args['prompt_to_step']
    print_tree = args['print_tree']
    render_board = args['render_board']
    mute = args['mute']
    num_games = args['num_games']
    debug = args['debug']

    # Initialize Othello environment
    env = othello.OthelloBaseEnv(
            board_size=board_size, 
            num_disk_as_reward=num_disk_as_reward,
            mute=mute
            )
    obs = env.reset()
    
    # Initialize policies for protagonist and opponen and determine which 
    # policies are mcts, which requires updating every move
    protagonist_policy = mcts.MCTS(copy_env(env), 
                            iterations=mcts_iterations, 
                            Q_net=Q_net, 
                            c=mcts_c)
    protagonist_policy.reset(copy_env(env))

    mcts_policies = [protagonist_policy]
    
    if opponent == 'rand':
        opponent_policy = simple_policies.RandomPolicy()
        opponent_policy.reset(env)
    elif opponent == 'maximin':
        opponent_policy = simple_policies.MaxiMinPolicy(max_search_depth=maximin_depth)
        opponent_policy.reset(env)
    elif opponent == 'mcts':
        opponent_policy = mcts.MCTS(copy_env(env), 
                            iterations=mcts_iterations, 
                            Q_net=opponent_Q_net, 
                            c=mcts_c)
        opponent_policy.reset(copy_env(env))
        mcts_policies.append(opponent_policy)


    while not env.terminated:
        if env.player_turn == WHITE_DISK:
            if protagonist_plays_white:
                action = protagonist_policy.get_action()
            else: # opponennt plays white
                if opponent == 'mcts':
                    action = opponent_policy.get_action()
                else: # opponent is not mcts
                    action = opponent_policy.get_action(obs)
        else: #env.player_turn == BLACK_DISK
            if protagonist_plays_white: # it's opponenent's turn
                if opponent == 'mcts':
                    action = opponent_policy.get_action()
                else: # opponent is not mcts
                    action = opponent_policy.get_action(obs)
            else: # protagonist plays black
                action = protagonist_policy.get_action()

        obs, _, _, _ = env.step(action)
        for policy in mcts_policies:
            policy.update_root(action)
    
    # print(env.winner, protagonist_plays_white)

    if env.winner == WHITE_DISK:
        if protagonist_plays_white: # protagonist won
            queue.put(1)
        else: # protagonist lost
            queue.put(0)
    else: #env.winner == BLACK_DISK
        if protagonist_plays_white: # protagonist lost
            queue.put(0)
        else: # protagonist won
            queue.put(1)

        

def compare_mcts(args, Q_net, opponent, opponent_Q_net=None):
    '''
    Compares MCTS augmented with a Q_net vs a opponent policy and returns winrate

    Args:
        args: a dictionary containing the following keys and values:
            board_size (int): the size of the Othello board.
            seed (int): the seed for the random number generator.
            mcts_iterations (int): the number of iterations to run MCTS for each move.
            initial_rand_steps (int): the number of initial random steps to take.
            num_disk_as_reward (bool): a flag indicating whether to use the number of disks as the reward.
            maximin_depth (int): the depth to search for the maximin policy.
            swap_colors (bool): a flag indicating whether to swap the colors played by the policies after each game.
            prompt_to_step (bool): a flag indicating whether to prompt the user to step through the game.
            print_tree (bool): a flag indicating whether to print the MCTS at the end of a game
            render_board (bool): a flag indicating whether to render the Othello board.
            debug (bool): a flag indicating whether to print debugging information.
            num_games (int): the number of games to simulate.

    Returns:
        
    '''

    assert(opponent in ['rand', 'maximin', 'mcts'])

    board_size = args['board_size']
    seed = args['seed']
    mcts_iterations = args['mcts_iterations']
    initial_rand_steps = args['initial_rand_steps']
    num_disk_as_reward = args['num_disk_as_reward']
    maximin_depth = args['maximin_depth']
    swap_colors = args['swap_colors']
    prompt_to_step = args['prompt_to_step']
    print_tree = args['print_tree']
    render_board = args['render_board']
    debug = args['debug']
    num_games = args['num_games']


    results = []
    job_batch_size = 8
    protagonist_plays_white = True
    for _ in tqdm(range(num_games // job_batch_size),
                  desc='Processing jobs: '):
        
        q = multiprocessing.Queue()
        processes = []
        for i in range(job_batch_size):
            protagonist_plays_white = not protagonist_plays_white
            p = multiprocessing.Process(
                target=simulate_game, args=(q, args, Q_net, opponent, protagonist_plays_white, opponent_Q_net
                ))
            processes.append(p)
            p.start()
        
        for p in processes:
            ret = q.get()
            results.append(ret)
    
        for p in processes:
            p.join

    results = np.array(results)
    return results

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


    # Evaluate Q_net models against rand and maximin
    PATH = './trained_models/mcts-iter100_num-games32_useLRSchedule/'
    rand_results = []
    maximin_results = []
    for iter in range(5):
        Q_net_file = 'board4_mcts-iter100_targetQdeep_correctCopyingiter' + str(iter)
        Q_net = Othello_QNet(args['board_size'])
        Q_net.load_state_dict(torch.load(PATH + Q_net_file))
        Q_net.eval()

        rand = compare_mcts(args, Q_net, opponent='rand')
        maximin = compare_mcts(args, Q_net, opponent='maximin')
        rand_results.append(rand)
        maximin_results.append(maximin)
        print(f'rand{iter}:', rand, np.sum(rand)/len(rand))
        print(f'maximin{iter}', maximin, np.sum(maximin)/len(maximin))

    rand_results = np.array(rand_results)
    maximin_results = np.array(maximin_results)
    np.save(PATH + 'rand_results', rand_results)
    np.save(PATH + 'maximin_results', maximin_results)
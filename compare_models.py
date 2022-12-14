import numpy as np
import argparse
import multiprocessing
from time import time
import gym
from game_utils import * 
from policies import mcts
from policies.mcts import copy_env, ReplayBuffer, BLACK_DISK, NO_DISK, WHITE_DISK


num_iter = 5_000_000
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
    env.reset()
    
    # Initialize policies for protagonist and opponen and determine which 
    # policies are mcts, which requires updating every move
    protagonist_policy = mcts.MCTS(copy_env(env), 
                            iterations=mcts_iterations, 
                            Q_net=Q_net, 
                            c=mcts_c)
    protagonist_policy.reset(copy_env(env))

    mcts_policies = [protagonist_policy]
    
    if opponent == 'rand':
        opponent_policy = simple_policies.RandomPolicy(seed=i)
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


    # Assign policies to correct colors
    if protagonist_plays_white:
        white_policy = protagonist_policy
        black_policy = opponent_policy
    else:
        black_policy = protagonist_policy
        white_policy = opponent_policy

    

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
    if opponent == 'mcts':
        assert(opponent_Q_net is not None)

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

    q = multiprocessing.Queue()
    processes = []
    results = []

    protagonist_plays_white = True
    for i in range(args['num_games']):
        protagonist_plays_white = not protagonist_plays_white
        p = multiprocessing.Process(
            target=simulate_game, args=(q, Q_net, opponent, protagonist_plays_white, opponent_Q_net
            ))
        processes.append(p)
        p.start()

    for p in processes:
        ret = q.get()
        results.append(ret)
    
    for p in processes:
        p.join
    results = np.array(results)
    print(results.shape)

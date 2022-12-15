import gym
from game_utils import * 
import numpy as np
from policies import mcts
from policies.mcts import copy_env, ReplayBuffer, BLACK_DISK, NO_DISK, WHITE_DISK
import multiprocessing
from tqdm import tqdm

def fill_buffer(queue, args, Q_net = None):
    # print("started game sim")
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

    # Create a training buffer
    RB = ReplayBuffer()

    # Game play 
    winner = {'BLACK': 0, 'WHITE': 0, 'DRAW': 0}

    # Initialize Othello environment
    env = othello.OthelloBaseEnv(
        board_size=board_size, 
        num_disk_as_reward=num_disk_as_reward,
        mute=mute
        )
    env.reset()

    rand_policy = simple_policies.RandomPolicy()
    rand_policy.reset(env)

    # Randomly play i number of moves:
    # for _ in range(i):
    #     action = rand_policy.get_action()
    #     env.step(action)

    # Initialize player policies
    mcts_policy = mcts.MCTS(copy_env(env), 
                            iterations=mcts_iterations, 
                            Q_net=Q_net, 
                            c=mcts_c)

    mcts_policy.reset(copy_env(env))
    root = mcts_policy.root
    game_history = []
    while not env.terminated:
            # Render the board
        if render_board and debug:
            env.render()
        if prompt_to_step and debug:
            input("Press enter to step.")
        
        # Get next move
        action = mcts_policy.get_action()

        # Step the environment and update MCTS
        obs, _, _, _ = env.step(action)
        mcts_policy.update_root(action)
        game_history.append(action)
        # print(f'policy actions/keys: {policy.env.get_possible_actions()} {policy.root.child.keys()}')

    # Update and Report Statistics
    if env.winner == BLACK_DISK:
            winner['BLACK'] += 1
    elif env.winner == WHITE_DISK:
        winner['WHITE'] += 1
    else:
        winner['DRAW'] += 1

    if debug:    
        print(winner)
        if print_tree:
            print(root.draw_tree())
        print(root.get_size())

    if render_board and debug:
        env.render()
    if prompt_to_step and debug:
        input("Press enter to restart")
    env.close()

    # Store moves from game tree into replay buffer
    RB.store(root, env.winner)

    # Put returns to queue
    queue.put((RB, winner, game_history))


def simulate_mcts(args, Q_net = None):
    '''
    Simulates a number of Othello games using Monte Carlo Tree Search (MCTS) and stores the game results in a replay buffer.

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
        ReplayBuffer: a replay buffer containing the results of the simulated games.
    '''

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

    # Create a training buffer
    RB = ReplayBuffer()

    # Game play 
    stats = {'BLACK': 0, 'WHITE': 0, 'DRAW': 0}
    history = []

    job_batch_size = 8
    for _ in tqdm(range(num_games // job_batch_size)):
        q = multiprocessing.Queue()
        processes = []
        for i in range(job_batch_size):
            p = multiprocessing.Process(
                target=fill_buffer,
                args=(q, args, Q_net)
            )
            processes.append(p)
            p.start()
        
        for p in processes:
            buffer, winner, game_history = q.get()
            RB.merge(buffer)
            history.append(game_history)

            for key in stats.keys():
                stats[key] += winner[key]

        for p in processes:
            p.join
    
    print(stats)
    # for game in history:
    #     print(game)

    print(len(RB.buffer))
    RB.shuffle()

    return RB



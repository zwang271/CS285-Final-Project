import gym
from game_utils import * 
import numpy as np
from policies import mcts
from policies.mcts import copy_env
from policies.mcts import ReplayBuffer

# Game Constants
BLACK_DISK = -1
NO_DISK = 0
WHITE_DISK = 1

# ARGS
board_size = 6
seed = 0
mcts_iterations = 200
initial_rand_steps = 0
num_disk_as_reward = False
prompt_to_step = False


# Game play 
while True:
    # Initialize Othello environment
    env = othello.OthelloBaseEnv(board_size=board_size, num_disk_as_reward=num_disk_as_reward)

    # Initialize player policies
    black_policy = mcts.MCTS(copy_env(env), iterations = mcts_iterations)
    white_policy = mcts.MCTS(copy_env(env), iterations = mcts_iterations)

    env.reset()
    black_policy.reset(copy_env(env))
    white_policy.reset(copy_env(env))

    while not env.terminated:
        # print(f'\n current turn: {env.player_turn}')
        # print(f'possible moves: {env.get_possible_actions()}')

        env.render()
        if prompt_to_step:
            input("Press enter to step.")
        
        print_tree = False
        if env.player_turn == BLACK_DISK:
            # print('BLACK TURN')
            action = black_policy.get_action(print_tree)
            print(f'black tree size = {black_policy.tree_size()}')
        else: # WHITE_DISK turn
            # print('WHITE TURN')
            action = white_policy.get_action(print_tree)
            print(f'white tree size = {white_policy.tree_size()}')

        # action = np.random.choice(env.get_possible_actions())

        print(f'action taken: {action}')

        env.step(action)
        # Update policy envs and reassign roots
        black_policy.env.step(action)
        white_policy.env.step(action)
        print(f'black: {black_policy.env.get_possible_actions()} {black_policy.root.child.keys()}')
        print(f'white: {white_policy.env.get_possible_actions()} {white_policy.root.child.keys()}')

        for policy in [black_policy, white_policy]:
            if action not in policy.root.child.keys():
                policy.root.expand()
            policy.root = policy.root.child[action]

        winner = env.winner

    env.render()
    # input("Press enter to restart")
    env.close()
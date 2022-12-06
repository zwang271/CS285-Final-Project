import argparse
import gym
from game_utils import * 
from policies import mcts

def train_mcts(args):
    print(f'\nTraining MCTS')
    print(f'opponent = {args["opponent"]}')
    if args["opponent"] == 'maximin':
        print(f'\topponent_search_depth = {args["opponent_search_depth"]}')
        print(f'\topponent_use_ab = {args["opponent_disable_ab"]}')
    print(f'board size = {args["board_size"]}')
    print(f'num_disk_as_reward = {args["num_disk_as_reward"]}')
    print(f'rand_seed = {args["rand_seed"]}')
    print(f'init_rand_steps = {args["init_rand_steps"]}')
    print(f'train_rounds = {args["train_rounds"]}')
    print(f'mcts_iterations = {args["mcts_iterations"]}')
    print("-"*60)

    protagonist_policy = create_policy(
        policy_type='mcts',
        board_size=args['board_size'],
        seed=args['rand_seed'],
        mcts_iterations = args["mcts_iterations"]
        )
    opponent_policy = create_policy(
        policy_type=args["opponent"],
        board_size=args["board_size"],
        seed=args['rand_seed'],
        search_depth=args['opponent_search_depth'],
        use_ab=args['opponent_disable_ab'],
        mcts_iterations = args["mcts_iterations"]
        )

    env = othello.OthelloEnv(white_policy=opponent_policy,
                             black_policy=protagonist_policy,
                             protagonist=-1,
                             board_size=args['board_size'],
                             seed=args['rand_seed'],
                             initial_rand_steps=args['init_rand_steps'],
                             num_disk_as_reward=args['num_disk_as_reward'],
                             render_in_step=True)

    for i in range(args['train_rounds']):
        print(f'\n Round number {i+1} \n {"-"*60}')
        env.reset()
        protagonist_policy.reset(env)
        env.render()
        done = False

        while not done:
            action = protagonist_policy.get_action(env)
            obs, reward, done, _ = env.step(action)
            env.render()
            protagonist_policy.visualize()
    
    env.close()


if __name__ == '__main__':
    # Parsing arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--opponent', default='rand', choices=['rand', 'maximin', 'mcts'])
    parser.add_argument('--board_size', default='6', type=int)
    parser.add_argument('--num_disk_as_reward', default=False, action='store_true')
    parser.add_argument('--rand_seed', default=0, type=int)
    parser.add_argument('--opponent_search_depth', default=3, type=int)
    parser.add_argument('--opponent_disable_ab', default=True, action='store_false')
    parser.add_argument('--init_rand_steps', default=0, type=int)
    parser.add_argument('--train_rounds', default=10, type=int)
    parser.add_argument('--mcts_iterations', default=100, type=int)

    parser = parser.parse_args()
    args = vars(parser)

    train_mcts(args)
import argparse
import gym
from game_utils import * 
from policies import mcts


if __name__ == '__main__':
    # Parsing arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--protagonist', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human', 'mcts'])
    parser.add_argument('--opponent', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human', 'mcts'])
    parser.add_argument('--protagonist-plays-white', default=False,
                        action='store_true')
    parser.add_argument('--num-disk-as-reward', default=False,
                        action='store_true')
    parser.add_argument('--board-size', default=8, type=int)
    parser.add_argument('--protagonist-search-depth', default=1, type=int)
    parser.add_argument('--opponent-search-depth', default=1, type=int)
    parser.add_argument('--rand-seed', default=0, type=int)
    parser.add_argument('--num-rounds', default=100, type=int)
    parser.add_argument('--init-rand-steps', default=0, type=int)
    parser.add_argument('--no-render', default=False, action='store_true')
    parser.add_argument('--use-ab', default=True, action='store_false')
    args, _ = parser.parse_known_args()


    protagonist = 1 if args.protagonist_plays_white else -1
    protagonist_agent_type = args.protagonist
    opponent_agent_type = args.opponent
    play(protagonist=protagonist,
         protagonist_agent_type=protagonist_agent_type,
         opponent_agent_type=opponent_agent_type,
         board_size=args.board_size,
         num_rounds=args.num_rounds,
         protagonist_search_depth=args.protagonist_search_depth,
         opponent_search_depth=args.opponent_search_depth,
         rand_seed=args.rand_seed,
         env_init_rand_steps=args.init_rand_steps,
         num_disk_as_reward=args.num_disk_as_reward,
         render=not args.no_render,
         use_ab=args.use_ab)
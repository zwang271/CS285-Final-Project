from envs.GymOthelloEnv import othello
from policies import simple_policies
from policies import mcts
from policies import model
# from GymOthelloEnv import simple_policies

def create_policy(policy_type='rand', board_size=8, seed=0, search_depth=1, use_ab=True, mcts_iterations=100):
    if policy_type == 'rand':
        policy = simple_policies.RandomPolicy(seed=seed)
    elif policy_type == 'greedy':
        policy = simple_policies.GreedyPolicy()
    elif policy_type == 'maximin':
        policy = simple_policies.MaxiMinPolicy(search_depth, use_ab)
    elif policy_type == 'mcts':
        Q_Net = model.Othello_QNet(board_size)
        policy = mcts.MCTS(iterations = mcts_iterations, Q_net = None)
    else:
        policy = simple_policies.HumanPolicy(board_size)
    return policy


def play(protagonist,
         protagonist_agent_type='greedy',
         opponent_agent_type='rand',
         board_size=8,
         num_rounds=100,
         protagonist_search_depth=1,
         opponent_search_depth=1,
         rand_seed=0,
         env_init_rand_steps=0,
         num_disk_as_reward=False,
         render=True,
         use_ab=True):
    print('protagonist: {}'.format(protagonist_agent_type))
    print('opponent: {}'.format(opponent_agent_type))

    protagonist_policy = create_policy(
        policy_type=protagonist_agent_type,
        board_size=board_size,
        seed=rand_seed,
        search_depth=protagonist_search_depth,
        use_ab=use_ab)
    opponent_policy = create_policy(
        policy_type=opponent_agent_type,
        board_size=board_size,
        seed=rand_seed,
        search_depth=opponent_search_depth,
        use_ab=use_ab)

    if protagonist == 1:
        white_policy = protagonist_policy
        black_policy = opponent_policy
    else:
        white_policy = opponent_policy
        black_policy = protagonist_policy

    if opponent_agent_type == 'human':
        render_in_step = True
    else:
        render_in_step = True

    env = othello.OthelloEnv(white_policy=white_policy,
                             black_policy=black_policy,
                             protagonist=protagonist,
                             board_size=board_size,
                             seed=rand_seed,
                             initial_rand_steps=env_init_rand_steps,
                             num_disk_as_reward=num_disk_as_reward,
                             render_in_step=render_in_step and render)

    win_cnts = draw_cnts = lose_cnts = 0
    for i in range(num_rounds):
        print('Episode {}'.format(i + 1))
        obs = env.reset()
        protagonist_policy.reset(env)
        if render:
            env.render()
        done = False
        while not done:
            action = protagonist_policy.get_action(env)
            obs, reward, done, _ = env.step(action)
            # print(obs)
            if render:
                env.render()
            if done:
                print('reward={}'.format(reward))
                if num_disk_as_reward:
                    total_disks = board_size ** 2
                    if protagonist == 1:
                        white_cnts = reward
                        black_cnts = total_disks - white_cnts
                    else:
                        black_cnts = reward
                        white_cnts = total_disks - black_cnts

                    if white_cnts > black_cnts:
                        win_cnts += 1
                    elif white_cnts == black_cnts:
                        draw_cnts += 1
                    else:
                        lose_cnts += 1
                else:
                    if reward == 1:
                        win_cnts += 1
                    elif reward == 0:
                        draw_cnts += 1
                    else:
                        lose_cnts += 1
                print('-' * 3)
    print('#Wins: {}, #Draws: {}, #Loses: {}'.format(
        win_cnts, draw_cnts, lose_cnts))
    env.close()
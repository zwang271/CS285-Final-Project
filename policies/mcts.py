import numpy as np
from envs.GymOthelloEnv import othello
import math
import torch
from copy import deepcopy

def copy_env(env):
    new_env = deepcopy(env)
    new_env.mute = True
    return new_env

class Node():

    def __init__(self, env, Q_net=None, c=1, depth=0, parent=None):
        
        self.env = env
        self.turn = env.player_turn
        self.value = 0 # Number of wins this node has
        self.count = 0 # Number of times this node has been visited
        self.parent = parent
        self.child = {}
        self.Q_net = Q_net
        self.c = c
        self.depth = depth

    def ucb(self):
        '''
        Computes the UCB1 value of this node using the formula:
        UCB1(node) = exloitation + exploration 
        UCB1(node) = wins / (visits) + sqrt(ln(parent_visits) / (visits))
        '''
        if self.parent is None:
            return 0

        eps = 1e-8
        # Vanilla  MCTS
        if self.Q_net is None:
            exploration_score = self.c * np.sqrt(self.parent.count / (self.count + eps))
            exploitation_score = self.value / (self.count + eps)
        
        # MCTS Augmented with a Neural Network
        else:
            pass
            # exploration_score = child.prior * np.sqrt(parent.count / (child.count + 1))
            # obs = torch.from_numpy(self.env.get_observation(separate=True)).float()
            # obs = obs.view(-1, 4, self.Q_net.board_size, self.Q_net.board_size)
            # # print(obs)
            # value, ac_probs = self.Q_net(obs)
            # value = value.item()
            # ac_probs = torch.flatten(ac_probs)
            # # print(value, ac_probs)
            # exploitation_score, ac_probs = value, ac_probs

        return exploitation_score + exploration_score

    def expand(self):
        '''
        Gets all legal moves from current board and adds to the
        tree as new nodes
        '''
        # print('\n', self.env.get_possible_actions())
        for ac in self.env.get_possible_actions():
            new_env = copy_env(self.env)
            # print(ac, new_env.get_possible_actions(), f'turn={new_env.player_turn}')
            new_env.step(ac)
            # print('\t', new_env.get_possible_actions(), f'turn={new_env.player_turn}')
            self.child[ac] = Node(
                env=new_env,
                Q_net=self.Q_net, 
                c=self.c,
                depth=self.depth+1, 
                parent=self
                )

    def select_child(self):
        '''
        Finds the action with the highest UCB1 score
        return: action, child
        '''
        max_score = -math.inf
        selected_ac, selected_child = None, None
        for ac in self.child.keys():
            child = self.child[ac]
            score = child.ucb()
            if score > max_score:
                max_score = score
                selected_child = child
                selected_ac = ac
        return selected_ac, selected_child

    def rollout(self):
        '''
        Performs a rollout from this game state
        return 1 if won, 0 if lost
        '''
        new_env = copy_env(self.env)

        while not new_env.terminated:
            # Light rollout
            if self.Q_net is None:
                action = np.random.choice(new_env.get_possible_actions())
            else:
                action # TODO
            new_env.step(action)

        win = new_env.winner * self.turn
        if win == 1:
            reward = 1
        elif win == 0:
            reward = 0.5
        else: #win == -1
            reward = 0

        return reward
    
    def draw_tree(self):
        tree = '  '*self.depth + f'{self.count} {self.ucb()} {self.value} \n'
        for node in self.child.values():
            tree = tree + node.draw_tree()
        return tree
    
    def get_size(self):
        total = 0
        to_visit = [self]
        while len(to_visit) > 0:
            current = to_visit.pop(0)
            total += 1
            to_visit.extend([current.child[key] for key in current.child.keys()])
        return total

class MCTS():

    def __init__(self, env, iterations=100, c=1, Q_net=None):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env
    
        self.root = Node(copy_env(env), Q_net, c)
        self.c = c
        self.iterations = iterations
        self.Q_net = Q_net

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env
        self.root = Node(copy_env(env), self.Q_net, self.c)

    def get_action(self, print_tree=False):
        '''
        Performs the MCTS algorithm for a fixed number of iterations
        1. Tree traversal using UCB1 value to find leaf node
        2. Expansion of leaf node 
        3. Rollout node
        4. Back propagation
        '''
        for i in range(self.iterations):
            current = self.root

            # Keep track of history of nodes for back propagation
            search_path = [current]

            while len(current.child) > 0:
                _, current = current.select_child()
                search_path.append(current)

            if not current.env.terminated:
                current.expand()
                _, current = current.select_child()
                search_path.append(current)

            value = current.rollout()
            for node in search_path:
                node.count += 1 
                node.value += value
        
        if print_tree:
            self.visualize()

        # if self.Q_net == None:
        #     value, ac_probs = 0, None
        # else:
        #     obs = torch.from_numpy(self.env.get_observation(separate=True)).float()
        #     obs = obs.view(-1, 4, self.Q_net.board_size, self.Q_net.board_size)
        #     # print(obs)
        #     value, ac_probs = self.Q_net(obs)
        #     value = value.item()
        #     ac_probs = torch.flatten(ac_probs)
        #     # print(value, ac_probs)
        
        # self.root = Node(env, 0, self.Q_net)
        # self.root.expand(ac_probs)
        
        # for _ in range(self.iterations):
        #     node = self.root
        #     search_path = [node]
        #     while len(node.child) > 0:
        #         ac, node = node.select_child()
        #         search_path.append(node)

        #     value = self.env.winner
        #     if not self.env.terminated:
        #         obs = torch.from_numpy(self.env.get_observation(separate=True)).float()
        #         obs = obs.view(-1, 4, self.Q_net.board_size, self.Q_net.board_size)
        #         # print(obs)
        #         value, ac_probs = self.Q_net(obs)
        #         value = value.item()
        #         ac_probs = torch.flatten(ac_probs)
        #         # print(value, ac_probs)
        #         node.expand(ac_probs)
            
        #     for node in search_path:
        #         node.value += value
        #         node.count += 1
        
        # action_dist = torch.distributions.categorical.Categorical(self.root.ac_probs)
        # move = action_dist.sample().item()
        # print(move)

        move, _ = self.root.select_child()
        return move
    
    def visualize(self):
        print(self.root.draw_tree())

    def tree_size(self):
        return self.root.get_size()


class ReplayBuffer():
    '''
    Stores tuples of (obs, ac, value, )
    '''
    def __init__(self, capacity=10_000):
        self.capacity = capacity  
        self.buffer = []  

    def reset(self):
        '''
        Resets the buffer
        '''

    def sample(self, num_samples):
        '''
        Returns a subset of the buffer of size num_samples
        '''
        pass

    def store(self, tree: MCTS):
        '''
        Stores all moves in MCTS into the buffer
        '''


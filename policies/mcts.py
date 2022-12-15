import numpy as np
from envs.GymOthelloEnv import othello
import math
import torch
from scipy.special import softmax
from policies import simple_policies
from collections import OrderedDict
from time import time

# Game Constants
BLACK_DISK = -1
NO_DISK = 0
WHITE_DISK = 1

global copy_time
copy_time = 0


def copy_env(env):
    global copy_time
    start = time()
    new_env = env.copy()
    copy_time += time() - start
    return new_env

class Node():

    def __init__(self, env, color = WHITE_DISK, Q_net=None, c=0.5, depth=0, parent=None, move=None):
        
        self.env = env
        self.color = color
        self.count = 0 # Number of times this node has been visited
        self.win = 0 # Number of wins this node has
        self.parent = parent
        self.child = OrderedDict()
        self.c = c
        self.depth = depth
        self.move = move
        self.Q_net = Q_net
        self.selected = None

        if Q_net is not None:
            obs = torch.from_numpy(self.env.get_observation(separate=True)).float()
            self.value, self.ac_dist = Q_net.evaluate(obs)
            self.ac_dist = self.ac_dist.reshape(-1)

    def ucb(self):
        '''
        Computes the UCB1 or PUCB value of this node using the formula:
        UCB1(node) = exploitation + exploration 
        '''
        if self.parent is None:
            return 0

        eps = 1e-8
        # Vanilla  MCTS
        if self.Q_net is None:
            exploration_score = self.c * np.sqrt(self.parent.count / (self.count + eps))
            exploitation_score = self.win / (self.count + eps)
        
        # MCTS Augmented with a Neural Network
        else:
            exploration_score = self.c * self.parent.ac_dist[self.move] * np.sqrt(self.parent.count) / (self.count + 1)
            # print(exploration_score)
            exploitation_score = self.win / (self.count + eps)

        return exploitation_score + exploration_score

    def expand(self):
        '''
        Gets all legal moves from current board and adds to the
        tree as new nodes
        '''
        # print('\n', self.env.get_possible_actions())
        for ac in self.env.possible_moves:
            new_env = copy_env(self.env)
            # print(ac, new_env.get_possible_actions(), f'turn={new_env.player_turn}')
            new_env.step(ac)
            # print('\t', new_env.get_possible_actions(), f'turn={new_env.player_turn}')
            self.child[ac] = Node(
                env=new_env,
                color = -self.color,
                Q_net=self.Q_net, 
                c=self.c,
                depth=self.depth+1, 
                parent=self,
                move=ac
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
        return color of winner
        '''
        if self.env.terminated:
            if self.env.winner == self.color:
                self.win = math.inf
                self.parent.win = -10000
            elif self.env.winner == -self.color:
                self.win = -10000
            return self.env.winner

        new_env = copy_env(self.env)
        obs = torch.from_numpy(new_env.get_observation(separate=True)).float()
        while not new_env.terminated:
            # Light rollout
            if self.Q_net is None:
                # action = maximin_policy.get_action(obs)
                action = np.random.choice(new_env.possible_moves)
            else:
                action = self.Q_net.get_action(obs)
                # action = np.random.choice(new_env.possible_moves)
                # print(action, new_env.get_possible_actions())
            obs = torch.from_numpy(new_env.step(action)[0]).float()

        return new_env.winner

    def update(self, winner = None, leaf_color = None, leaf_v = None):
        '''
        Updates the node with the result of a game.

        Args:  
            winner: the winner of the game, as represented by a value from the game's state.
            leaf_color: the color of the leaf node found at expansion phase in mcts
        '''
        self.count += 1
        
        if winner is not None:
            if winner == self.color:
                self.win += 1
            else: #winner == -self.color:
                self.win -= 1
        else: # leaf_color is not none
            if self.env.terminated:
                if self.env.winner == self.color:
                    self.win = 10000
                    self.parent.win = -10000
                elif self.env.winner == -self.color:
                    self.win = -10000
            # CRITICAL
            if leaf_color == self.color:
                self.win += leaf_v
            else: #leaf_color == -self.color:
                self.win -= leaf_v
    
    def draw_tree(self):
        '''
        Returns a string representation of the MCTS tree rooted at this node.
        The representation includes the node's depth, count, UCB value, and value.

        Returns:
            str: the string representation of the tree.
        '''
        tree = '  '*self.depth + f'{self.count} {self.ucb()} {self.win} \n'
        for node in self.child.values():
            tree = tree + node.draw_tree()
        return tree
    
    def get_size(self):
        '''
        Returns the total number of nodes in the MCTS tree rooted at this node.

        Returns:
            int: the number of nodes in the tree.
        '''
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
    
        self.root = Node(copy_env(env), Q_net=Q_net, c=c)
        self.c = c
        self.iterations = iterations
        self.Q_net = Q_net

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env
        self.root = Node(copy_env(env), Q_net=self.Q_net, c=self.c)

    def get_action(self, print_tree=False):
        '''
        Performs the MCTS algorithm for a fixed number of iterations
        1. Tree traversal using UCB1 value to find leaf node
        2. Expansion of leaf node 
        3. Rollout node
        4. Back propagation
        '''
        # add dirichlet noise for more exploration with Q_net
        if self.Q_net is not None:
            eps = 0.25 * np.random.rand()
            board_size = self.env.board_size
            self.root.ac_dist = ((1-eps)*self.root.ac_dist + 
                                (eps)*(1/board_size**2)*torch.ones(board_size**2))

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
            
            if self.Q_net is None:
                winner = current.rollout()
                for node in search_path:
                    node.update(winner=winner)
            else:
                leaf_color = current.color
                leaf_v = current.value
                for node in search_path:
                    node.update(leaf_color=leaf_color, leaf_v=leaf_v)
        
        if print_tree:
            self.visualize()

        if self.Q_net is None:
            move, child = self.root.select_child()
            self.root.selected = child
        else:
            t = 0.25
            actions = self.root.env.possible_moves
            ac_dist = np.zeros(len(actions))
            for i in range(len(actions)):
                ac = actions[i]
                ac_dist[i] = self.root.child[ac].count
            ac_dist = ac_dist ** (1/t) / np.sum(ac_dist ** (1/t))
            move = np.random.choice(actions, 1, p=ac_dist).item()
            # print(move, actions, ac_dist)
            self.root.selected = self.root.child[move]
        return move

    def update_root(self, move):
        self.env.step(move)
        if move not in self.root.child.keys():
            self.root.expand()
        self.root = self.root.child[move]
    
    def visualize(self):
        print(self.root.draw_tree())

    def tree_size(self):
        return self.root.get_size()

    def print_copy_time(self):
        print(f'copy time: {copy_time}')


class ReplayBuffer():
    '''
    Stores tuples of (obs, ac, value, )
    '''
    def __init__(self, capacity=30_000):
        self.capacity = capacity  
        self.buffer = [] 
        self.last_sample = 0 

    def reset(self):
        '''
        Resets the buffer
        '''
        self.buffer = []

    def merge(self, RB):
        self.buffer.extend(RB.buffer)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def shuffle(self):
        np.random.shuffle(self.buffer)

    def sample(self, num_samples=1000):
        '''
        Returns a random subset of the buffer of size num_samples.
        If num_samples is greater than the length of the buffer, 
        returns the entire buffer.

        Returns:
            tuple: (observations, actions, values)
        '''
        num_samples = min(num_samples, len(self.buffer))
        observations, actions, values = [], [], []

        end = min(self.last_sample+1000, len(self.buffer))
        for obs, ac, value in self.buffer[self.last_sample:end]:
            observations.append(obs)
            actions.append(ac)
            values.append(value)
        self.last_sample = end
        if end == len(self.buffer):
            self.last_sample = 0
            self.shuffle()
        return np.array(observations), np.array(actions), np.array(values)

    def store(self, root: Node, winner):
        '''
        Stores all moves in MCTS into the buffer.

        Args:
            root: the root node of the MCTS tree.
        '''
        current = root
        board_size = root.env.board_size
        ac_dist_size = board_size**2

        while current.selected is not None:
            ac_dist = np.zeros(ac_dist_size)
            for ac, node in current.child.items():
                ac_dist[ac] = node.count

            ac_dist = ac_dist / np.sum(ac_dist)
            t = 0.25
            ac_dist = ac_dist ** (1/t) / np.sum(ac_dist ** (1/t))

            ac_dist_2d = np.reshape(ac_dist, (board_size, board_size))
            
            # Store all symmetries of each board state as well
            for i in range(4):
                current.env.rotate90()
                ac_dist_2d = np.rot90(ac_dist_2d)
                sym_ac_dist = np.ndarray.flatten(ac_dist_2d)
                self.buffer.append((
                    current.env.get_observation(separate=True),
                    sym_ac_dist,
                    winner * current.color
                ))

                current.env.flip()
                ac_dist_2d = np.flip(ac_dist_2d, axis=1)
                sym_ac_dist = np.ndarray.flatten(ac_dist_2d)
                self.buffer.append((
                    current.env.get_observation(separate=True),
                    sym_ac_dist,
                    winner * current.color
                ))
        
                current.env.flip()
                ac_dist_2d = np.flip(ac_dist_2d, axis=1)

            current = current.selected

        # Trim the buffer if it has exceeded the capacity
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
        

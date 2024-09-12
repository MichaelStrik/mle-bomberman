import os
import pickle
from collections import deque
from enum import Enum

# import random
import numpy as np

import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class Actions(Enum):
    """ Enum class. """
    UP  = 1
    RIGHT = 2
    LEFT = 3
    DOWN = 4
    WAIT = 5
    BOMB = 6


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    self.q_table = {}
    
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {}
    else:
        self.logger.info("Loading Q-table from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
    

    self.alpha = 1        # learning rate
    self.gamma = 0.9        # discount factor
    self.epsilon = 0.1
    self.epsilon_decay = 0.995  # ?
    self.epsilon_min = 0.1  # minimum exploration probability
    

def find_crates(game_state):
    """
    Findet alle Kisten auf dem Spielfeld und gibt deren Positionen zurück.

    :param game_state: Das aktuelle Spielzustand-Dictionary.
    :return: Eine Liste von Tupeln mit den Positionen der Kisten.
    """
    crates = []
    field = game_state['field']  # Das Spielfeld
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x, y] == 1:  # 1 steht normalerweise für eine Kiste
                crates.append((x, y))
    return crates


def get_valid_actions(game_state: dict) -> list:
    """
    """
    
    x, y = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    # crates = find_crates(game_state) 

    valid_actions = ['WAIT']

    directions = {
        'UP': (x, y - 1),
        'RIGHT': (x + 1, y),
        'DOWN': (x, y + 1),
        'LEFT': (x - 1, y)
    }

    # Verfügbarkeit der Richtungen überprüfen
    for action, (nx, ny) in directions.items():
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0 and not any(bomb[0] == (nx, ny) for bomb in bombs):  # Kein Hindernis und keine Bombe
                valid_actions.append(action)
    

    # dropping bomb possible?


    return valid_actions


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # preparation: get features, q_values and valid actions
    features = state_to_features(game_state)
    features = tuple(features)
    if features in self.q_table:
        q_values = self.q_table[features]
    else:
        q_values = np.zeros(len(ACTIONS))

    valid_actions = get_valid_actions(game_state)

    # choose action
    if self.train:
        r = np.random.rand()
    else:
        r = 1.0

    if r < self.epsilon:
        # exploration
        self.logger.info(f"RANDOM (r={r}:.2):")
        action = np.random.choice(valid_actions)
    else:
        # exploitation
        self.logger.info("Q-CHOICE:")
        valid_q_values = [q_values[ACTIONS.index(action)] for action in valid_actions]
        action = valid_actions[np.argmax(valid_q_values)]

    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # direct environment
    pos = game_state['self'][3]
    idx_x = np.arange(pos[0]-2, pos[0]+3)
    idx_y = np.arange(pos[1]-2, pos[1]+3)
    # prevent out of bounds access
    idx_x = np.fmax(idx_x, np.zeros_like(idx_x, dtype=int))
    # TODO which one is COLS, which one is ROWS?
    idx_x = np.fmin(idx_x, (s.COLS-1)*np.ones_like(idx_x, dtype=int))
    idx_y = np.fmax(idx_y, np.zeros(len(idx_y), dtype=int))
    idx_y = np.fmin(idx_y, (s.COLS-1)*np.ones_like(idx_y, dtype=int))
    # read from field
    env5x5_field = game_state['field'][idx_x, idx_y]
    env5x5_field = env5x5_field.flatten()
    env5x5_coins = [(coin[0]-pos[0],coin[1]-pos[1]) for coin in game_state['coins'] \
                    if coin[0]-pos[0]<=2 or coin[1]-pos[1]<=2]
    env5x5_coins = tuple(env5x5_coins)
    # env5x5_coins = np.array(env5x5_coins)[0]
    
    # nearest reachable coin
    node = bfs_coin(pos, game_state['field'], game_state['coins'])
    dir_enum = Actions(5) # to encode direction
    if node is not None and node.parent is not None:
        # store distance to coin
        dist = node.distance

        # recurse back to the first step taken
        while (node.parent.x, node.parent.y) != pos:
            node = node.parent
        coin_dir = (node.x-pos[0], node.y-pos[1])
        
        if coin_dir[0] == 1 and coin_dir[1] == 0:
            coin_step = dir_enum.RIGHT.value
        elif coin_dir[0] == -1 and coin_dir[1] == 0:
            coin_step = dir_enum.LEFT.value
        elif coin_dir[0] == 0 and coin_dir[1] == 1:
            coin_step = dir_enum.DOWN.value
        elif coin_dir[0] == 0 and coin_dir[1] == -1:
            coin_step = dir_enum.UP.value
        else:
            # should never happen
            raise AssertionError("The direction to BFS_COIN could not be determined.")
    else:
        coin_step = dir_enum.WAIT.value
        dist = np.inf   # set dist=inf so that no reward is given 
                        # for action WAIT (reward decays with dist)
    
    # that's it for now
    channels = [tuple(env5x5_field), tuple(env5x5_coins), (coin_step, dist)]
    
    return channels

    # concatenate them as a feature array
    stacked_channels = np.concatenate(channels)
    return stacked_channels


class Node:
    """
    Class to represent nodes in path-finding algorithms.
    Any Node has a coordinate, a parent that it precedes and a distance (the amount of steps
    that have been taken to get there).
    """
    def __init__(self, x, y, parent, distance=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.distance = distance


def bfs_coin(start, field, coins) -> Node | None :
    """
    Variant of bfs that searches for coins.
    If any coin can be reached, a Node object containing the coin's position (x, y)
    and other useful information.
    Returns None if no coin can be reached.

    :param start:   The position (x,y) from where to start the search with bfs.
    :param field:   The games's current field.
    :param coins:   The coordinates of all coins that currently lay on the field.
    """

    start = Node(start[0], start[1], parent=None, distance=0)
    queue = deque([start])
    visited = set()
    visited.add( (start.x, start.y) )

    while queue:
        v = queue.popleft()
        # TODO stop when coin found
        if (v.x, v.y) in coins:
            return v

        for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nx, ny = v.x + dx, v.y + dy
            if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                    and field[nx, ny] == 0 and (nx, ny) not in visited):
                queue.append( Node(nx, ny, parent=v, distance=v.distance + 1) )
                visited.add( (nx, ny) )
    
    return None

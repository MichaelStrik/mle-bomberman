import os
import pickle
import copy
from collections import deque
from enum import Enum

# import random
import numpy as np

import settings as s
from .agent_variables import VEC_TO_DIR


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class Actions(Enum):
    """ Enum class. """
    UP  = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5


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
    if game_state['self'][2]:
        valid_actions.append('BOMB')

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
        action = np.random.choice(valid_actions)
        self.logger.info(f"RANDOM (eps={self.epsilon:.2}, r={r:.2}): " + action)
    else:
        # exploitation
        # determine ALL valid actions with the highest q_value 
        # (could be several with the same maximal value)
        valid_q_values = [q_values[ACTIONS.index(action)] for action in valid_actions]
        q_val_max = np.max(valid_q_values)
        q_val_max_idx = np.argwhere(valid_q_values == q_val_max)

        valid_actions = np.array(valid_actions)
        q_actions = valid_actions[q_val_max_idx]

        # if there's not a unique maximum, make a random choice among them
        #FakeItTillYouMakeIt
        action = np.random.choice(q_actions.reshape(-1))
        self.logger.info("Q-CHOICE: " + action)


    return action


def get_env5x5_field(pos, field) -> tuple:
    # direct environment
    idx_x = np.arange(pos[0]-2, pos[0]+3)
    idx_y = np.arange(pos[1]-2, pos[1]+3)
    # prevent out of bounds access
    idx_x = np.fmax(idx_x, np.zeros_like(idx_x, dtype=int))
    # NOTE which one is COLS, which one is ROWS?--> doesn't matter as long as COLS=ROWS / field is a square
    idx_x = np.fmin(idx_x, (s.COLS-1)*np.ones_like(idx_x, dtype=int))
    idx_y = np.fmax(idx_y, np.zeros(len(idx_y), dtype=int))
    idx_y = np.fmin(idx_y, (s.COLS-1)*np.ones_like(idx_y, dtype=int))
    # read from field
    env5x5_field = field[idx_x]
    env5x5_field = env5x5_field[:, idx_y]
    env5x5_field = env5x5_field.flatten()

    return tuple(env5x5_field)


def name_to_action_enum(name):
    if name == 'UP':
        enum = Actions.UP.value
    elif name == 'RIGHT':
        enum = Actions.RIGHT.value
    elif name == 'DOWN':
        enum = Actions.DOWN.value
    elif name == 'LEFT':
        enum = Actions.LEFT.value
    elif name == 'WAIT':
        enum = Actions.WAIT.value
    elif name == 'BOMB':
        enum = Actions.BOMB.value
    else:
        ValueError("Enum class Action does not know '"+name+"'\n.")
    return enum


def state_to_features(game_state: dict) -> list:
    """
    Converts the game state to the input of your model, i.e., a feature vector.

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
    env5x5_field = get_env5x5_field(pos, game_state['field'])

    # relative positions of coins
    env5x5_coins = [(coin[0]-pos[0],coin[1]-pos[1]) for coin in game_state['coins'] \
                    if abs(coin[0]-pos[0])<=2 and abs(coin[1]-pos[1])<=2]
    env5x5_coins.sort()
    env5x5_coins = tuple(env5x5_coins)
    
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

    # dangerous steps (where there will be no escape from explosions)
    dangerous_actions = identify_dangerous_actions(pos, game_state['field'], game_state['bombs'], game_state['explosion_map'])
    
    # that's it for now
    channels = [env5x5_field, env5x5_coins, (coin_step, dist), dangerous_actions]
    
    return channels


class Node:
    """
    Class to represent nodes in path-finding algorithms.
    Any Node has a coordinate, a parent that precedes it and a distance (the amount of steps
    that have been taken to get there).
    """
    def __init__(self, pos, parent, distance=None):
        self.x = pos[0]
        self.y = pos[1]
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

    start = Node(start, parent=None, distance=0)
    queue = deque([start])
    visited = set()
    visited.add( (start.x, start.y) )

    while queue:
        v = queue.popleft()
        if (v.x, v.y) in coins:
            return v

        for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nx, ny = v.x + dx, v.y + dy
            if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                    and field[nx, ny] == 0 and (nx, ny) not in visited):
                queue.append( Node((nx, ny), parent=v, distance=v.distance + 1) )
                visited.add( (nx, ny) )
    
    return None


def simulate_explosion_map(explosion_map, bombs, k):
    """
    Simulates how explosion_map would look like in 'k' steps taking into account
    the current explosion_map and the current placed bombs.

    NOTE: the counters in the simulated explosion_map can not guaranteed to be correct
    as a 2 may be overwritten by a 1.
    NOTE: there is never a 2 in an explosion_map ;(
    """
    if k ==0:
        return explosion_map
    elif k >= 1:
        explosion_map_simulated = np.zeros_like(explosion_map)
    else:
        ValueError("Not a valid k.")    

    for bomb in bombs:
        timer = bomb[1]
        if timer-k == -1:
            # bomb exploded in last step, there is an explosion and it will continue in the next step
             x,y = bomb[0]
             explosion_map_simulated[x,max([y-3,0]):min([y+3+1,s.ROWS])] = 1
             explosion_map_simulated[max([x-3,0]):min([x+3+1,s.COLS]),y] = 1

    return explosion_map_simulated


def simulate_bombs(bombs, k):
    """
    Same as simulate_explosion_map, just for bombs.
    """
    bombs_simulated = []
    for bomb in bombs:
        bomb_sim = (bomb[0], max([-1, bomb[1]-k])) # reduce bomb timer by k
        if bomb_sim[1] >= 0:
            bombs_simulated.append(bomb_sim)

    return bombs_simulated


def simulate_explosions(field, explosion_map, bombs, k):
    """
    Returns an array with values 0 and 1 predicting whether there will be an explosion on the game's 'field' in 'k' steps.
    The prediction is based on 'explosion_map' and 'bombs' which are assumed to be the explosion_map and bombs in the current step.
    """
    explosion_map_k_minus_one = simulate_explosion_map(explosion_map, bombs, k-1) # explosion_map at k-1 steps in the future
    bombs_k_minus_one = simulate_bombs(bombs, k-1) # bombs at k-1 steps in the future

    # build an array which has value 1 where CURRENTLY (or rather k steps in the future) there is an explosion.
    # that's completely different that what explosion_map_k represents
    # note: an explosion will occur where explosion_map_k_minus_one equals 1 
    # and on the fields and surroundings where a bomb is about about to explode k-1 steps in the future
    explosions = explosion_map_k_minus_one
    for bomb in bombs_k_minus_one:
        if bomb[1] == 0:
            x, y = bomb[0]
            explosions[(x,y)] = 1
            # write surrounding explosion fields, taking into account stone fields
            if field[(x+1,y)] != -1:
                explosions[x:x+3+1,y] = 1
            if field[(x-1,y)] != -1:
                explosions[x-3:x+1,y] = 1
            if field[(x,y+1)] != -1:
                explosions[x,y:y+3+1] = 1
            if field[(x,y-1)] != -1:
                explosions[x,y-3:y+1] = 1

    return explosions


def is_safe(new_pos, field, explosion_map, bombs, steps, last_pos=None):
    """
    Returns True if pos is safe (no explosion at pos) in 'steps' steps, and False otherwise, taking into account the current explosion_maps
    and bombs lying on the field.
    Also returns False if a field is blocked by a bomb, i.e., the position can't be walked on.
    """
    # predict explosions 'steps' steps in the future
    explosions_steps = simulate_explosions(field, explosion_map, bombs, steps)
    no_explosion = not explosions_steps[new_pos] # will there be an explosion at 'new_pos'?
    # bombs
    bombs_simulated = simulate_bombs(bombs, steps)
    if last_pos is None:
        no_bomb = (not any(new_pos == bomb[0] for bomb in bombs_simulated))
    else:
        no_bomb = (not any(new_pos == bomb[0] for bomb in bombs_simulated if not bomb[0] == last_pos))

    return (no_explosion and no_bomb)


def dfs_escape_danger(start, field, bombs, explosion_map):
    """
    Variant of dfs that searches for ways out of dangerous fields.

    :param start:   A start Node object from where to start the search with dfs.
                    The member variable start.distance must be set to the step counter to start with.
                    start.distance = 0 means we start at the current game step.
    :param field:   The games's current field.
    :param bombs:   The bombs that currently lie on the field, including their respective timers.
    :param explosion_map: The current explosion_map.
    """

    stack = [start]

    # was start an explosion-free position in the first place?
    if any((start.x, start.y) == bomb[0] for bomb in simulate_bombs(bombs, start.distance)):
        last_pos = (start.x, start.y)
    else:
        last_pos = None

    if not is_safe((start.x, start.y), field, explosion_map, bombs, start.distance, last_pos=last_pos):
        return None

    while stack:
        v = stack.pop()
        steps = v.distance # we misuse the distance for a game step counter here

        if steps>=4:
            # after 4 steps, no new explosion fields add and we have found a safe way
            return v

        for dx, dy in [ (0, 0), (-1, 0), (1, 0), (0, 1), (0, -1) ]:
            nx, ny = v.x + dx, v.y + dy
            # if position is feasible, push to stack
            if (    0 <= nx < field.shape[0] 
                    and 0 <= ny < field.shape[1]
                    and field[nx, ny] == 0
                    and is_safe((nx, ny), field, explosion_map, bombs, steps, last_pos=(v.x, v.y))
                    ):
                stack.append( Node((nx, ny), parent=v, distance=steps+1) )

    return None


def bomb_is_safe(pos, field, bombs, explosion_map):
    """
    Checks if the action 'BOMB' would be safe at this position in the current game state.
    It does so by searching for escapes with dfs_escape_danger(...).
    """
    bombs_new = copy.copy(bombs)
    bombs_new.append((pos, 2))
    start = Node(pos, parent=None, distance=1)
    
    return bool(dfs_escape_danger(start, field, bombs_new, explosion_map))


def identify_dangerous_actions(pos, field, bombs, explosion_map):
    # identify dangerous steps
    steps_dangerous = []
    for step in [(0,0), (-1,0), (0,-1), (1,0), (0,1)]:
        start_pos = np.array(pos)+np.array(step)
        start_pos = tuple(start_pos)
        if not (0 <= start_pos[0] < field.shape[0] 
                and 0 <= start_pos[1] < field.shape[1]
                and field[start_pos] == 0
                and not any(start_pos == bomb[0] for bomb in bombs if bomb[0]==pos)):
            continue
        # TODO check if we can take that step in the first place
        start = Node(start_pos, parent=None, distance=1)
        if dfs_escape_danger(start, field, bombs, explosion_map) is None:
            steps_dangerous.append(step)

    bomb_safe = bomb_is_safe(pos, field, bombs, explosion_map)
    
    # encode information
    steps_dangerous_encoded = []
    for step in steps_dangerous:
        name = VEC_TO_DIR[step]
        action_enum = name_to_action_enum(name)
        steps_dangerous_encoded.append(action_enum)
    if not bomb_safe:
        steps_dangerous_encoded.append(name_to_action_enum('BOMB'))

    steps_dangerous_encoded = list(set(steps_dangerous_encoded)) # safety first: make it unique if it's not
    steps_dangerous_encoded.sort()

    return tuple(steps_dangerous_encoded)

import os
import pickle
import copy
from collections import deque
from enum import Enum

# import random
import numpy as np

import settings as s
from .agent_variables import VEC_TO_DIR

from random import shuffle


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

    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    

def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

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


def get_valid_actions(pos, can_drop_bomb, field, bombs) -> list:
    """
    """
    
    x, y = pos

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
    if can_drop_bomb:
        valid_actions.append('BOMB')

    return valid_actions


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

    # nearest reachable field next to crate
    # nearest reachable coin
    node = bfs_crate(pos, game_state['field'])
    dir_enum = Actions(5) # to encode direction
    if node is not None and node.parent is not None:
        # store distance to coin
        crate_dist = node.distance

        # recurse back to the first step taken
        while (node.parent.x, node.parent.y) != pos:
            node = node.parent
        crate_dir = (node.x-pos[0], node.y-pos[1])
        
        crate_dir_name = VEC_TO_DIR[crate_dir]
        crate_dir_id = name_to_action_enum(crate_dir_name)
        
    else:
        crate_dir_id = name_to_action_enum('WAIT')
        crate_dist = np.inf   # set dist=inf so that no reward is given 
                        # for action WAIT (reward decays with dist)
    

    # bombs around agent
    crate_counter_5x5 = 0
    for tile in env5x5_field:
        if tile == 1:
            crate_counter_5x5 += 1


    # dangerous steps (where there will be no escape from explosions)
    can_drop_bomb = game_state['self'][2]
    dangerous_actions = identify_dangerous_actions(pos, can_drop_bomb, game_state['field'], game_state['bombs'], game_state['explosion_map'])
    
    # that's it for now
    channels = [env5x5_field, env5x5_coins, (coin_step, dist), dangerous_actions, (crate_dir_id, crate_dist, crate_counter_5x5)]
    
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


def bfs_crate(start, field) -> Node | None :
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
        
        for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nx, ny = v.x + dx, v.y + dy
            if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                    and field[nx, ny] == 0 and (nx, ny) not in visited):
                if field[(nx,ny)] == 1:
                    return v
                queue.append( Node((nx, ny), parent=v, distance=v.distance + 1) )
                visited.add( (nx, ny) )
    
    return None


def simulate_explosion_map(field, explosion_map, bombs, k):
    """
    Simulates how explosion_map would look like in 'k' steps taking into account
    the current explosion_map and the current placed bombs.

    NOTE: simulate_explosion_map
    NOTE: there is never a 2 in an explosion_map ;(
    """
    if k ==0:
        return copy.copy(explosion_map)
    elif k >= 1:
        explosion_map_simulated = np.zeros_like(explosion_map)
    else:
        ValueError("Not a valid k.")    

    for bomb in bombs:
        timer = bomb[1]
        if timer-k == -1:
            # bomb exploded in last step, there is an explosion and it will continue in the next step
            x,y = bomb[0]
            if field[(x+1,y)] != -1:
                explosion_map_simulated[x:min(x+3+1,s.COLS),y] = 1
            if field[(x-1,y)] != -1:
                explosion_map_simulated[max(x-3,0):x+1,y] = 1
            if field[(x,y+1)] != -1:
                explosion_map_simulated[x,y:min(y+3+1,s.COLS)] = 1
            if field[(x,y-1)] != -1:
                explosion_map_simulated[x,max(y-3,0):y+1] = 1
            
    return explosion_map_simulated


def simulate_bombs(bombs, k):
    """
    Same as simulate_explosion_map, just for bombs.
    Bombs are taken out when their timer would fall below -2 after the 'k' steps.
    """
    bombs_simulated = []
    for bomb in bombs:
        bomb_sim = (bomb[0], max([-2, bomb[1]-k])) # reduce bomb timer by k
        if bomb_sim[1] >= -2:
            bombs_simulated.append(bomb_sim)

    return bombs_simulated


def simulate_explosions(field, explosion_map, bombs, k):
    """
    Returns an array with values 0 and 1 predicting whether there will be an explosion on the game's 'field' in 'k' steps.
    The prediction is based on 'explosion_map' and 'bombs' which are assumed to be the explosion_map and bombs in the current step.
    """
    explosion_map_k_minus_one = simulate_explosion_map(field, explosion_map, bombs, k-1) # explosion_map at k-1 steps in the future
    bombs_k_minus_one = simulate_bombs(bombs, k-1) # bombs at k-1 steps in the future

    # build an array which has value 1 where CURRENTLY (or rather k steps in the future) there is an explosion.
    # that's completely different that what explosion_map_k represents
    # note: an explosion will occur where explosion_map_k_minus_one equals 1 
    # and on the fields and surroundings where a bomb is about about to explode k-1 steps in the future
    explosions = explosion_map_k_minus_one
    for bomb in bombs_k_minus_one:
        if bomb[1] == 0 or bomb[1] == -1:
            x, y = bomb[0]
            explosions[(x,y)] = 1
            # write surrounding explosion fields, taking into account stone fields
            if field[(x+1,y)] != -1:
                explosions[x:min(x+3+1, s.COLS),y] = 1
            if field[(x-1,y)] != -1:
                explosions[max(x-3,0):x+1,y] = 1
            if field[(x,y+1)] != -1:
                explosions[x,y:min(y+3+1, s.COLS)] = 1
            if field[(x,y-1)] != -1:
                explosions[x,max(y-3,0):y+1] = 1

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

    # do we start on a bomb?
    if any((start.x, start.y) == bomb[0] for bomb in simulate_bombs(bombs, start.distance)):
        last_pos = (start.x, start.y)
    else:
        last_pos = None

    # was start an explosion-free position in the first place?
    if not is_safe((start.x, start.y), field, explosion_map, bombs, start.distance, last_pos=last_pos):
        return None

    while stack:
        v = stack.pop()
        steps = v.distance # we misuse the distance for a game step counter here

        if steps>=5:
            # after 4 steps, no new explosion fields add and we have found a safe way
            return v

        for dx, dy in [ (0, 0), (-1, 0), (1, 0), (0, 1), (0, -1) ]:
            nx, ny = v.x + dx, v.y + dy
            # if position is feasible, push to stack
            if (    0 <= nx < field.shape[0] 
                    and 0 <= ny < field.shape[1]
                    and field[nx, ny] == 0
                    # is position safe and not blocked by a bomb? (exception: we already stand on that bomb)
                    and is_safe((nx, ny), field, explosion_map, bombs, steps+1, last_pos=(v.x, v.y))
                    ):
                stack.append( Node((nx, ny), parent=v, distance=steps+1) )

    return None


def bomb_is_safe(pos, field, bombs, explosion_map):
    """
    Checks if the action 'BOMB' would be safe at this position in the current game state.
    It does so by searching for escapes with dfs_escape_danger(...).
    """
    bombs_new = copy.copy(bombs)
    bombs_new.append((pos, 4))
    start = Node(pos, parent=None, distance=1)
    
    return bool(dfs_escape_danger(start, field, bombs_new, explosion_map))


def identify_dangerous_actions(pos, can_drop_bomb, field, bombs, explosion_map):
    # identify dangerous steps
    steps_dangerous = []
    for step in [(0,0), (-1,0), (0,-1), (1,0), (0,1)]:
        start_pos = np.array(pos)+np.array(step)
        start_pos = tuple(start_pos)
        if not (0 <= start_pos[0] < field.shape[0] 
                and 0 <= start_pos[1] < field.shape[1]
                and field[start_pos] == 0
                and not any(start_pos == bomb[0] for bomb in bombs if not bomb[0]==pos)):
            continue
        start = Node(start_pos, parent=None, distance=1)
        if dfs_escape_danger(start, field, bombs, explosion_map) is None:
            steps_dangerous.append(step)

    bomb_safe = bomb_is_safe(pos, field, bombs, explosion_map)
    
    # encode information
    valid_actions = get_valid_actions(pos, can_drop_bomb, field, bombs)
    steps_dangerous_encoded = []
    for step in steps_dangerous:
        name = VEC_TO_DIR[step]
        if name in valid_actions:
            action_enum = name_to_action_enum(name)
            steps_dangerous_encoded.append(action_enum)
    if not bomb_safe:
        steps_dangerous_encoded.append(name_to_action_enum('BOMB'))

    steps_dangerous_encoded = list(set(steps_dangerous_encoded)) # safety first: make it unique if it's not
    steps_dangerous_encoded.sort()

    return tuple(steps_dangerous_encoded)

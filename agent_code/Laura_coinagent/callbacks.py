import os
import pickle
import random
from collections import deque

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


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
    #if self.train or not os.path.isfile("my-saved-model.pt"):
        #self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
    #else:
        #self.logger.info("Loading model from saved state.")
        #with open("my-saved-model.pt", "rb") as file:
            #self.model = pickle.load(file)
    
    # Prüfen, ob eine gespeicherte Q-Tabelle vorhanden ist
    if os.path.isfile("my-qlearning-model.pt"):
        with open("my-qlearning-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded Q-table from my-qlearning-model.pt")
    else:
        # Falls keine gespeicherte Q-Tabelle vorhanden ist, initialisieren wir sie leer.
        self.q_table = {}
        self.logger.info("No saved model found. Starting with an empty Q-table.")
    
    self.alpha = 0.5
    self.gamma = 0.9
    self.epsilon = 0.5
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.1
    self.coordinate_history = deque([], 5)

    #self.bomb_history = deque([], 5)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    #random_prob = .1
    #if self.train and random.random() < random_prob:
        #self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0])

    #self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)
    features = state_to_features(game_state)
    state = tuple(features)

    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    
    # Alle Aktionen überprüfen
    q_values = self.q_table[state]
    valid_actions=check_valid_action(self, game_state)
    #valid_actions = []
    
    #for action in ACTIONS:
        #if action == 'BOMB':
            #if is_safe_position(game_state, game_state['self'][3], get_bomb_timers(game_state)):
                #valid_actions.append(action)
        #else:
            #valid_actions.append(action)
    
    if np.random.rand() < self.epsilon:
        action = np.random.choice(valid_actions)
        self.logger.info("random action")
    else:
        self.logger.debug("Querying Q-table for action.")
        valid_q_values = [q_values[ACTIONS.index(action)] for action in valid_actions]
        action = valid_actions[np.argmax(valid_q_values)]
        #action = np.argmax(q_values)
    self.coordinate_history.append(game_state['self'][3])
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
    coin_index=np.argmin(np.abs(np.array(game_state['coins'])[:,0]-np.array(game_state['self'][3][0]))+np.abs(np.array(game_state['coins'])[:,1]-np.array(game_state['self'][3][1])))
    nearest_coin=game_state['coins'][coin_index]
    
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)#nearest_coin)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)
    return np.concatenate([nearest_coin])

def check_valid_action(self, game_state):
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
     # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys) and
                (self.coordinate_history.count(d) < 2)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    #if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.info(f'Valid actions: {valid_actions}')
    return valid_actions


        
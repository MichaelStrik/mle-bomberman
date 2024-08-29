import os
import pickle
import numpy as np
from collections import deque
#import numpy as np
import random
from collections import deque
import tensorflow as tf
import keras as kr
#from tensorflow.keras import layers

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


# Q-Network
def build_model(state_size, action_size):
    model = tf.keras.Sequential()
    model.add(kr.layers.Dense(24, input_dim=state_size, activation='relu'))
    model.add(kr.layers.Dense(24, activation='relu'))
    model.add(kr.layers.Dense(action_size, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=0.5))
    return model

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
    self.logger.info("Setting up agent from saved state.")
    
    # Prüfen, ob eine gespeicherte Q-Tabelle vorhanden ist
    #if os.path.isfile("my-sarsa-model.pt"):
        #with open("my-sarsa-model.pt", "rb") as file:
            #self.q_table = pickle.load(file)
        #self.logger.info("Loaded Q-table from my-sarsa-model.pt")
    #else:
        # Falls keine gespeicherte Q-Tabelle vorhanden ist, initialisieren wir sie leer.
        #self.q_table = {}
        #self.logger.info("No saved model found. Starting with an empty Q-table.")
    
    self.alpha = 0.5
    self.gamma = 0.9
    self.epsilon = 0.5
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.1

    self.state_size=7 #number of features we derive from gamestate
    self.memory_size=1000000

     # Prüfen, ob eine gespeicherte Q-Tabelle vorhanden istcustom_objects={'mae': 'mae'
    if os.path.isfile('q_network_model.h5'):
        #with open('q_network_model.h5', "rb") as file:
        self.target_model=tf.keras.models.load_model('Target_network_model.h5',custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        self.model = kr.models.load_model('q_network_model.h5',custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        self.logger.info("Loaded models")
    else:
        # Falls keine gespeicherte Q-Tabelle vorhanden ist, initialisieren wir sie leer.
        self.model = build_model(self.state_size, len(ACTIONS))
        self.target_model = build_model(self.state_size, len(ACTIONS))
        self.target_model.set_weights(self.model.get_weights())
        self.logger.info("No saved model found.")

    # Replay Memory
    self.memory = deque(maxlen=self.memory_size)

def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    state = tuple(features)
    state = np.reshape(state, [1, self.state_size])
    
    #if state not in self.q_table:
     #   self.q_table[state] = np.zeros(len(ACTIONS))
    
    valid_actions = get_valid_actions(game_state)

    if np.random.rand() < self.epsilon:
        # Epsilon-greedy: Zufällige Aktion auswählen, um Exploration zu fördern
        action = np.random.choice(valid_actions)
    else:
        #self.logger.debug("Querying Q-table for action.")
        #q_values = self.q_table[state]
        #valid_q_values = [q_values[ACTIONS.index(action)] for action in valid_actions]
        #action = valid_actions[np.argmax(valid_q_values)]
        q_values = self.model.predict(state)[0]
        valid_q_values = [q_values[ACTIONS.index(action)] for action in valid_actions]
        action = valid_actions[np.argmax(valid_q_values)]
    return action

def get_valid_actions(game_state):
    x, y = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    crates = find_crates(game_state) 

    valid_actions = []

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

    if len(valid_actions) == 0:
        valid_actions.append('WAIT')
    
    if 'BOMB' not in valid_actions and any(field[nx, ny] == 1 for nx, ny in directions.values()) and is_safe_to_place_bomb(game_state):
        valid_actions.append('BOMB')

    # Hier entfernen wir die Bombe als Option, wenn es keine sicheren Kisten gibt
    if 'BOMB' in valid_actions:
        safe_crates = []
        for crate in crates:
            if is_safe_position(crate, game_state):  # is_safe_position anstelle von is_safe_to_place_bomb
                safe_crates.append(crate)
        if len(safe_crates) == 0:
            valid_actions.remove('BOMB')

    return valid_actions


def is_safe_to_place_bomb(game_state, position=None):
    """Überprüft, ob es sicher ist, eine Bombe zu platzieren."""
    if position is None:
        position = game_state['self'][3]  # Aktuelle Position des Agenten
    x, y = position
    field = game_state['field']

    # Prüfen, ob der Agent nach dem Platzieren der Bombe einen sicheren Fluchtweg hat
    for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0 and is_safe_position((nx, ny), game_state):
                return True
    return False


def is_safe_position(position, game_state):
    """Überprüft, ob eine Position sicher vor einer Explosion ist."""
    bombs = game_state['bombs']
    field = game_state['field']
    x, y = position

    for bomb_pos, _ in bombs:
        bx, by = bomb_pos
        if (bx == x and abs(by - y) <= 3) or (by == y and abs(bx - x) <= 3):
            return False
    return True

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    # Feature 1: Position of the agent
    x, y = game_state['self'][3]

    # Feature 2: Distance to the nearest coin
    coins = game_state['coins']
    field = game_state['field']
    if coins:
        distances = [bfs(field, (x, y), coin) for coin in coins]
        min_distance = min(distances)
        closest_coin = coins[distances.index(min_distance)]
        coin_dx = closest_coin[0] - x
        coin_dy = closest_coin[1] - y
        coin_feature = np.array([coin_dx, coin_dy, min_distance])
    else:
        coin_feature = np.array([0, 0, float('inf')])  # Kein Münze in Sicht

    # Feature 3: Distance to the nearest crate (box)
    crates = np.argwhere(field == 1)
    if crates.any():
        crate_distances = [bfs(field, (x, y), tuple(crate)) for crate in crates]
        min_crate_distance = min(crate_distances)
        closest_crate = crates[crate_distances.index(min_crate_distance)]
        crate_dx = closest_crate[0] - x
        crate_dy = closest_crate[1] - y
        crate_feature = np.array([crate_dx, crate_dy, min_crate_distance])
    else:
        crate_feature = np.array([0, 0, float('inf')])  # Keine Kisten in Sicht

    # Feature 4: Distance to the nearest bomb
    bomb_distances = [np.linalg.norm(np.array([x, y]) - np.array(bomb[0])) for bomb in game_state['bombs']]
    bomb_feature = np.array([min(bomb_distances)]) if bomb_distances else np.array([float('inf')])

    # Combine all features into a single array
    return np.concatenate([coin_feature, crate_feature, bomb_feature])

def bfs(field, start, target):
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)

    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == target:
            return dist

        for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                    and field[nx, ny] == 0 and (nx, ny) not in visited):
                queue.append(((nx, ny), dist + 1))
                visited.add((nx, ny))
    
    return float('inf')

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
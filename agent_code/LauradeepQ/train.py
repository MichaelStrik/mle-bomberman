import numpy as np
import random
from collections import deque
import tensorflow as tf
#from tensorflow.keras import layers
import keras as kr

import pickle
import os
import logging
#import numpy as np
from .callbacks import state_to_features
import events as e

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#logging.getLogger('tensorflow').setLevel(logging.ERROR)
#tf.autograph.set_verbosity(3)


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Q-Network
def build_model(state_size, action_size):
    model = tf.keras.Sequential()
    model.add(kr.layers.Dense(24, input_dim=state_size, activation='relu'))
    model.add(kr.layers.Dense(24, activation='relu'))
    model.add(kr.layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.5))
    return model


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    self.logger.info("Setting up training variables.")
    # Q-table initialisieren
    #self.q_table = {}
    self.alpha = 0.3  # Lernrate davor 0.1
    self.gamma = 0.95  # Discountfaktor um so höher um so stätrker werden zufällige gewinne belohnt
    self.epsilon = 1.0  # Epsilon für epsilon-greedy 
    self.epsilon_decay = 0.995  # Epsilon-Decay für Exploration-Exploitation Tradeoff
    self.epsilon_min = 0.1

    self.last_positions = []
    self.last_actions = []
#warum muss man hier nochmal alles setup, wird das nicht aus setup aus callback übernommen?
    self.state_size=7 #number of features we derive from gamestate
    self.memory_size=1000000
    self.batch_size = 64

    # Prüfen, ob eine gespeicherte Q-Tabelle vorhanden ist
    if os.path.isfile('q_network_model.h5'):
        #with open('q_network_model.h5', "rb") as file:
        self.model = tf.keras.models.load_model('q_network_model.h5',custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        self.target_model=tf.keras.models.load_model('target_network_model.h5',custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.5))
        self.logger.info("Loaded models")
    else:
        # Falls keine gespeicherte Q-Tabelle vorhanden ist, initialisieren wir sie leer.
        self.model = build_model(self.state_size, len(ACTIONS))
        self.target_model = build_model(self.state_size, len(ACTIONS))
        self.target_model.set_weights(self.model.get_weights())
        self.logger.info("No saved model found.")

    # Replay Memory
    self.memory = deque(maxlen=self.memory_size)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Events: {", ".join(map(str, events))}')
    
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    old_state = tuple(old_state)
    new_state = tuple(new_state)

    valid_actions = get_valid_actions(new_game_state)

    old_state = np.reshape(old_state, [1, self.state_size])  #ich hoffe game_state wird passierend auf action geupdatet und nicht mit new_state 
    new_state = np.reshape(new_state, [1, self.state_size]) 

   # if old_state not in self.q_table:
    #    self.q_table[old_state] = np.zeros(len(ACTIONS))
    
   # if new_state not in self.q_table:
    #    self.q_table[new_state] = np.zeros(len(ACTIONS))

    action_idx = ACTIONS.index(self_action)
    
    # Belohnung für das Legen einer Bombe in der Nähe von Kisten
    if self_action == 'BOMB' and any([np.linalg.norm(np.array(old_game_state['self'][3]) - np.array(box)) <= 1 for box in np.argwhere(old_game_state['field'] == 1)]) and is_safe_to_place_bomb(old_game_state, old_game_state['self'][3]):
        events.append("BOMB_PLACED_NEAR_BOXES")

    # Belohnung für keinen selbstmord
    if self_action == 'BOMB' and is_safe_to_place_bomb(old_game_state, old_game_state['self'][3]):
        events.append("SAFE_BOMB_PLACEMENT")
    
    # Bewegung erkennen und mögliche Schleifen vermeiden
    current_position = new_game_state['self'][3]
    if len(self.last_positions) >= 3 and current_position in self.last_positions[-2:]:
        events.append("STUCK_IN_LOOP")

    # Position und Aktion speichern
    self.last_positions.append(current_position)
    self.last_actions.append(self_action)

    # Maximale Länge des Verlaufs begrenzen
    if len(self.last_positions) > 4:
        self.last_positions.pop(0)
        self.last_actions.pop(0)

    # Reward, jetzt mit alten und neuen Zuständen
    reward = reward_from_events(self, events, old_game_state, new_game_state)

    # SARSA Update-Regel
   # if new_game_state is not None:
    #    next_action = np.argmax(self.q_table[new_state])
     #   self.q_table[old_state][action_idx] = self.q_table[old_state][action_idx] + \
      #                                        self.alpha * (reward + self.gamma * self.q_table[new_state][next_action] - self.q_table[old_state][action_idx])
    #else:
     #   self.q_table[old_state][action_idx] = self.q_table[old_state][action_idx] + \
      #self.alpha * (reward - self.q_table[old_state][action_idx])

#Muss im Training auch epsilon greedy angewendet werden?
    if np.random.rand() <= self.epsilon:
        #next_action = random.randrange(len(ACTIONS))    #hier auch valid_actions?
        next_action = np.random.choice(valid_actions)
    else:
        q_values = self.model.predict(new_state)[0]
        valid_q_values = [q_values[ACTIONS.index(next_action)] for next_action in valid_actions]
        next_action = valid_actions[np.argmax(valid_q_values)]
        #next_action = np.argmax(q_values)
        
        self.memory.append((old_state, next_action, reward, new_state)) #done??
        
        if len(self.memory) > self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)
            for old_state, next_action, reward, new_state in minibatch:
                target = reward + self.gamma * np.amax(self.target_model.predict(new_state)[0])
                target_f = self.model.predict(old_state)   #q value is updated here
                target_f[0][ACTIONS.index(next_action)] = target
                self.model.fit(old_state, target_f, epochs=1, verbose=0)  #q network is trained here, modelfit performs gradient descent
            
    # Update Target Network
    if new_game_state['step'] % 10 == 0:
        self.target_model.set_weights(self.model.get_weights())

    # Epsilon-Decay
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

def get_valid_actions(new_game_state):
    x, y = new_game_state['self'][3]
    field = new_game_state['field']
    bombs = new_game_state['bombs']
    crates = find_crates(new_game_state) 

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
    
    if 'BOMB' not in valid_actions and any(field[nx, ny] == 1 for nx, ny in directions.values()) and is_safe_to_place_bomb(new_game_state):
        valid_actions.append('BOMB')

    # Hier entfernen wir die Bombe als Option, wenn es keine sicheren Kisten gibt
    if 'BOMB' in valid_actions:
        safe_crates = []
        for crate in crates:
            if is_safe_position(crate, new_game_state):  # is_safe_position anstelle von is_safe_to_place_bomb
                safe_crates.append(crate)
        if len(safe_crates) == 0:
            valid_actions.remove('BOMB')

    return valid_actions
    
def find_crates(new_game_state):
    """
    Findet alle Kisten auf dem Spielfeld und gibt deren Positionen zurück.

    :param game_state: Das aktuelle Spielzustand-Dictionary.
    :return: Eine Liste von Tupeln mit den Positionen der Kisten.
    """
    crates = []
    field = new_game_state['field']  # Das Spielfeld
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x, y] == 1:  # 1 steht normalerweise für eine Kiste
                crates.append((x, y))
    return crates
    
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
def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Events at end of round: {", ".join(map(str, events))}')

    last_state = state_to_features(last_game_state)
    last_state = tuple(last_state)
    
    #if last_state not in self.q_table:
     #   self.q_table[last_state] = np.zeros(len(ACTIONS))

    action_idx = ACTIONS.index(last_action)
    
    # Reward mit den letzten Zuständen
    reward = reward_from_events(self, events, last_game_state, None)

    # End of round Q-Update
    #self.q_table[last_state][action_idx] = self.q_table[last_state][action_idx] + \
                                           #self.alpha * (reward - self.q_table[last_state][action_idx])

    # Modell speichern
   # model_file = "my-sarsa-model.pt"
    #with open(model_file, "wb") as file:
    #    pickle.dump(self.q_table, file)
    #self.logger.info(f"Q-table saved to {model_file}.")

     # Memory speichern?????
    #model_file = "my-memory.pt"
    #with open(model_file, "wb") as file:
    #    pickle.dump(self.memory, file)
    #self.logger.info(f"Memory saved to {model_file}.")

    #trainiertes Q-network speichern
    self.model.save('q_network_model.h5')

    #trainiertes Target-network speichern
    self.target_model.save('Target_network_model.h5')

def reward_from_events(self, events: list[str], old_game_state: dict, new_game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 10,
        e.MOVED_UP: -0.05,
        e.MOVED_DOWN: -0.05,
        e.MOVED_LEFT: -0.05,
        e.MOVED_RIGHT: -0.05,
        e.WAITED: -0.1,
        e.INVALID_ACTION: -1,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -50,
        "STUCK_IN_LOOP": -5,
        "BOMB_PLACED_NEAR_BOXES": 2,
        "BOX_DESTROYED": 4,
        "ESCAPED_BOMB": 3,
        "SAFE_BOMB_PLACEMENT": 5,
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])

    # Annäherung an Münzen und Kisten belohnen
    if old_game_state is not None and new_game_state is not None:
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]

        # Belohnung für das Annähern an eine Kiste
        if new_game_state['field'][new_position] == 1:
            old_min_dist = min([np.linalg.norm(np.array(old_position) - np.array(box)) for box in np.argwhere(old_game_state['field'] == 1)])
            new_min_dist = min([np.linalg.norm(np.array(new_position) - np.array(box)) for box in np.argwhere(new_game_state['field'] == 1)])
            if new_min_dist < old_min_dist:
                reward_sum += 0.5

        # Flucht aus der Bombenreichweite belohnen
        old_bomb_distances = [np.linalg.norm(np.array(old_position) - np.array(bomb[0])) for bomb in old_game_state['bombs']]
        new_bomb_distances = [np.linalg.norm(np.array(new_position) - np.array(bomb[0])) for bomb in new_game_state['bombs']]

        if old_bomb_distances and new_bomb_distances:  # Überprüfen, ob beide Listen nicht leer sind
            if min(new_bomb_distances) > min(old_bomb_distances):
                reward_sum += 1  # Belohnung für das Entkommen aus der Bombenreichweite

    if "BOMB_PLACED_NEAR_BOXES" in events and "ESCAPED_BOMB" in events:
        reward_sum += game_rewards["SAFE_BOMB_PLACEMENT"]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def is_safe_to_place_bomb(game_state, position):
    """Check if it is safe to place a bomb at the given position."""
    bombs = game_state['bombs']
    x, y = position
    field = game_state['field']  # Hier 'field' anstatt 'arena' verwenden
    directions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    
    for d in directions:
        if field[d] == 0 and is_safe_after_bomb(d, bombs):  # Funktion zum Prüfen der Sicherheit
            return True
    return False



def is_safe_after_bomb(game_state, position):
    """Check if the agent is safe after placing a bomb at the given position."""
    return len(get_safe_positions_after_bomb(game_state, position)) > 0

def get_safe_positions_after_bomb(game_state, bomb_position):
    """Get all safe positions the agent can move to after placing a bomb."""
    if not bomb_position or len(bomb_position) != 2:
        return []  # Falls bomb_position ungültig ist, gebe eine leere Liste zurück

    x, y = bomb_position
    field = game_state['field']
    explosion_map = game_state['explosion_map']

    directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    safe_positions = []

    for nx, ny in directions:
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0 and explosion_map[nx, ny] == 0:
                safe_positions.append((nx, ny))

    return safe_positions
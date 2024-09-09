import os
import pickle
import numpy as np
import logging
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import events as e
def bfs(field, start, target):
    from collections import deque
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

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    # Example feature extraction process
    # Assume that you have walls, bombs, and player positions as before
    walls = game_state['field']
    bombs = game_state['bombs']
    player = game_state['self']

    # Convert walls to a numpy array
    walls_features = np.array(walls, dtype=np.int32)

    # Initialize bomb features with zeros (same shape as walls)
    bombs_features = np.zeros_like(walls_features)
    for bomb in bombs:
        bombs_features[bomb[0], bomb[1]] = 1  # Mark bomb positions

    # Initialize player features
    player_features = np.zeros_like(walls_features)
    player_x, player_y = player[1], player[2]
    player_features[player_x, player_y] = 1  # Mark player position

    # Stack the channels
    channels = [walls_features, bombs_features, player_features]
    stacked_channels = np.stack(channels, axis=0)

    # Flatten the stacked channels to match the shape expected by the Q-table
    flattened_state = stacked_channels.flatten()

    return flattened_state  # This should now match the expected shape




class DQNAgent:
    def __init__(self):
        # Logger einrichten
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Set up training variables
        self.setup_training()

    def setup_training(self):
        """
        Initialise self for training purpose.
        This is called after `setup` in callbacks.py.
        """
        self.logger.info("Setting up DQN training variables.")
        
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Neural network model for Q-learning
        self.model = self._build_model()

    def _build_model(self):
        """Builds a Neural Network model."""
        model = Sequential()
        model.add(Dense(24, input_dim=5, activation='relu'))  # 5 input features
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(ACTIONS), activation='linear'))  # Output layer for each action
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores the experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(ACTIONS)
        act_values = self.model.predict(state)
        return ACTIONS[np.argmax(act_values[0])]

    def replay(self):
        for state, action, reward, next_state, done in self.memory:
            # Überprüfen, ob next_state vorhanden ist
            if next_state is not None:
                next_state = np.expand_dims(next_state, axis=0)  # Eingabeform (1, 5)
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target = reward
            
            state = np.expand_dims(state, axis=0)  # Eingabeform (1, 5)
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load_model(self, name):
        """Loads a pre-trained model."""
        self.model.load_weights(name)

    def save_model(self, name):
        """Saves the current model."""
        self.model.save_weights(name)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    reward = reward_from_events(self, events, old_game_state, new_game_state)
    
    done = new_game_state is None  # Zum Beispiel wenn das Spiel vorbei ist
    self.agent.remember(old_state, self_action, reward, new_state, done)


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Called at the end of each game to hand out final rewards and do training.
    """
    last_state = state_to_features(last_game_state)
    reward = reward_from_events(self, events, last_game_state, None)
    
    done = True  # Da das Spiel zu Ende ist
    self.agent.remember(last_state, last_action, reward, None, done)  # Hier sollte self.agent auf eine DQNAgent-Instanz verweisen
    self.agent.replay()  # Beispiel für eine Replay-Methode


def reward_from_events(self, events, old_game_state, new_game_state):
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.WAITED: -0.2,
        e.INVALID_ACTION: -1,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -10,
        "STUCK_IN_LOOP": -8
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])

    if old_game_state and new_game_state:
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]

        if new_game_state['coins']:
            old_min_dist = min([np.linalg.norm(np.array(old_pos) - np.array(coin)) for coin in old_game_state['coins']])
            new_min_dist = min([np.linalg.norm(np.array(new_pos) - np.array(coin)) for coin in new_game_state['coins']])

            if new_min_dist < old_min_dist:
                reward_sum += 0.5

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def setup_training(self):
    """
    Setup training-specific configurations or variables.
    This function is required by the environment and should be present in your agent's code.
    """
    pass
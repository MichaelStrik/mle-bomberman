import os
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class DQN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu', input_shape=(input_dim,))
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(output_dim, activation=None)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

class DQNAgent:
    def __init__(self):
        self.setup()

    def setup(self):
        self.logger.info("Setting up agent with DQN model.")

        input_dim = 3  # Anzahl der Features
        output_dim = len(ACTIONS)

        # Model und Target-Model initialisieren
        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.optimizer = optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        if os.path.isfile("dqn_model.h5"):
            self.model.load_weights("dqn_model.h5")
            self.target_model.load_weights("dqn_model.h5")
            self.logger.info("Loaded DQN model from dqn_model.h5")
        else:
            self.logger.info("No saved model found. Starting with a new DQN model.")

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_steps = 1000
        self.step_counter = 0

    def act(self, game_state: dict) -> str:
        features = state_to_features(game_state)
        state = np.array([features])

        # Epsilon-Greedy Entscheidung
        if np.random.rand() < self.epsilon:
            action = np.random.choice(ACTIONS)
        else:
            q_values = self.model.predict(state)
            action = ACTIONS[np.argmax(q_values[0])]

        return action
        
    def state_to_features(game_state: dict) -> np.array:
        if game_state is None:
            return np.zeros(3)

        x, y = game_state['self'][3]
        coins = game_state['coins']
        field = game_state['field']

        if coins:
            distances = []
            for coin in coins:
                path_length = bfs(field, (x, y), coin)
                distances.append(path_length)

            min_distance = min(distances)
            closest_coin = coins[distances.index(min_distance)]
            dx = closest_coin[0] - x
            dy = closest_coin[1] - y
            result = np.array([dx, dy, min_distance])
        else:
            result = np.array([0, 0, 0])

        print(f"State features: {result}")  # Debug output
        return result


    def get_valid_actions(game_state):
        x, y = game_state['self'][3]
        field = game_state['field']
        valid_actions = []

        directions = {
            'UP': (x, y - 1),
            'RIGHT': (x + 1, y),
            'DOWN': (x, y + 1),
            'LEFT': (x - 1, y)
        }
        
        for action, (nx, ny) in directions.items():
            if field[nx, ny] == 0:  # Freies Feld
                valid_actions.append(action)
        
        if len(valid_actions) == 0:
            valid_actions.append('WAIT')
        
        return valid_actions

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

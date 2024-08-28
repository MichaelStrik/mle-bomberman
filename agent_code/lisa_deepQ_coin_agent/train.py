import os
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import events as e
from .callbacks import DQNAgent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

    
class DQNAgentTraining:
    def __init__(self):
        self.setup_training()

    def setup_training(self):
        self.logger.info("Setting up training variables for DQN.")
        self.agent = DQNAgent()
        input_dim = 3
        output_dim = len(ACTIONS)

        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.optimizer = optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        model_weights_path = "dqn_model.weights.h5"
        if os.path.isfile(model_weights_path):
            self.model.load_weights(model_weights_path)
            self.target_model.load_weights(model_weights_path)
            self.logger.info(f"Loaded DQN model from {model_weights_path}")
        else:
            self.logger.info(f"No saved model found. Starting with a new DQN model.")

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_steps = 1000
        self.step_counter = 0

    
    def replay(self):
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*[self.memory[idx] for idx in batch])
        
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        
        next_q_values = self.target_model.predict(next_state_batch)
        target_q = reward_batch + self.gamma * np.max(next_q_values, axis=1)
        
        masks = tf.one_hot(action_batch, len(ACTIONS))
        with tf.GradientTape() as tape:
            q_values = self.model(state_batch)
            q_action = tf.reduce_sum(q_values * masks, axis=1)
            loss = self.loss_fn(target_q, q_action)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))



    def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
        old_state = state_to_features(old_game_state)
        new_state = state_to_features(new_game_state)
        action_idx = ACTIONS.index(self_action)
        reward = reward_from_events(self, events, old_game_state, new_game_state)
        self.memory.append((old_state, action_idx, reward, new_state))
        if len(self.memory) > self.batch_size:
            self.replay()
        self.step_counter += 1
        if self.step_counter % self.update_target_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def end_of_round(self, last_game_state, last_action, events):
        last_state = state_to_features(last_game_state)
        action_idx = ACTIONS.index(last_action)
        reward = reward_from_events(self, events, last_game_state, None)
        self.memory.append((last_state, action_idx, reward, None))
        
        # Save the model weights
        self.model.save_weights("dqn_model.weights.h5")
        self.logger.info("DQN model saved.")

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
        
        if old_game_state is not None and new_game_state is not None:
            old_position = old_game_state['self'][3]
            new_position = new_game_state['self'][3]

            if new_game_state['coins']:
                old_min_dist = min([np.linalg.norm(np.array(old_position) - np.array(coin)) for coin in old_game_state['coins']])
                new_min_dist = min([np.linalg.norm(np.array(new_position) - np.array(coin)) for coin in new_game_state['coins']])

                if new_min_dist < old_min_dist:
                    reward_sum += 0.5

        self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
        return reward_sum

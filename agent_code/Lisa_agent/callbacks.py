import os
import pickle
import random
import numpy as np
from typing import List

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.q_table = {}  
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0 #später verringern
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.1
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    """
    state = state_to_features(game_state)
    state_tuple = tuple(state)

    if state_tuple not in self.q_table:
        self.q_table[state_tuple] = np.zeros(len(ACTIONS))

    if self.train and random.uniform(0, 1) < self.epsilon:
        self.logger.debug("Choosing action randomly for exploration.")
        return np.random.choice(ACTIONS)
    else:
        self.logger.debug("Choosing best action based on Q-table.")
        return ACTIONS[np.argmax(self.q_table[state_tuple])]

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e. a feature vector.
    """
    if game_state is None:
        return None

    agent_x, agent_y = game_state['self'][3]
    walls = game_state['field'] == -1
    bombs = np.zeros_like(game_state['field'])
    for bomb in game_state['bombs']:
        bombs[bomb[0]] = 1

    features = np.array([agent_x, agent_y, walls.sum(), bombs.sum()])
    return features


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    This function is called at the end of each game round.
    We save the Q-table and adjust epsilon for exploration decay.
    """

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)

    if self.train:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    self.logger.info(f"End of round. Epsilon: {self.epsilon}")


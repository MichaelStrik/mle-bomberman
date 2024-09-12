import os
from collections import namedtuple, deque

import numpy as np
import pickle
from typing import List

import events as e
from .callbacks import state_to_features, Actions

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
STEP_TOWARD_BFS_COIN = "STEP_TOWARD_BFS_COIN"

# Actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Q-table
    self.q_table = {}
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Training Q-table from scratch.")
        self.q_table = {}
    else:
        self.logger.info("Continuing training of Q-table from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)

    # parameters, info
    self.round = 0
    self.alpha = 0.2  # Lernrate davor 0.1
    self.gamma = 0.95  # Diskontfaktor um so höher um so stärker werden zufällige gewinne belohnt
    self.epsilon = 0.5  # exploration probability (epsilon-greedy method)
    self.epsilon_decay = 0.95  # Epsilon-Decay für Exploration-Exploitation Tradeoff
    self.alpha_decay = 0.995
    self.epsilon_min = 0.1



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
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

    new_state = state_to_features(new_game_state)
    old_state = state_to_features(old_game_state)

    old_state = tuple(old_state)
    new_state = tuple(new_state)
    
    action_idx = ACTIONS.index(self_action)

    # Logger: write game events to logger
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # NOTE generalization from smaller fields happens here
    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))
    
    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))

    coin_step_id = old_state[2][0]
    coin_step_name = Actions(coin_step_id).name
    if coin_step_name == self_action and e.COIN_COLLECTED not in events:
        events.append(STEP_TOWARD_BFS_COIN)

    
    # Reward: hand out rewards
    reward = reward_from_events(self, events, old_game_state, new_game_state)
    # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # Update: SARSA
    if new_game_state is not None:
        next_action = np.argmax(self.q_table[new_state])
        self.q_table[old_state][action_idx] = \
            self.q_table[old_state][action_idx] + \
            self.alpha * (reward + self.gamma * self.q_table[new_state][next_action] - self.q_table[old_state][action_idx])
    else:
        self.q_table[old_state][action_idx] = self.q_table[old_state][action_idx] + \
                                              self.alpha * (reward - self.q_table[old_state][action_idx])
    
    # NOTE Lisa adds a decay of epsilon in her code here --> good idea. is there an alternative?
    # How fast should epsilon decay?


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    self.round += 1
    self.epsilon *= self.epsilon_decay
    self.alpha   *= self.alpha_decay
    # self.epsilon = self.epsilon*self.round/(self.round+1)

    last_state = state_to_features(last_game_state)
    last_state = tuple(last_state)

    # TODO what if state is not existent in q-table yet?
    # do we fill the initial q-table with random values or zeros?
    # (we probably don't actually want to fill the q_table variable for memory reasons)

    action_idx = ACTIONS.index(last_action)
    
    # Reward mit den letzten Zuständen
    reward = reward_from_events(self, events, last_game_state, None)

    # End of round Q-Update
    self.q_table[last_state][action_idx] = self.q_table[last_state][action_idx] + \
                                           self.alpha * (reward - self.q_table[last_state][action_idx])

    # TODO save the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)


def reward_from_events(self, events: List[str], old_game_state, new_game_state) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    state_old = state_to_features(old_game_state)
    coin_dist = state_old[2][1]

    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        STEP_TOWARD_BFS_COIN: 1/(coin_dist)
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

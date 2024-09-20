import pickle
import os
import numpy as np
from .callbacks import state_to_features, get_valid_actions
import events as e

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Check whether a saved Q-table exists
    if os.path.isfile("my-sarsa-model.pt"):
        with open("my-sarsa-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
        self.logger.info("Loaded Q-table from my-sarsa-model.pt")
    else:
        self.q_table = {}
        self.logger.info("No saved model found. Starting with an empty Q-table.")
    
    self.alpha = 0.3
    self.gamma = 0.9
    self.epsilon = 1.0
    self.epsilon_decay = 0.95
    self.epsilon_min = 0.1
        
    self.last_positions = []
    self.last_actions = []
    

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

    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))
    
    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))

    action_idx = ACTIONS.index(self_action)
    
    # Save position and action
    current_position = new_game_state['self'][3]
    self.last_positions.append(current_position)
    self.last_actions.append(self_action)

    # Recognize whether it wiggles back and forth
    if len(self.last_positions) >= 4:
        if self.last_positions[-1] == self.last_positions[-3] and self.last_positions[-2] == self.last_positions[-4]:
            events.append("STUCK_IN_LOOP")

    # Limit the maximum length of the history of the last positions and actions
    if len(self.last_positions) > 5:
        self.last_positions.pop(0)
        self.last_actions.pop(0)

    # Reward
    reward = reward_from_events(self, events, old_game_state, new_game_state)

    # valid actions
    valid_actions = get_valid_actions(new_game_state)
       
    # Exploration vs Exploitation
    if np.random.rand() < self.epsilon:
        action = np.random.choice(valid_actions)
    else:
        # SARSA Update-Regel / Epsilon-greedy
        if new_game_state is not None:
            next_action = np.argmax(self.q_table[new_state])
            self.q_table[old_state][action_idx] = self.q_table[old_state][action_idx] + \
                                                self.alpha * (reward + self.gamma * self.q_table[new_state][next_action] - self.q_table[old_state][action_idx])
        else:
            self.q_table[old_state][action_idx] = self.q_table[old_state][action_idx] + \
                                                self.alpha * (reward - self.q_table[old_state][action_idx])

    # Epsilon-Decay
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay


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
    
    if last_state not in self.q_table:
        self.q_table[last_state] = np.zeros(len(ACTIONS))

    action_idx = ACTIONS.index(last_action)
    
    # Reward mit den letzten Zuständen
    reward = reward_from_events(self, events, last_game_state, None)

    # End of round Q-Update
    self.q_table[last_state][action_idx] = self.q_table[last_state][action_idx] + \
                                           self.alpha * (reward - self.q_table[last_state][action_idx])

    # Modell speichern
    model_file = "my-sarsa-model.pt"
    with open(model_file, "wb") as file:
        pickle.dump(self.q_table, file)
    self.logger.info(f"Q-table saved to {model_file}.")


def reward_from_events(self, events: list[str], old_game_state: dict, new_game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
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
        old_features = state_to_features(old_game_state)
        new_features = state_to_features(new_game_state)

        if old_features is not None and new_features is not None:
            # Distance to coins 
            old_min_distance = old_features[2]  
            new_min_distance = new_features[2]  

            # Reward if closer to coin 
            if new_min_distance < old_min_distance:
                reward_sum += 0.5
            elif new_min_distance == old_min_distance:
                reward_sum -= 0.05  
            else:
                reward_sum -= 0.5  

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum
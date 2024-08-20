from collections import namedtuple, deque

import numpy as np
import pickle
from typing import List

import events as e
from .callbacks import state_to_features

#neu von mir 
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 4  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
STUCK_PENALTY = "STUCK_PENALTY" #new

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # New Event 1: Check if the agent moved towards a coin
    if moved_towards_coin(old_game_state, new_game_state):
        events.append("MOVED_TOWARDS_COIN")

    # New Event 2: Check if the agent moved towards danger (e.g., closer to a bomb)
    if moved_towards_danger(old_game_state, new_game_state):
        events.append("MOVED_TOWARDS_DANGER")

    # New Event 3: Check if the agent escaped from a bomb
    if escaped_bomb(old_game_state, new_game_state):
        events.append("ESCAPED_BOMB")

    # New Event 4: Check if the agent moved into a safe zone (no bombs nearby)
    if new_game_state and is_in_safe_zone(new_game_state):
        events.append("MOVED_TO_SAFE_ZONE")
    
    # new event 5: Stuck     
    if old_game_state['self'][3] == new_game_state['self'][3]:
        events.append(STUCK_PENALTY)

    # new evemt 6:  Check if the agent placed a bomb with no enemies nearby
    if self_action == 'BOMB' and not is_enemy_nearby(old_game_state):
        events.append("BOMB_WITH_NO_ENEMY_NEARBY")

    if self_action == 'BOMB':
        if not is_enemy_nearby(old_game_state) or not has_safe_escape(old_game_state):
            events.append("UNSAFE_BOMB_PLACEMENT")

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(
        state_to_features(old_game_state), 
        self_action, 
        state_to_features(new_game_state), 
        reward_from_events(self, events)
    ))

# new event functions
def is_in_safe_zone(game_state):
    """
    Check if the agent is in a safe zone (no bombs nearby).
    """
    agent_x, agent_y = game_state['self'][3]
    for bomb in game_state['bombs']:
        bomb_x, bomb_y = bomb[0]
        # If a bomb is within a certain distance (e.g., 3 tiles), it's not safe
        if abs(agent_x - bomb_x) <= 3 and abs(agent_y - bomb_y) <= 3:
            return False
    return True
def manhattan_distance(position1, position2):
    return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])


def moved_towards_danger(old_game_state, new_game_state):
    if old_game_state is None or new_game_state is None:
        return False

    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]
    old_bombs = old_game_state['bombs']
    new_bombs = new_game_state['bombs']

    def closest_bomb(position, bombs):
        if len(bombs) == 0:
            return None
        return min(bombs, key=lambda bomb: manhattan_distance(position, bomb[0]))

    old_closest = closest_bomb(old_position, old_bombs)
    new_closest = closest_bomb(new_position, new_bombs)

    if old_closest is None or new_closest is None:
        return False

    old_distance = manhattan_distance(old_position, old_closest[0])
    new_distance = manhattan_distance(new_position, new_closest[0])

    return new_distance < old_distance

def escaped_bomb(old_game_state, new_game_state):
    if old_game_state is None or new_game_state is None:
        return False

    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]
    old_bombs = old_game_state['bombs']

    def closest_bomb(position, bombs):
        if len(bombs) == 0:
            return None
        return min(bombs, key=lambda bomb: manhattan_distance(position, bomb[0]))

    old_closest = closest_bomb(old_position, old_bombs)

    if old_closest is None:
        return False

    old_distance = manhattan_distance(old_position, old_closest[0])
    new_distance = manhattan_distance(new_position, old_closest[0])

    return new_distance > old_distance


def moved_towards_coin(old_game_state, new_game_state):
    if old_game_state is None or new_game_state is None:
        return False

    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]
    old_coins = old_game_state['coins']
    new_coins = new_game_state['coins']

    # Find the closest coin in both the old and new game states
    def closest_coin(position, coins):
        if len(coins) == 0:
            return None
        return min(coins, key=lambda coin: manhattan_distance(position, coin))

    old_closest = closest_coin(old_position, old_coins)
    new_closest = closest_coin(new_position, new_coins)

    # If no coins are found, return False
    if old_closest is None or new_closest is None:
        return False

    old_distance = manhattan_distance(old_position, old_closest)
    new_distance = manhattan_distance(new_position, new_closest)

    # Return True if the agent moved closer to the closest coin
    return new_distance < old_distance

def is_enemy_nearby(game_state, threshold=4):
    """
    Checks if any enemy is within a certain Manhattan distance (threshold) from the agent.
    """
    agent_position = game_state['self'][3]
    enemies = game_state['others']  # This contains positions of other players

    for enemy in enemies:
        enemy_position = enemy[3]  # Position of the enemy
        if manhattan_distance(agent_position, enemy_position) <= threshold:
            return True
    
    return False

def has_safe_escape(game_state):
    """
    Check if the agent has a safe escape route after placing a bomb.
    This checks if there is a path to a safe zone (no bombs and no danger).
    """
    agent_position = game_state['self'][3]
    
    # Check adjacent tiles for a safe path
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # DOWN, RIGHT, UP, LEFT
    for direction in directions:
        next_position = (agent_position[0] + direction[0], agent_position[1] + direction[1])
        if is_position_safe(game_state, next_position):
            return True
    return False

def is_position_safe(game_state, position):
    """
    Check if a given position is safe (no bombs nearby and not in blast radius).
    """
    for bomb in game_state['bombs']:
        bomb_x, bomb_y = bomb[0]
        if abs(position[0] - bomb_x) <= 3 and abs(position[1] - bomb_y) <= 3:
            return False  # In the blast radius of a bomb
    return True

def print_q_table(self):
    for state, actions in self.q_table.items():
        print(f"State: {state}, Q-Values: {actions}")

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

    # Final state transition at the end of the round
    self.transitions.append(Transition(
        state_to_features(last_game_state), 
        last_action, 
        None,  # No next state since the round is over
        reward_from_events(self, events)
    ))

    # Update the Q-table after the round ends using the transitions
    for transition in self.transitions:
        # Konvertiere den Zustand (np.ndarray) in einen hashbaren Typ wie tuple
        state = tuple(transition.state) if transition.state is not None else None
        next_state = tuple(transition.next_state) if transition.next_state is not None else None

        # Hole den alten Q-Wert und berechne den neuen
        old_value = self.q_table.get(state, np.zeros(len(ACTIONS)))[ACTIONS.index(transition.action)]
        next_max = np.max(self.q_table.get(next_state, np.zeros(len(ACTIONS)))) if next_state is not None else 0
        new_value = old_value + self.learning_rate * (transition.reward + self.discount_factor * next_max - old_value)
        if state is not None:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(ACTIONS))
            self.q_table[state][ACTIONS.index(transition.action)] = new_value
            
    # Decay epsilon after each round (for exploration-exploitation trade-off)
    self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    # Store the model (Q-table)
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)
    
    print_q_table(self) 

    self.logger.info(f"End of round. Epsilon: {self.epsilon}")


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5.0,  # Reward for collecting a coin
        e.KILLED_OPPONENT: 10.0,  # High reward for eliminating an opponent
        "MOVED_TO_SAFE_ZONE": 3.0, 
        "MOVED_TOWARDS_COIN": 2.0, 
        "MOVED_TOWARDS_DANGER": -3.0,  
        "ESCAPED_BOMB": 1.0,
        "BOMB_WITH_NO_ENEMY_NEARBY": -25.0, 
        STUCK_PENALTY: -1.0 ,
        "UNSAFE_BOMB_PLACEMENT": -50.0
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
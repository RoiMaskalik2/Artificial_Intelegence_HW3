import numpy as np
import random

DRONE_LOCATION = "drone_location"
RESET = "reset"
WAIT = "wait"
PICK = "pick"
MOVE_UP = "move_up"
MOVE_DOWN = "move_down"
MOVE_LEFT = "move_left"
MOVE_RIGHT = "move_right"
DELIVER = "deliver"

GAMMA = 0.9
ALPHA = 0.9
random_action_prob = 0.1

NUM_PACKAGES_DRONE = 2

ids = ["111111111", "222222222"]


class DroneAgent:
    def __init__(self, n, m):
        self.mode = 'train'  # do not change this!

        self.actions = [RESET, WAIT, PICK, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, DELIVER]
        # TODO: maybe increase the state to include number of packages
        self.current_packages_on_drone = 0

        self.q_values = np.zeros((n, m, NUM_PACKAGES_DRONE, len(self.actions)))
        self.action_num_dict = self.create_action_num_dict()
        self.num_action_dict = self.create_num_action_dict()

    def create_action_num_dict(self):
        # Create a dictionary that converts an action to a number for np.array access
        return {
            RESET: 0,
            WAIT: 1,
            PICK: 2,
            MOVE_UP: 3,
            MOVE_DOWN: 4,
            MOVE_LEFT: 5,
            MOVE_RIGHT: 6,
            DELIVER: 7
        }

    def create_num_action_dict(self):
        # Create a dictionary that converts a number to an action string
        return {
            0: RESET,
            1: WAIT,
            2: PICK,
            3: MOVE_UP,
            4: MOVE_DOWN,
            5: MOVE_LEFT,
            6: MOVE_RIGHT,
            7: DELIVER
        }

    def select_action(self, obs0):
        # TODO: maybe implement differently between train and eval

        obs_location_x, obs_location_y = obs0[DRONE_LOCATION]
        best_action_index = np.argmax(
            self.q_values[obs_location_x, obs_location_y, self.current_packages_on_drone])

        if self.mode == 'train' and random.uniform(0, 1) < random_action_prob:
            action = random.choice(self.actions)
        else:
            action = self.num_action_dict[best_action_index]

        if action == PICK:
            self.current_packages_on_drone += 1
        if action == DELIVER:
            self.current_packages_on_drone -= 1

        return action

    def train(self):
        self.mode = 'train'  # do not change this!

    def eval(self):
        self.mode = 'eval'  # do not change this!

    def update(self, obs0, action, obs1, reward):
        # TODO: maybe convert the obs dictionary to a string with pickle
        obs0_location_x, obs0_location_y = obs0[DRONE_LOCATION]
        obs1_location_x, obs1_location_y = obs1[DRONE_LOCATION]

        action_index = self.action_num_dict[action]
        old_q_value = self.q_values[
            obs0_location_x, obs0_location_y, self.current_packages_on_drone, action_index]
        # Got stuck on adding package number on the next action
        td_value = reward + GAMMA * np.max(
            self.q_values[obs1_location_x, obs1_location_y]) - old_q_value

        self.q_values[obs0_location_x, obs0_location_y, self.current_packages_on_drone,
                      action_index] = old_q_value + ALPHA * \
                                      td_value

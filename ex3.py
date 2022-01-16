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
PACKAGES = "packages"

GAMMA = 0.9
ALPHA = 0.9
random_action_prob = 0.2

NUM_PACKAGES_DRONE = 2

START_OF_EPISODE = 0
MIDDLE_OF_EPISODE = 1
END_OF_EPISODE = 2
NUM_PARTS_OF_EPISODE = 3

ids = ["111111111", "222222222"]


class DroneAgent:
    def __init__(self, n, m):
        self.mode = 'train'  # do not change this!

        self.actions = [RESET, WAIT, PICK, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, DELIVER]
        # TODO: maybe increase the state to include number of packages
        self.current_packages_on_drone = 0
        self.visited = set()
        self.current_round = 0

        self.q_values = np.zeros((n, m,NUM_PARTS_OF_EPISODE, NUM_PACKAGES_DRONE + 1, 2,
                                  len(self.actions)))
        self.action_num_dict = self.create_action_num_dict()
        self.num_action_dict = self.create_num_action_dict()

    def get_part_of_episode(self):
        if self.current_round < 10:
            return START_OF_EPISODE
        if self.current_round < 20:
            return MIDDLE_OF_EPISODE
        return END_OF_EPISODE
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

    def package_exists_on_drone_location(self, obs):
        location = obs[DRONE_LOCATION]
        packages = obs[PACKAGES]

        return int(sum(1 for pl in packages if pl[1] == location) > 0)

    def get_packages_on_drone(self, obs):
        packages = obs[PACKAGES]
        return sum(1 for pl in packages if isinstance(pl[1], str))

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
        if not obs0[PACKAGES]:
            return RESET
        obs_location_x, obs_location_y = obs0[DRONE_LOCATION]
        num_packages_on_drone = self.get_packages_on_drone(obs0)
        is_package_exists = self.package_exists_on_drone_location(obs0)

        state_obs = (obs_location_x, obs_location_y, num_packages_on_drone)
        q_values = self.q_values[
            obs_location_x, obs_location_y,self.get_part_of_episode(), num_packages_on_drone, is_package_exists]

        if self.mode == 'train' and (random.uniform(0, 1) < random_action_prob):
            if 0 in q_values:
                not_visited_action_index = np.random.choice(np.flatnonzero(q_values == 0))
                action = self.num_action_dict[not_visited_action_index]
            else:
                action = random.choice(self.actions)
        else:
            best_action_index = np.random.choice(np.flatnonzero(q_values == q_values.max()))
            action = self.num_action_dict[best_action_index]

        return action

    def train(self):
        self.mode = 'train'  # do not change this!

    def eval(self):
        self.mode = 'eval'  # do not change this!

    def update(self, obs0, action, obs1, reward):
        # TODO: maybe convert the obs dictionary to a string with pickle
        if action == DELIVER and reward == -1:
            reward = -20

        if reward == 100:
            reward *= 100

        obs0_location_x, obs0_location_y = obs0[DRONE_LOCATION]
        obs1_location_x, obs1_location_y = obs1[DRONE_LOCATION]

        obs0_packages_on_drone = self.get_packages_on_drone(obs0)
        obs1_packages_on_drone = self.get_packages_on_drone(obs1)

        current_part_of_episode = self.get_part_of_episode()
        self.current_round += 1
        next_part_of_episode = self.get_part_of_episode()

        is_package_exists_obs0 = self.package_exists_on_drone_location(obs0)
        is_package_exists_obs1 = self.package_exists_on_drone_location(obs1)

        visited_observation = (obs0_location_x, obs0_location_y,current_part_of_episode,
                               obs0_packages_on_drone)
        self.visited.add(visited_observation)
        action_index = self.action_num_dict[action]

        old_q_value = self.q_values[obs0_location_x, obs0_location_y,current_part_of_episode,
                                    obs0_packages_on_drone,
                                    is_package_exists_obs0, action_index]
        td_value = reward + GAMMA * np.max(
            self.q_values[
                obs1_location_x, obs1_location_y,next_part_of_episode, obs1_packages_on_drone,
                is_package_exists_obs1]) - old_q_value

        self.q_values[obs0_location_x, obs0_location_y,current_part_of_episode,
                      obs0_packages_on_drone,
                      is_package_exists_obs0, action_index] = (
                old_q_value + ALPHA * td_value)

        if self.current_round == 29 or action == RESET:
            self.current_round = 0

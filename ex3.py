import numpy as np
import random

DRONE_LOCATION = "drone_location"
TARGET_LOCATION = "target_location"
DRONE = "drone"
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

        self.learning_actions = [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, RESET]

        # TODO: maybe increase the state to include number of packages
        self.current_packages_on_drone = 0
        # self.visited = set()
        self.current_round = 0

        self.q_values = {}
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
            MOVE_UP: 0,
            MOVE_DOWN: 1,
            MOVE_LEFT: 2,
            MOVE_RIGHT: 3,
            RESET: 4
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
            0: MOVE_UP,
            1: MOVE_DOWN,
            2: MOVE_LEFT,
            3: MOVE_RIGHT,
            4: RESET
        }

    def select_action(self, obs0):
        print(obs0)
        # TODO: maybe implement differently between train and eval
        if not obs0[PACKAGES]:
            return RESET

        if self.can_deliver(obs0):
            return DELIVER

        if self.can_pick(obs0):
            return PICK
        # obs_location_x, obs_location_y = obs0[DRONE_LOCATION]
        # num_packages_on_drone = self.get_packages_on_drone(obs0)
        # is_package_exists = self.package_exists_on_drone_location(obs0)

        # state_obs = (obs_location_x, obs_location_y, num_packages_on_drone)

        obs_as_key = self.obs_to_key(obs0)

        self.add_obs_if_not_exists(obs_as_key)

        q_values = self.q_values[obs_as_key]

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
        if action == DELIVER and reward == -1:  # Illegal deliver
            reward = -20

        if reward == 100:
            reward *= 100

        # obs0_location_x, obs0_location_y = obs0[DRONE_LOCATION]
        # obs1_location_x, obs1_location_y = obs1[DRONE_LOCATION]

        # obs0_packages_on_drone = self.get_packages_on_drone(obs0)
        # obs1_packages_on_drone = self.get_packages_on_drone(obs1)

        # current_part_of_episode = self.get_part_of_episode()
        self.current_round += 1
        # next_part_of_episode = self.get_part_of_episode()

        # is_package_exists_obs0 = self.package_exists_on_drone_location(obs0)
        # is_package_exists_obs1 = self.package_exists_on_drone_location(obs1)

        # visited_observation = (obs0_location_x, obs0_location_y, current_part_of_episode,
        #                        obs0_packages_on_drone)
        # self.visited.add(visited_observation)

        if action in self.learning_actions:
            action_index = self.action_num_dict[action]

            obs0_as_key = self.obs_to_key(obs0)

            obs1_as_key = self.obs_to_key(obs1)

            old_q_value = self.q_values[obs0_as_key]

            self.add_obs_if_not_exists(obs1_as_key)

            td_value = reward + GAMMA * np.max(
                self.q_values[obs1_as_key]) - old_q_value

            self.q_values[obs0_as_key] = (
                    old_q_value + ALPHA * td_value)

        if self.current_round == 29 or action == RESET:
            self.current_round = 0

    def can_deliver(self, obs):
        drone_location = obs[DRONE_LOCATION]
        target_location = obs[TARGET_LOCATION]
        if drone_location == target_location:
            packages = obs[PACKAGES]
            for package, package_location in packages:
                if package_location == DRONE:
                    return True
        return False

    def can_pick(self, obs):
        drone_location = obs[DRONE_LOCATION]
        packages_on_drone = 0
        pickable_packages = 0

        packages = obs[PACKAGES]
        for package, package_location in packages:
            if package_location == DRONE:
                packages_on_drone += 1
            elif package_location == drone_location:
                pickable_packages += 1

        return packages_on_drone < 2 and pickable_packages > 0


    def obs_to_key(self, obs):
        '''
        Turns an observation into string key
        '''
        return str(obs)

    def add_obs_if_not_exists(self, obs_as_key):
        if obs_as_key not in self.q_values:
            self.q_values[obs_as_key] = np.zeros(len(self.learning_actions))
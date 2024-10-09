import time
from abc import ABC

import gym
from gym import spaces
import numpy as np
import torchvision
from models.resnet101_network import myResnet
from utils.MT_device import MT_Device
from loguru import logger
import torch


class wz_1v1_env(gym.Env, ABC):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # - human: render to the current display or terminal and
        #  return nothing. Usually for human consumption.
        # - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
        #  representing RGB values for an x-by-y pixel image, suitable
        #  for turning into a video.
        # - ansi: Return a string (str) or StringIO.StringIO containing a
        #  terminal-style text representation. The text can include newlines
        #  and ANSI escape sequences (e.g. for colors).

        'video.frames_per_second': 1
    }

    def __init__(self, device, gpu, resnet101=None):
        if resnet101 is None:
            resnet101 = torchvision.models.resnet101(pretrained=True).eval()
            self.resnet101 = myResnet(resnet101).cuda(device).requires_grad_(False)
        else:
            self.resnet101 = resnet101
        if type(device) == type('str'):  # '770b1af'
            self.device = MT_Device(device, gpu)
        else:
            self.device = device
        self.gpu = gpu
        # state space
        self.state = None
        # action space
        self.action_space = spaces.Discrete(5)  # 0still,1NorthEast,2NorthWest,3SouthEast,4SouthWest
        self.action_direction = dict()
        x = 0.2
        y = 0.8
        right_delta = 0.05
        upper_delta = 0.18
        self.action_direction['M'] = [x, y]
        self.action_direction['NE'] = [x + right_delta, y - upper_delta]
        self.action_direction['NW'] = [x - right_delta, y - upper_delta]
        self.action_direction['SE'] = [x + right_delta, y + upper_delta]
        self.action_direction['SW'] = [x - right_delta, y + upper_delta]
        self.action_direction_postion_record = self.action_direction['M']
        self.action_direction_counts = 0

        # reward space
        self.hero_position = [0.485, 0.488]
        self.canvas_range = [1080, 2400]
        self.last_red_map_map_level = None

    def reset_env(self, screen_torch):
        # screen = self.device.UI_device.get_screen()
        # screen_np = np.asarray(screen)
        # screen_torch = torch.from_numpy(screen_np).cuda(self.gpu).unsqueeze(0).permute(0, 3, 2, 1) / 255
        _, self.state = self.resnet101(screen_torch)
        self.action_direction_counts = 0
        block, last_red_map = self.device.UI_device.mask_red_lane(self.hero_position, self.canvas_range)
        self.last_red_map_map_level = self._cal_red_map_level(block, last_red_map)
        return self.state

    def _cal_red_map_level(self, block, red_map):
        level = int(block / 2)
        red_map_level = list()
        for l in range(level):  # from outer to inner
            sum_red = 0
            for i in range(block):
                sum_red += red_map[l][i]
                sum_red += red_map[block - 1 - l][i]
                sum_red += red_map[i][l]
                sum_red += red_map[i][block - 1 - l]
            sum_red = sum_red - red_map[l][l] - red_map[block - 1 - l][block - 1 - l] - red_map[l][block - 1 - l] - \
                      red_map[block - 1 - l][l]
            red_map_level.append(sum_red)
        return red_map_level

    def step_env(self, action):
        # 0still,1NorthEast,2NorthWest,3SouthEast,4SouthWest
        no_up = True
        if action == 0:
            stop_position = self.action_direction['M']
            no_up = False
        elif action == 1:
            stop_position = self.action_direction['NE']
        elif action == 2:
            stop_position = self.action_direction['NW']
        elif action == 3:
            stop_position = self.action_direction['SE']
        elif action == 4:
            stop_position = self.action_direction['SW']
        else:
            raise Exception('action unset')
        self.device.slide(self.action_direction_postion_record, stop_position, flutter=0.01, no_down=False, no_up=no_up)
        logger.info('action: %s start: %s stop:%s' % (action, self.action_direction_postion_record, stop_position))
        self.action_direction_postion_record = stop_position
        self.action_direction_counts += 1

        #time.sleep(0.3)  # wait for action move
        # screen_torch = self.device.UI_device.get_screen_torch()
        # _, self.state = self.resnet101(screen_torch)
        # block, red_map = self.device.UI_device.mask_red_lane(hero_position=self.hero_position,
        #                                                      canvas_range=self.canvas_range)
        # red_map_level = self._cal_red_map_level(block, red_map)
        reward = 0
        # for j in range(int(block / 2)):
        #     red_more = red_map_level[j] - self.last_red_map_map_level[j]
        #     if red_more > 0:
        #         reward += red_more * (j + 1)
        return self.state, reward


if __name__ == '__main__':
    env = wz_1v1_env()
    env.reset()
    env.step(env.action_space.sample())
    print(env.state)
    env.step(env.action_space.sample())
    print(env.state)

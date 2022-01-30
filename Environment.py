import time

import gym
from gym import spaces
import numpy as np
import pyscreenshot as ImageGrab
import cv2
import pyautogui as py


def processImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 31))
    return image


def checkDead(image):
    # 48, 17 -> 1
    if image[17][48] == 1:
        return True  # Is Dead
    else:
        return False  # Isn't dead


class SnowRiderEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}  # Will run in browser so its console only suck my dick if you disagree
    ACTION_SPACE = 4  # Left, Jump, Right, Do Nothing
    # Resized to 64x64 because of memory issues
    HEIGHT = 31
    WIDTH = 200
    N_CHANNELS = 1  # Grayscale image

    def __init__(self):
        super(SnowRiderEnv, self).__init__()
        self.action_space = spaces.Discrete(self.ACTION_SPACE)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, self.N_CHANNELS), dtype=np.uint8)

    def step(self, action):
        # 800x124 -> 200x31d
        observation = ImageGrab.grab(bbox=(551, 507, 1351, 631))
        observation = processImage(np.asarray(observation))

        reward = 1  # Give a reward for being alive... thats it
        done = checkDead(observation)
        info = {}

        if action == 0:
            py.keyDown('a')
            time.sleep(0.07)
            py.keyUp('a')
            print('a')
        elif action == 1:
            py.keyDown('d')
            time.sleep(0.07)
            py.keyUp('d')
            print('d')
        elif action == 2:
            py.keyDown('space')
            time.sleep(0.07)
            py.keyUp('space')
            print('space')
        else:
            time.sleep(0.07)
            print('nothing')

        # Else do nothing

        return observation, reward, done, info

    def reset(self):
        py.press('space')
        time.sleep(1)
        py.press('space')

        observation = ImageGrab.grab(bbox=(551, 507, 1351, 631))
        observation = processImage(np.asarray(observation))

        return observation  # reward, done, info can't be included

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # Fuck u look at the browser instead

    def close(self):
        pass

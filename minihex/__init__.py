from gym.envs.registration import register
import numpy as np
import random


register(
    id='hex-v1',
    entry_point='minihex.HexEnv:HexEnv'
)
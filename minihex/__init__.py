from gym.envs.registration import register

register(
    id='hex-v1',
    entry_point='minihex.HexEnv:HexEnv'
)
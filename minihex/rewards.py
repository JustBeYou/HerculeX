
def end_game_reward(env):
    if env.winner == env.player:
        return 1
    elif env.winner == env.opponent:
        return -1
    return 0
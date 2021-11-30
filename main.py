import gym
from minihex.RandomAgent import RandomAgent
from minihex.rewards import end_game_reward
from time import perf_counter_ns, sleep

BOARD_SIZE = 11

def main():
    our_agent = RandomAgent(board_size=BOARD_SIZE)
    opponent_agent = RandomAgent(board_size=BOARD_SIZE)
    env = gym.make("hex-v1",
               opponent_policy=opponent_agent.get_action,
               reward_function=end_game_reward,
               board_size=BOARD_SIZE)

    run_episode(env, our_agent, debug=True)

    episodes = 1000
    execs = profile_it(lambda: run_episode(env, our_agent), count=episodes)
    print(f"{execs} executions per second (ran {episodes} episodes)")

    #avg = average_reward(env, our_agent, episodes=episodes)
    #print(f"Average reward over {episodes} episodes: {avg}")

def profile_it(func, count=1000):
    start = perf_counter_ns()
    for i in range(count):
        func()
    end = perf_counter_ns()
    return int(count / (end - start) * 1e9)

def average_reward(env, agent, episodes=1000,):
    total_reward = 0
    for i in range(episodes):
        total_reward += run_episode(env, agent)
    return total_reward / episodes

def run_episode(env, agent, debug=False):
    total_reward = 0
    state, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if debug:
            env.render()
            sleep(0.1)
            
    if debug:
        print(f"Reward: {total_reward}")
        input("> Press any key to continue")

    return total_reward

if __name__ == "__main__":
    main()
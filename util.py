from time import perf_counter_ns, sleep

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
        if debug:
            env.render()
            sleep(0.1)

        action = agent.get_action(state=(state[0].copy(), state[1].copy()), connected_stones=env.simulator.regions.copy(), history=env.history)
        state, reward, done, info = env.step(action)
        total_reward += reward

    if debug:
        env.render()
        sleep(0.1)

            
    if debug:
        print(f"Reward: {total_reward}")
        input("> Press any key to continue")

    return total_reward
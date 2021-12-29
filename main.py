import gym
import h5py
from minihex.rewards import end_game_reward
import argparse
import util
from numpy import mean
import numpy as np
from agents.experience_utils.experience_collector import ExperienceCollector
from agents.experience_utils.experience_buffer import ExperienceBuffer

BOARD_SIZE = 11

def main():
    args = parse_args()
    action_handlers[args.action](args)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        'action',
        choices=action_handlers.keys(),
        metavar='action',
        help=f"Could be one of the following: {', '.join(action_handlers.keys())}"
    )

    parser.add_argument(
        '--agent',
        help="Our agent, we want the best for him/her/their.",
        default='RandomAgent',
    )

    parser.add_argument(
        '--opponent',
        help="The enemy of the state. Must perish.",
        default='RandomAgent',
    )

    parser.add_argument(
        '--board-size',
        default=BOARD_SIZE,
    )

    parser.add_argument(
        '--reward-function',
        choices=rewards.keys(),
        default='end_game',
        metavar='reward',
        help=f"Could be one of the following: {', '.join(rewards.keys())}"
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed the PRNGs in this env'
    )

    parser.add_argument(
        '--load-agent-path',
        default=None,
    )

    parser.add_argument(
        '--load-opponent-path',
        default=None,
    )

    parser.add_argument(
        '--save-agent-path',
        default=None,
    )

    parser.add_argument(
        '--save-opponent-path',
        default=None,
    )

    return parser.parse_args()

def get_class_of_module(mod_name, cls_name):
    mod = __import__(f'{mod_name}.{cls_name}')
    mod = getattr(mod, cls_name)
    return getattr(mod, cls_name)

def create_env(args, col1, col2, col_flag=False):
    our_agent_class = get_class_of_module('agents', args.agent)
    opponent_class = get_class_of_module('agents', args.opponent)

    our_agent = our_agent_class(board_size=args.board_size, constant=0.5, num_simulations=10)
    opponent_agent = opponent_class(board_size=args.board_size)

    if col_flag:
        our_agent.set_collector(col1)
        opponent_agent.set_collector(col2)

    if args.load_agent_path is not None:
        our_agent.load(args.load_agent_path)

    if args.load_opponent_path is not None:
        opponent_agent.load(args.load_opponent_path)

    env = gym.make("hex-v1",
               opponent_policy=opponent_agent.get_action,
               reward_function=rewards[args.reward_function],
               board_size=BOARD_SIZE)

    env.seed(args.seed)

    return env, our_agent, opponent_agent

def benchmark_perf(args):
    our_collector, opponent_collector = ExperienceCollector(), ExperienceCollector()
    env, our_agent, opponent_agent = create_env(args, our_collector, opponent_collector)

    episodes = args.episodes
    execs = util.profile_it(lambda: util.run_episode(env, our_agent), count=episodes)
    print(f"{execs} executions per second (ran {episodes} episodes)")

def benchmark_reward(args):
    env, our_agent, opponent_agent = create_env(args)

    episodes = args.episodes
    avg = util.average_reward(env, our_agent, episodes=args.episodes)
    print(f"Average reward over {episodes} episodes: {avg}")

def debug_run(args):
    our_collector, opponent_collector = ExperienceCollector(), ExperienceCollector()
    env, our_agent, opponent_agent = create_env(args, our_collector, opponent_collector)

    util.run_episode(env, our_agent, debug=True)

def combine_experience(collectors):
    combined_states = np.concatenate([c.states for c in collectors])
    combined_actions = np.concatenate([c.actions for c in collectors])
    combined_rewards = np.concatenate([c.rewards for c in collectors])

    return ExperienceBuffer(
        states = combined_states,
        actions = combined_actions,
        rewards= combined_rewards)

def run(args):
    our_collector, opponent_collector = ExperienceCollector(), ExperienceCollector()
    env, our_agent, opponent_agent = create_env(args, our_collector, opponent_collector, True)

    episodes = args.episodes
    rewards_hist = []
    period = 1000
    for i in range(1, episodes+1):
        our_collector.init_episode()
        opponent_collector.init_episode()

        if i % period == 0:
            print(f"Training {i/episodes*100:.2f}% done. Average reward last {period} episodes: {mean(rewards_hist):.2f}")

        reward = util.run_episode(env, our_agent, debug=False)

        our_collector.complete_episode(reward)
        opponent_collector.complete_episode(-reward)

        exp = combine_experience([our_collector, opponent_collector])

        with open('test_exp.json', 'w') as file:
            exp.save(file)

        if len(rewards_hist) >= period:
            rewards_hist.pop(0)
        rewards_hist.append(reward)

    if args.save_agent_path is not None:
        our_agent.save(args.save_agent_path)

    if args.save_opponent_path is not None:
        opponent_agent.save(args.save_opponent_path)

action_handlers = {
    "benchmark-perf": benchmark_perf,
    "benchmark-reward": benchmark_reward,
    "debug-run": debug_run,
    "run": run,
}

rewards = {
    "end_game": end_game_reward,
}

if __name__ == "__main__":
    main()
from environment.intersection_env import IntersectionEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

def split_state(full_state, agent_id, total_agents=2):
    size = len(full_state) // total_agents
    return full_state[agent_id * size:(agent_id + 1) * size]

def train_multi_agent(agent_type="q", episodes=100):
    env = IntersectionEnv()
    state_size = len(split_state(env.reset(), 0))
    action_size = 4  # N, E, S, W directions

    if agent_type == "q":
        agentA = QLearningAgent(state_size, action_size)
        agentB = QLearningAgent(state_size, action_size)
    elif agent_type == "dqn":
        agentA = DQNAgent(state_size, action_size)
        agentB = DQNAgent(state_size, action_size)
    else:
        raise ValueError("Invalid agent type")

    rewards = []
    avg_queues = []

    for ep in range(episodes):
        full_state = env.reset()
        total_reward = 0
        queue_sum = 0
        steps = 0

        for t in range(200):
            stateA = split_state(full_state, 0)
            stateB = split_state(full_state, 1)

            actionA = agentA.get_action(stateA)
            actionB = agentB.get_action(stateB)
            actions = [actionA, actionB]

            next_state, reward, done, _ = env.step(actions)

            nextA = split_state(next_state, 0)
            nextB = split_state(next_state, 1)

            if agent_type == "q":
                agentA.update(stateA, actionA, reward / 2, nextA)
                agentB.update(stateB, actionB, reward / 2, nextB)
            else:
                agentA.remember(stateA, actionA, reward / 2, nextA)
                agentB.remember(stateB, actionB, reward / 2, nextB)
                agentA.train_step()
                agentB.train_step()

            full_state = next_state
            total_reward += reward
            queue_sum += np.mean(env.queues)
            steps += 1

        avg_queue = queue_sum / steps
        rewards.append(total_reward)
        avg_queues.append(avg_queue)

        print(f"[{agent_type.upper()}] Episode {ep+1} - Total Reward: {total_reward:.1f}, AvgQueue: {avg_queue:.2f}")

    # Save results
    with open("training_metrics.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward", "AvgVehicleQueue"])
        for i in range(episodes):
            writer.writerow([i+1, rewards[i], avg_queues[i]])

    return rewards

def plot_rewards(rewards, label):
    plt.plot(rewards, label=label)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Multi-Agent Learning Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["q", "dqn"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    rewards = train_multi_agent(agent_type=args.agent, episodes=args.episodes)
    plot_rewards(rewards, label=f"{args.agent.upper()} (A & B)")

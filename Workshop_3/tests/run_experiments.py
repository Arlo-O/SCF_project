from environment.intersection_env import IntersectionEnv
from agents.q_learning_agent import QLearningAgent

env = IntersectionEnv()
agent = QLearningAgent(state_size=10, action_size=4)

for episode in range(100):
    state = env.reset()
    total_reward = 0
    for _ in range(200):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    print(f"Episode {episode}, Total Reward: {total_reward}")

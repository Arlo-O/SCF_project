def compute_reward(action):
    # Encourage clearing traffic and respecting pedestrian needs
    if action == 0:
        return +1
    elif action == 1:
        return -0.5
    else:
        return 0

import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_metrics(csv_file, label_prefix=""):
    data = pd.read_csv(csv_file)

    episodes = data["Episode"]
    rewards = data["TotalReward"]
    queues = data["AvgVehicleQueue"]
    peds = data["PedestriansServed"]
    ped_waits = data["AvgPedestrianWait"]

    # --- Reward plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Reward per Episode ({label_prefix})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/reward_plot_{label_prefix.lower()}.png")
    print(f"[✓] Saved: reward_plot_{label_prefix.lower()}.png")

    # --- Vehicle Queue plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, queues, label="Avg Vehicle Queue", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Avg Queue Length")
    plt.title(f"Average Vehicle Queue ({label_prefix})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/queue_plot_{label_prefix.lower()}.png")
    print(f"[✓] Saved: queue_plot_{label_prefix.lower()}.png")

    # --- Pedestrian Count plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, peds, label="Pedestrians Served", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.title(f"Pedestrians Served per Episode ({label_prefix})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/pedestrians_plot_{label_prefix.lower()}.png")
    print(f"[✓] Saved: pedestrians_plot_{label_prefix.lower()}.png")

    # --- Pedestrian Wait Time plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, ped_waits, label="Avg Pedestrian Wait", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Time Steps")
    plt.title(f"Average Pedestrian Wait Time ({label_prefix})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/ped_wait_plot_{label_prefix.lower()}.png")
    print(f"[✓] Saved: ped_wait_plot_{label_prefix.lower()}.png")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_metrics.py training_metrics.csv [q|dqn]")
    else:
        csv_path = sys.argv[1]
        label = sys.argv[2].upper()
        plot_metrics(csv_path, label)

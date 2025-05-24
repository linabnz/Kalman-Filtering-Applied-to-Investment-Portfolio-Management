# report_qlearning.py
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

print("\n==================== RL Reporting Module ====================")
print(" Module : Deep Q-Learning on Co-integrated Stock Pairs")
print(" Training: 100 Episodes")
print(" Purpose : Visual summary of learning progression")
print("============================================================\n")

# Load rewards data
with open(r"C:\Users\lbenzemma\Desktop\Projets Master2 MOSEF\Kalman-Filtering-Applied-to-Investment-Portfolio-Management-1\data\rewards_by_pair.json", "r") as f:
    rewards_by_pair = json.load(f)

#  Rolling average plot
plt.figure(figsize=(12, 6))
window = 10
for pair, rewards in rewards_by_pair.items():
    rewards_series = pd.Series(rewards)
    smooth = rewards_series.rolling(window=window).mean()
    plt.plot(smooth, label=pair)

plt.title("Smoothed Reward Progression (Rolling Avg 10 episodes)", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Smoothed Reward", fontsize=12)
plt.legend(title="Pairs", loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("rolling_rewards.png")
plt.show()

#  Final reward comparison
final_rewards = {pair: rewards[-1] for pair, rewards in rewards_by_pair.items()}
plt.figure(figsize=(10, 5))
plt.bar(final_rewards.keys(), final_rewards.values(), color='lightgreen')
plt.title("Final Total Reward at Episode 100", fontsize=14)
plt.ylabel("Final Reward", fontsize=12)
plt.xlabel("Pairs", fontsize=12)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("final_rewards_bar.png")
plt.show()

#  Summary table
summary_data = []
for pair, rewards in rewards_by_pair.items():
    start = rewards[0]
    end = rewards[-1]
    gain = end - start
    summary_data.append({
        "Pair": pair,
        "Start Reward": round(start, 2),
        "End Reward": round(end, 2),
        "Total Gain": round(gain, 2)
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(by="Total Gain", ascending=False)
summary_df.to_csv("reward_summary.csv", index=False)

print(" Generated:")
print(" - rolling_rewards.png")
print(" - final_rewards_bar.png")
print(" - reward_summary.csv")
print("Use these in your report or presentation.")


import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1000
n_agents = 10
price_options = [0.1, 0.2, 0.3, 0.4]
conversion_rates = [0.9, 0.6, 0.4, 0.2]
K = len(price_options)

def get_reward(price_index):
    prob = conversion_rates[price_index]
    return price_options[price_index] if np.random.rand() < prob else 0

# UCB simulation
def simulate_ucb():
    final_counts = np.zeros(K)
    for agent in range(n_agents):
        n_k = np.zeros(K)
        r_k = np.zeros(K)
        for t in range(1, T+1):
            ucb = np.zeros(K)
            for k in range(K):
                if n_k[k] == 0:
                    ucb[k] = 1e5
                else:
                    avg = r_k[k] / n_k[k]
                    ucb[k] = avg + np.sqrt(2 * np.log(t) / n_k[k])
            choice = np.argmax(ucb)
            reward = get_reward(choice)
            n_k[choice] += 1
            r_k[choice] += reward
            final_counts[choice] += 1
    return final_counts

# EXP3 simulation
def simulate_exp3(gamma=0.3):
    final_counts = np.zeros(K)
    for agent in range(n_agents):
        weights = np.ones(K)
        for t in range(1, T+1):
            probs = (1 - gamma) * (weights / weights.sum()) + gamma / K
            choice = np.random.choice(K, p=probs)
            reward = get_reward(choice)
            estimated_reward = reward / probs[choice]
            weights[choice] *= np.exp(gamma * estimated_reward / K)
            final_counts[choice] += 1
    return final_counts

# Run simulations
ucb_counts = simulate_ucb()
exp3_counts = simulate_exp3()

# Print results
print("Price Option | UCB Selections | EXP3 Selections")
for p, u, e in zip(price_options, ucb_counts.astype(int), exp3_counts.astype(int)):
    print(f"{p:<13} {u:<15} {e}")

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
x = np.arange(len(price_options))
bar_width = 0.35
plt.bar(x - bar_width/2, ucb_counts, width=bar_width, label='UCB')
plt.bar(x + bar_width/2, exp3_counts, width=bar_width, label='EXP3')
plt.xticks(x, [f"{p:.2f}" for p in price_options])
plt.xlabel("Price Options")
plt.ylabel("Number of Selections")
plt.title("Final Arm Selection Counts (UCB vs EXP3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("final_selection_counts.png")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Settings
T = 1000
n_agents = 10
price_options = [0.1, 0.2, 0.3, 0.4]
conversion_rates = [0.9, 0.6, 0.4, 0.2]
K = len(price_options)

def get_reward(price_index):
    prob = conversion_rates[price_index]
    return price_options[price_index] if np.random.rand() < prob else 0

def simulate_curve_probabilities_extended(algorithm, gamma=0.3):
    counts_over_time = np.zeros((K, T))
    for agent in range(n_agents):
        n_k = np.zeros(K)
        r_k = np.zeros(K)
        r2_k = np.zeros(K)
        weights = np.ones(K)

        for t in range(1, T+1):
            if algorithm == "EXP3":
                probs = (1 - gamma) * (weights / weights.sum()) + gamma / K
                choice = np.random.choice(K, p=probs)
                reward = get_reward(choice)
                estimated_reward = reward / probs[choice]
                weights[choice] *= np.exp(gamma * estimated_reward / K)
            else:
                ucb = np.zeros(K)
                for k in range(K):
                    if n_k[k] == 0:
                        ucb[k] = 1e5
                    else:
                        avg = r_k[k] / n_k[k]
                        if algorithm == "UCB":
                            ucb[k] = avg + np.sqrt(2 * np.log(t) / n_k[k])
                        elif algorithm == "UCB-V":
                            var = (r2_k[k] / n_k[k]) - (avg ** 2)
                            ucb[k] = avg + 3 * np.log(t) / n_k[k] + np.sqrt(2 * np.log(t) * var / n_k[k])
                        elif algorithm == "UCB-Tuned":
                            var = (r2_k[k] / n_k[k]) - (avg ** 2)
                            var = min(1/4, var)
                            ucb[k] = avg + np.sqrt(np.log(t) * var / n_k[k]) + np.sqrt(2 * np.log(t) / n_k[k])
                choice = np.argmax(ucb)
                reward = get_reward(choice)
                n_k[choice] += 1
                r_k[choice] += reward
                r2_k[choice] += reward ** 2
            counts_over_time[choice, t-1] += 1
    return counts_over_time / n_agents

# Run all simulations
ucb_p = np.cumsum(simulate_curve_probabilities_extended("UCB"), axis=1) / (np.arange(1, T+1))
exp3_p = np.cumsum(simulate_curve_probabilities_extended("EXP3"), axis=1) / (np.arange(1, T+1))
ucbv_p = np.cumsum(simulate_curve_probabilities_extended("UCB-V"), axis=1) / (np.arange(1, T+1))
ucbt_p = np.cumsum(simulate_curve_probabilities_extended("UCB-Tuned"), axis=1) / (np.arange(1, T+1))

# Plot
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
algos4 = [
    ("Panel A: EXP3", exp3_p),
    ("Panel B: UCB", ucb_p),
    ("Panel C: UCB-V", ucbv_p),
    ("Panel D: UCB-Tuned", ucbt_p)
]
colors = ['blue', 'orange', 'green', 'red']

for ax, (title, probs) in zip(axs.ravel(), algos4):
    for i in range(K):
        ax.plot(probs[i], label=f'arm {i+1} (s = {price_options[i]:.2f})', color=colors[i])
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time / arm draw")
    ax.set_ylabel("probability of arm draw")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)

plt.suptitle("Speed of Convergence to Optimal Arm Draw – All Algorithms", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("panel_all_algorithms_convergence.png")
plt.show()

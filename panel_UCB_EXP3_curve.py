
import numpy as np
import matplotlib.pyplot as plt

# Simulation settings
T = 1000
n_agents = 10
price_options = [0.1, 0.2, 0.3, 0.4]
conversion_rates = [0.9, 0.6, 0.4, 0.2]
K = len(price_options)

def get_reward(price_index):
    prob = conversion_rates[price_index]
    return price_options[price_index] if np.random.rand() < prob else 0

# Run simulation and log selection probabilities
def simulate_curve_probabilities(algorithm, gamma=0.3):
    counts_over_time = np.zeros((K, T))
    for agent in range(n_agents):
        if algorithm == "EXP3":
            weights = np.ones(K)
            for t in range(1, T+1):
                probs = (1 - gamma) * (weights / weights.sum()) + gamma / K
                choice = np.random.choice(K, p=probs)
                reward = get_reward(choice)
                estimated_reward = reward / probs[choice]
                weights[choice] *= np.exp(gamma * estimated_reward / K)
                counts_over_time[choice, t-1] += 1
        elif algorithm == "UCB":
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
                counts_over_time[choice, t-1] += 1
    return counts_over_time / n_agents

# Run simulations
ucb_prob_curve = simulate_curve_probabilities("UCB")
exp3_prob_curve = simulate_curve_probabilities("EXP3", gamma=0.3)

# Convert to cumulative probability over time
ucb_prob = np.cumsum(ucb_prob_curve, axis=1) / (np.arange(1, T+1))
exp3_prob = np.cumsum(exp3_prob_curve, axis=1) / (np.arange(1, T+1))

# Plot result
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
algos = [("UCB algorithm", ucb_prob), ("EXP3 algorithm", exp3_prob)]
colors = ['blue', 'orange', 'green', 'red']

for ax, (title, probs) in zip(axs, algos):
    for i in range(K):
        ax.plot(probs[i], label=f'arm {i+1} (s = {price_options[i]:.2f})', color=colors[i])
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time / arm draw")
    ax.set_ylabel("probability of arm draw")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)

plt.suptitle("Speed of Convergence to Optimal Arm Draw (UCB vs EXP3)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("panel_UCB_EXP3_curve.png")
plt.show()

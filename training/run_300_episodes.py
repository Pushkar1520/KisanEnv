import os
import sys
import json
import time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from env import KisanEnv
from agents.farmer_agent import FarmerAgent, CHECKPOINT_DIR
from episode_tracker import save_episode, REWARDS_FILE
def clear_previous_data():
    if os.path.exists(REWARDS_FILE):
        os.remove(REWARDS_FILE)
        print(f"[CLEANUP] Removed {REWARDS_FILE}")
    agent_path = os.path.join(CHECKPOINT_DIR, "farmer_agent.json")
    if os.path.exists(agent_path):
        os.remove(agent_path)
        print(f"[CLEANUP] Removed {agent_path}")
    soil_path = os.path.join(PROJECT_ROOT, ".soil_state.json")
    with open(soil_path, "w") as f:
        json.dump({"soil_health": 0.7}, f)
    print(f"[CLEANUP] Reset soil_health to 0.7")
def run_training(num_episodes=300):
    print(f"\n{'='*60}")
    print(f"  KisanEnv 2.0 -- Phase 1: Q-Learning ({num_episodes} episodes)")
    print(f"{'='*60}\n")
    env = KisanEnv()
    env._is_baseline = True
    agent = FarmerAgent()
    print(f"[INIT] FarmerAgent(alpha={agent.alpha}, gamma={agent.gamma}, eps={agent.epsilon})")
    print(f"[INIT] Q-table: EMPTY ({len(agent.q_table)} states)")
    print(f"[INIT] Reasoning: DISABLED (reasoning_quality = 0)")
    print(f"[INIT] Contrastive bonus: DISABLED")
    print(f"[INIT] Epsilon decay: 0.993 per episode\n")
    all_rewards = []
    start_time = time.time()
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        while not done:
            farm_state = obs.get("farm_state", {})
            farm_state["day"] = obs.get("day", farm_state.get("day", 1))
            action_name, _display_reasoning = agent.select_action(farm_state)
            action_text = f"ACTION: {action_name}\nREASONING: "
            obs, reward, done, info = env.step(action_text)
            next_farm_state = obs.get("farm_state", {})
            next_farm_state["day"] = obs.get("day", next_farm_state.get("day", 1))
            agent.update(reward, next_farm_state, done)
        ep_reward = info.get("episode_reward", 0.0)
        agent.end_episode(ep_reward)
        all_rewards.append(ep_reward)
        save_episode(ep, ep_reward, env.climate_agent.current_difficulty)
        if ep <= 5 or ep % 50 == 0:
            bd = info.get("reward_breakdown", {})
            parts = []
            for k in ["final_profit", "crop_yield", "soil_preservation",
                       "decision_quality", "reasoning_quality", "resource_efficiency",
                       "insurance_usage"]:
                if k in bd:
                    parts.append(f"{k[:6]}={bd[k]['contribution']:.3f}")
            print(f"  Ep {ep:>3d} | R={ep_reward:.4f} | {' '.join(parts)}")
        if ep % 10 == 0:
            avg10 = sum(all_rewards[-10:]) / min(len(all_rewards), 10)
            above = sum(1 for r in all_rewards if r > 0.44)
            elapsed = time.time() - start_time
            print(
                f"  [{ep:>3d}/{num_episodes}] "
                f"Avg(10)={avg10:.4f} | "
                f"Best={max(all_rewards):.4f} | "
                f"Above 0.44: {above}/{ep} | "
                f"eps={agent.epsilon:.3f} | "
                f"Diff={env.climate_agent.current_difficulty} | "
                f"{ep/elapsed:.1f} ep/s"
            )
    elapsed = time.time() - start_time
    avg_first10 = sum(all_rewards[:10]) / 10
    avg_last10 = sum(all_rewards[-10:]) / 10
    print(f"\n{'='*60}")
    print(f"  Training Complete -- {num_episodes} episodes in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Best Reward:        {max(all_rewards):.4f}")
    print(f"  Worst Reward:       {min(all_rewards):.4f}")
    print(f"  Average Reward:     {sum(all_rewards)/len(all_rewards):.4f}")
    print(f"  Avg First 10:       {avg_first10:.4f}")
    print(f"  Avg Last 10:        {avg_last10:.4f}")
    print(f"  Improvement:        {avg_last10 - avg_first10:+.4f}")
    print(f"  Above 0.44:         {sum(1 for r in all_rewards if r > 0.44)}/{num_episodes}")
    print(f"  Final Epsilon:      {agent.epsilon:.4f}")
    print(f"  Final Difficulty:   {env.climate_agent.current_difficulty}")
    print(f"  Q-table states:     {len(agent.q_table)}")
    print(f"{'='*60}\n")
    agent.save()
    print(f"[SAVED] Agent Q-table -> checkpoints/farmer_agent.json")
    print(f"[SAVED] Episode rewards -> episode_rewards.json")
    return all_rewards
def plot_results():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot")
        return
    if not os.path.exists(REWARDS_FILE):
        return
    with open(REWARDS_FILE, "r") as f:
        data = json.load(f)
    episodes = data.get("episodes", [])
    if len(episodes) < 3:
        return
    rewards = [float(ep["reward"]) for ep in episodes]
    difficulties = [int(ep.get("difficulty", 1)) for ep in episodes]
    x = np.arange(1, len(rewards) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={'width_ratios': [2.5, 1]})
    fig.patch.set_facecolor('#0a0e14')
    fig.suptitle('KisanEnv 2.0 -- Phase 1: Q-Learning Training',
                 color='#d4a843', fontsize=16, fontweight='bold', y=0.98)
    ax1.set_facecolor('#0a0e14')
    ax1.plot(x, rewards, color='#5ba3e0', alpha=0.15, linewidth=0.8, label='Raw Reward')
    window = min(15, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    smooth_x = np.arange(window, len(rewards) + 1)
    ax1.plot(smooth_x, smoothed, color='#5ba3e0', linewidth=2.5,
             label=f'Smoothed (w={window})')
    ax1.axhline(y=0.44, color='#e84040', linestyle='--', linewidth=1.5,
                label='Baseline (0.44) -- Phase 2 target')
    for i in range(1, len(difficulties)):
        if difficulties[i] > difficulties[i-1]:
            ax1.axvline(x=i+1, color='#d4a843', linestyle=':', alpha=0.7)
            ax1.text(i+2, max(rewards)*0.95, f'Lvl {difficulties[i]}',
                     color='#d4a843', fontsize=8, rotation=90, va='top')
    ax1.set_xlabel('Episode', color='#a0b0c0', fontsize=10)
    ax1.set_ylabel('Episode Reward', color='#a0b0c0', fontsize=10)
    ax1.set_title('Q-Learning Progress (No Reasoning)', color='#d4a843', fontsize=13)
    ax1.tick_params(colors='#6a7a8a')
    ax1.spines['bottom'].set_color('#2a3a4a')
    ax1.spines['left'].set_color('#2a3a4a')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper left', facecolor='#0a0e14', edgecolor='#2a3a4a',
               labelcolor='#a0b0c0', fontsize=9)
    ax1.set_ylim(0, max(0.6, max(rewards) + 0.05))
    ax2.set_facecolor('#0a0e14')
    avg_first = sum(rewards[:10]) / min(10, len(rewards))
    avg_mid = sum(rewards[140:160]) / min(20, max(1, len(rewards[140:160])))
    avg_last = sum(rewards[-10:]) / min(10, len(rewards))
    best = max(rewards)
    baseline = 0.44
    labels = ['First 10\nAvg', 'Mid\nAvg', 'Last 10\nAvg', 'Best', 'Baseline\n(Phase 2)']
    values = [avg_first, avg_mid, avg_last, best, baseline]
    colors = ['#c84b31', '#d4a843', '#30b09a', '#5ba3e0', '#e84040']
    bars = ax2.bar(labels, values, color=colors, edgecolor='none', width=0.6)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', color='#e0d8c8',
                 fontsize=9, fontweight='bold')
    ax2.set_title('Learning Progress', color='#d4a843', fontsize=13)
    ax2.tick_params(colors='#6a7a8a')
    ax2.spines['bottom'].set_color('#2a3a4a')
    ax2.spines['left'].set_color('#2a3a4a')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "reward_curve.png")
    plt.savefig(output_path, dpi=150, facecolor='#0a0e14', bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved -> results/reward_curve.png")
if __name__ == "__main__":
    clear_previous_data()
    rewards = run_training(num_episodes=500)
    plot_results()
    print("\n[DONE] Phase 1 complete. Ready for Phase 2 (Qwen GRPO on Colab).")

import numpy as np
import time
import matplotlib.pyplot as plt

def plot_rewards(episode_rewards, window=100, filename="training_progress.png"):
    """Plot the rewards over episodes with a moving average."""
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.6, label='Rewards')
    
    # Compute and plot moving average if enough episodes
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, label=f'{window}-episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def print_training_stats(episode, total_reward, epsilon, duration, steps=None):
    """Print training statistics in a concise format."""
    steps_info = f", {steps} steps" if steps is not None else ""
    print(f"Episode {episode}: Reward={total_reward:.1f}, Epsilon={epsilon:.4f}, Time={duration:.1f}s{steps_info}")

def log_episode_results(log_file, episode, total_reward, env_reward, pos_reward, 
                        epsilon, duration, steps, scores_made, scores_against):
    """Log episode results to the specified file."""
    import os
    
    # Create header if file doesn't exist yet
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("episode,total_reward,env_reward,pos_reward,epsilon,duration,steps,scores_made,scores_against\n")
    
    # Create a log entry as a comma-separated line
    log_entry = f"{episode},{total_reward:.6f},{env_reward:.6f},{pos_reward:.6f},"
    log_entry += f"{epsilon:.6f},{duration:.6f},{steps},{scores_made},{scores_against}\n"
    
    # Append to the log file
    with open(log_file, 'a') as f:
        f.write(log_entry)
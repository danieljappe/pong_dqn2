import time
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import gc
from environment import PongEnvironment
from agent import DQNAgent
from utils import plot_rewards, print_training_stats, log_episode_results

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configuration
checkpoint_path = "models/long_run_final.h5"  # Your final model from previous run
start_episode = 500
additional_episodes = 500  # How many more episodes to train
run_name = "long_runv2"  # New name to track improved version

# Create environment and agent
env = PongEnvironment()
state_shape = (84, 84, 4)
action_size = env.action_space.n

# Create agent with improved learning parameters
agent = DQNAgent(
    state_shape=state_shape,
    action_size=action_size,
    learning_rate=0.00015,  # Increased from 0.0001
    gamma=0.99,
    epsilon=0.368,  # Use the last epsilon value from your output
    epsilon_min=0.05,
    epsilon_decay=0.997,  # Slightly faster decay
    buffer_capacity=100000,
    batch_size=32
)

# Load model weights
print(f"Loading weights from {checkpoint_path}")
agent.model.load_weights(checkpoint_path)
agent.target_model.load_weights(checkpoint_path)

# Load previous rewards if available
try:
    # Try to read the CSV file to extract rewards
    import pandas as pd
    df = pd.read_csv("logs/long_run.csv")  # Original log file
    episode_rewards = df['total_reward'].tolist()
    print(f"Loaded {len(episode_rewards)} previous reward values")
except Exception as e:
    print(f"Error loading previous rewards: {e}")
    episode_rewards = []
    print("Starting with empty rewards list")

# Set up other training variables
scores_made = 1652  # From your output
scores_against = 10500  # From your output

print(f"Continuing training from episode {start_episode} for {additional_episodes} more episodes")
print(f"Using improved learning parameters for faster learning")

# Main training loop (similar to your original train_dqn function)
# Create directories if needed
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Get log file - new file for improved version
log_file = f"logs/{run_name}.csv"

# Training loop
try:
    for episode in range(start_episode, start_episode + additional_episodes):
        # Reset environment
        observation = env.reset()
        state = env.preprocess_observation(observation)
        
        # Track episode statistics
        total_reward = 0
        env_reward = 0
        pos_reward = 0
        steps = 0
        start_time = time.time()
        episode_scores_made = 0
        episode_scores_against = 0
        
        # Episode loop
        for step in range(10000):  # max steps
            # Select action
            action = agent.select_action(state)
            
            # Take action with frame skipping
            next_observation, reward, done, info = env.step_with_skip(action, skip=4)
            
            # Track reward components
            total_reward += reward
            if 'env_reward' in info:
                env_reward += info['env_reward']
                
                # Track scores
                if info['env_reward'] == 1:
                    scores_made += 1
                    episode_scores_made += 1
                    # Enhanced reward for scoring
                    bonus = 3.0  # Increased from 2.0
                    print(f"  🏆 Agent scored! (+{1.0 + bonus:.1f})")
                elif info['env_reward'] == -1:
                    scores_against += 1
                    episode_scores_against += 1
                    print(f"  ❌ Opponent scored")
                    
            if 'pos_reward' in info:
                pos_reward += info['pos_reward']
            
            # Preprocess next state
            next_state = env.preprocess_observation(next_observation)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train on every step
            agent.train()
            
            # Update state and statistics
            state = next_state
            steps += 1
            
            # End episode if done
            if done:
                break
        
        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Calculate episode duration
        duration = time.time() - start_time
        steps_per_second = steps / duration if duration > 0 else 0
        
        # Store episode rewards
        episode_rewards.append(total_reward)
        
        # Log episode results
        log_episode_results(
            log_file, episode, total_reward, env_reward, pos_reward,
            agent.epsilon, duration, steps, scores_made, scores_against
        )
        
        # Monitor system resources
        try:
            import psutil
            process = psutil.Process(os.getpid())
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_used_gb = memory_info.used / (1024 ** 3)
            memory_total_gb = memory_info.total / (1024 ** 3)
            system_info = f" | CPU: {cpu_percent}% | RAM: {memory_used_gb:.1f}/{memory_total_gb:.1f} GB ({memory_info.percent}%)"
        except:
            system_info = ""
        
        # Print stats
        print(f"Episode {episode} | Total: {total_reward:.1f} | Game Score: {episode_scores_made}-{episode_scores_against} (Total: {scores_made}-{scores_against}){system_info}")
        print(f"  └─ Env Reward: {env_reward:.1f} | Pos Reward: {pos_reward:.1f} | ε: {agent.epsilon:.3f} | Time: {duration:.1f}s | Steps: {steps} ({steps_per_second:.1f}/s)")
        
        # Plot rewards periodically
        if episode % 10 == 0:
            # Create a plot that shows both original and continued training
            plt.figure(figsize=(12, 6))
            plt.plot(episode_rewards, alpha=0.6, label='Rewards')
            
            # Calculate moving average if we have enough episodes
            if len(episode_rewards) >= 100:
                moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
                plt.plot(range(99, len(episode_rewards)), moving_avg, 'r-', label='100-episode Moving Average')
            
            # Add a vertical line at the continuation point
            plt.axvline(x=start_episode-1, color='g', linestyle='--', alpha=0.7, label='Continuation Point')
            
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title(f'Training Progress - Episode {episode}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"models/{run_name}_progress.png")
            plt.close()
            
            print(f"  📊 Training progress saved to models/{run_name}_progress.png")
        
        # Save model when we achieve a new best reward
        if total_reward > max(episode_rewards[:-1], default=-21):
            agent.model.save(f"models/{run_name}_best.h5")
            print(f"  💾 New best reward: {total_reward:.1f}! Model saved to models/{run_name}_best.h5")
            
        # Save model periodically
        if episode % 50 == 0:
            agent.model.save(f"models/{run_name}_episode_{episode}.h5")
            print(f"  💾 Checkpoint saved at episode {episode}")
            
        # Memory cleanup
        tf.keras.backend.clear_session()
        gc.collect()
        
        print()  # Blank line for readability
    
    # Save final model
    agent.model.save(f"models/{run_name}_final.h5")
    print(f"Training complete. Final model saved to models/{run_name}_final.h5")
    print(f"Results saved to {log_file}")
    
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving model...")
    agent.model.save(f"models/{run_name}_interrupted.h5")
    print(f"Model saved to models/{run_name}_interrupted.h5")
    
finally:
    env.close()
import time
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import gc  # Add garbage collector
from environment import PongEnvironment
from agent import DQNAgent
from utils import plot_rewards, print_training_stats, log_episode_results
import multiprocessing

# Force CPU only - more reliable configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Forced CPU mode for reliable training")

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configure CPU threading for better performance
NUM_CPU = multiprocessing.cpu_count()
print(f"Number of CPU cores: {NUM_CPU}")
tf.config.threading.set_inter_op_parallelism_threads(NUM_CPU // 2)
tf.config.threading.set_intra_op_parallelism_threads(NUM_CPU // 2)
print(f"TensorFlow thread settings: inter_op={NUM_CPU // 2}, intra_op={NUM_CPU // 2}")

def monitor_system_resources():
    """Monitor system resources (CPU usage and memory)"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_used_gb = memory_info.used / (1024 ** 3)
        memory_total_gb = memory_info.total / (1024 ** 3)
        return f"CPU: {cpu_percent}% | RAM: {memory_used_gb:.1f}/{memory_total_gb:.1f} GB ({memory_info.percent}%)"
    except ImportError:
        return None  # psutil not available

def train_dqn(render_mode=None, frame_skip=4, episodes=500, report_interval=10, run_name="default_run"):
    """Train a DQN agent to play Pong."""
    print(f"Starting DQN training run: {run_name}")
    
    # Create log and model directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create log file path
    log_file = f"logs/{run_name}.csv"
    
    # Create the Pong environment
    env = PongEnvironment(render_mode=render_mode)
    
    # Define parameters with frame stacking
    state_shape = (84, 84, 4)  # 4 stacked frames
    action_size = env.action_space.n  # Should be 3 now (UP, DOWN, NOOP)
    
    # Create the DQN agent with improved learning parameters
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        learning_rate=0.0001,       # Reduced for more stable learning
        gamma=0.99,                 # Unchanged discount factor
        epsilon=1.0,                # Start with full exploration
        epsilon_min=0.05,           # Lower minimum epsilon for better exploitation
        epsilon_decay=0.998,        # Slower decay for more exploration
        buffer_capacity=100000,     # Increased buffer size
        batch_size=32               # Increased batch size for better learning
    )
    
    # Training parameters
    max_steps_per_episode = 10000
    
    # Track statistics
    episode_rewards = []
    best_reward = -21
    scores_made = 0
    scores_against = 0
    
    # Track training speed
    total_steps = 0
    total_duration = 0
    
    # Create a plot to track progress
    plt.figure(figsize=(12, 6), num='DQN Training Progress')
    
    try:
        # Training loop
        for episode in range(episodes):
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
            
            # Training step frequency (train every step for better learning)
            train_frequency = 1  # Train on every step
            
            # Episode loop
            for step in range(max_steps_per_episode):
                # Select action
                action = agent.select_action(state)
                
                # Take action with frame skipping
                next_observation, reward, done, info = env.step_with_skip(action, skip=frame_skip)
                
                # Track reward components
                total_reward += reward
                if 'env_reward' in info:
                    env_reward += info['env_reward']
                    
                    # Track scores
                    if info['env_reward'] == 1:
                        scores_made += 1
                        episode_scores_made += 1
                    elif info['env_reward'] == -1:
                        scores_against += 1
                        episode_scores_against += 1
                        
                if 'pos_reward' in info:
                    pos_reward += info['pos_reward']
                
                # Preprocess next state
                next_state = env.preprocess_observation(next_observation)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train the agent every N steps to reduce CPU load
                if step % train_frequency == 0:
                    agent.train()
                
                # Update state and statistics
                state = next_state
                steps += 1
                
                # End episode if done
                if done:
                    break
            
            # Decay epsilon once per episode - much more stable than decaying every step
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            
            # Calculate episode duration and update totals
            duration = time.time() - start_time
            total_steps += steps
            total_duration += duration
            steps_per_second = steps / duration if duration > 0 else 0
            
            # Store episode rewards
            episode_rewards.append(total_reward)
            
            # Log episode results
            log_episode_results(
                log_file, episode, total_reward, env_reward, pos_reward,
                agent.epsilon, duration, steps, scores_made, scores_against
            )
            
            # Get system resource stats
            system_stats = monitor_system_resources()
            system_info = f" | {system_stats}" if system_stats else ""
            
            # Print episode statistics with detailed breakdown
            print(f"Episode {episode} | Total: {total_reward:.1f} | Game Score: {episode_scores_made}-{episode_scores_against} (Total: {scores_made}-{scores_against})")
            print(f"  └─ Env Reward: {env_reward:.1f} | Pos Reward: {pos_reward:.1f} | ε: {agent.epsilon:.3f} | Time: {duration:.1f}s | Steps: {steps} ({steps_per_second:.1f}/s){system_info}")
            
            # Plot rewards periodically
            if episode % report_interval == 0 and episode > 0:
                # Save regular static plot
                plot_rewards(episode_rewards, filename=f"models/{run_name}_progress.png")
                print(f"  📊 Training progress saved to models/{run_name}_progress.png")
                
                # Update interactive plot if running in a notebook or interactive shell
                try:
                    plt.clf()
                    plt.plot(episode_rewards, 'b-', alpha=0.6, label='Rewards')
                    if len(episode_rewards) >= 100:
                        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
                        plt.plot(range(99, len(episode_rewards)), moving_avg, 'r-', label='100-episode Moving Average')
                    plt.xlabel('Episode')
                    plt.ylabel('Total Reward')
                    plt.title(f'Training Progress - Episode {episode}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.pause(0.01)
                except Exception:
                    pass
                
            # Save model when we achieve a new best reward
            if total_reward > best_reward:
                best_reward = total_reward
                agent.model.save(f"models/{run_name}_best.h5")
                print(f"  💾 New best reward: {best_reward:.1f}! Model saved to models/{run_name}_best.h5")
                
            # Save model periodically
            if episode % 50 == 0 and episode > 0:
                agent.model.save(f"models/{run_name}_episode_{episode}.h5")
                print(f"  💾 Checkpoint saved at episode {episode}")
                
                # Performance stats
                avg_steps_per_second = total_steps / total_duration if total_duration > 0 else 0
                print(f"  ⏱️ Performance: {avg_steps_per_second:.1f} steps/second average")
                
            # Clean up memory more aggressively (after every episode)
            tf.keras.backend.clear_session()
            import gc
            gc.collect()  # Force garbage collection
                
            # Print a blank line for readability
            print()
        
        # Close environment
        env.close()
        
        # Save final model
        agent.model.save(f"models/{run_name}_final.h5")
        print(f"Training complete. Final model saved to models/{run_name}_final.h5")
        print(f"Results saved to {log_file}")
        # Save rewards history to a text file
        np.savetxt(f"logs/{run_name}_rewards.txt", episode_rewards)
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted. Saving model...")
        agent.model.save(f"models/{run_name}_interrupted.h5")
        print(f"Model saved to models/{run_name}_interrupted.h5")
        print(f"Results saved to {log_file}")
        env.close()
    except Exception as e:
        print(f"Error during training: {e}")
        env.close()
    finally:
        plt.close()

def test_random_agent(render_mode="human", steps=1000):
    """Run a random agent to test the environment."""
    print("Testing environment with random actions...")
    
    # Create the Pong environment
    env = PongEnvironment(render_mode=render_mode)
    
    # Get initial observation
    observation = env.reset()
    
    total_reward = 0
    
    # Run some random actions to see the game in action
    for step in range(steps):
        # Take a random action
        action = env.action_space.sample()
        
        # Step the environment
        next_observation, reward, done, info = env.step(action)
        
        total_reward += reward
        
        # Print reward if non-zero (scoring)
        if reward != 0:
            print(f"Step {step}: Reward {reward}, Total: {total_reward}")
        
        # Check if episode is done
        if done:
            print(f"Episode ended after {step+1} steps with total reward {total_reward}")
            observation = env.reset()
            total_reward = 0
    
    # Close the environment
    env.close()

def continue_training(checkpoint_path, render_mode=None, frame_skip=4, start_episode=0, 
                     episodes=500, report_interval=10, run_name="continued_run"):
    """Continue training from a saved checkpoint."""
    print(f"Continuing training from checkpoint: {checkpoint_path}")
    
    # Create log and model directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create log file path
    log_file = f"logs/{run_name}.csv"
    
    # Create the Pong environment
    env = PongEnvironment(render_mode=render_mode)
    
    # Define parameters with frame stacking
    state_shape = (84, 84, 4)
    action_size = env.action_space.n
    
    # Create a temporary agent to get the structure
    temp_agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=0.5,  # Will be overridden
        epsilon_min=0.05,
        epsilon_decay=0.998,
        buffer_capacity=100000,
        batch_size=32
    )
    
    # Load the saved model weights
    temp_agent.model.load_weights(checkpoint_path)
    temp_agent.target_model.load_weights(checkpoint_path)
    
    # Now create the real agent with the loaded weights
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=0.2,  # Set epsilon based on training progress
        epsilon_min=0.05,
        epsilon_decay=0.998,
        buffer_capacity=100000,
        batch_size=32
    )
    
    agent.model.set_weights(temp_agent.model.get_weights())
    agent.target_model.set_weights(temp_agent.model.get_weights())
    del temp_agent
    
    # Load previous rewards if available
    try:
        episode_rewards = list(np.loadtxt(f"logs/{run_name}_rewards.txt"))
        print(f"Loaded {len(episode_rewards)} previous reward values")
    except:
        episode_rewards = []
        print("No previous rewards found, starting fresh reward tracking")

if __name__ == "__main__":
    # Configuration options
    mode = "train"           # Options: "train", "continue", "test", "random" 
    checkpoint_path = None   # Path to checkpoint file when using "continue" mode
    start_episode = 0        # Episode to start from when continuing
    render_enable = False    # Set to True to visualize, False for faster training
    frame_skip = 4           # Number of frames to skip per action
    run_name = "cpu_dqn"     # Name for this training run (used for logging)
    
    # Set render mode based on configuration
    render_mode = "human" if render_enable else None
    
    # [rest of your existing code]
    
    if mode == "train":
        train_dqn(render_mode=render_mode, frame_skip=frame_skip, run_name=run_name)
    elif mode == "continue" and checkpoint_path:
        continue_training(checkpoint_path, render_mode=render_mode, frame_skip=frame_skip, 
                         start_episode=start_episode, run_name=run_name)
    elif mode == "random":
        test_random_agent(render_mode=render_mode)
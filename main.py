import time
import numpy as np
import tensorflow as tf
import os
from environment import PongEnvironment
from agent import DQNAgent
from utils import plot_rewards, print_training_stats, log_episode_results

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train_dqn(render_mode=None, frame_skip=4, episodes=500, report_interval=10, run_name="default_run"):
    """Train a DQN agent to play Pong."""
    print(f"Starting DQN training run: {run_name}")
    
    # Create log file path
    log_file = f"logs/{run_name}.csv"
    
    # Create the Pong environment
    env = PongEnvironment(render_mode=render_mode)
    
    # Define parameters
    state_shape = (84, 84, 1)
    action_size = env.action_space.n
    
    # Create the DQN agent
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.9995,
        buffer_capacity=10000,
        batch_size=32
    )
    
    # Training parameters
    max_steps_per_episode = 5000
    
    # Track statistics
    episode_rewards = []
    best_reward = -21
    scores_made = 0
    scores_against = 0
    
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
                
                # Train the agent (every 4 steps for efficiency)
                if step % 4 == 0:
                    agent.train()
                
                # Update state and statistics
                state = next_state
                steps += 1
                
                # End episode if done
                if done:
                    break
            
            # Calculate episode duration
            duration = time.time() - start_time
            
            # Store episode rewards
            episode_rewards.append(total_reward)
            
            # Log episode results
            log_episode_results(
                log_file, episode, total_reward, env_reward, pos_reward,
                agent.epsilon, duration, steps, scores_made, scores_against
            )
            
            # Print episode statistics with detailed breakdown
            print(f"Episode {episode} | Total: {total_reward:.1f} | Game Score: {episode_scores_made}-{episode_scores_against} (Total: {scores_made}-{scores_against})")
            print(f"  └─ Env Reward: {env_reward:.1f} | Pos Reward: {pos_reward:.1f} | ε: {agent.epsilon:.3f} | Time: {duration:.1f}s | Steps: {steps}")
            
            # Plot rewards periodically
            if episode % report_interval == 0 and episode > 0:
                plot_rewards(episode_rewards, filename=f"models/{run_name}_progress.png")
                print(f"  📊 Training progress saved to models/{run_name}_progress.png")
                
            # Save model when we achieve a new best reward
            if total_reward > best_reward:
                best_reward = total_reward
                agent.model.save(f"models/{run_name}_best.h5")
                print(f"  💾 New best reward: {best_reward:.1f}! Model saved to models/{run_name}_best.h5")
                
            # Save model periodically
            if episode % 50 == 0 and episode > 0:
                agent.model.save(f"models/{run_name}_episode_{episode}.h5")
                print(f"  💾 Checkpoint saved at episode {episode}")
                
            # Print a blank line for readability
            print()
        
        # Close environment
        env.close()
        
        # Save final model
        agent.model.save(f"models/{run_name}_final.h5")
        print(f"Training complete. Final model saved to models/{run_name}_final.h5")
        print(f"Results saved to {log_file}")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted. Saving model...")
        agent.model.save(f"models/{run_name}_interrupted.h5")
        print(f"Model saved to models/{run_name}_interrupted.h5")
        print(f"Results saved to {log_file}")
        env.close()
    except Exception as e:
        print(f"Error during training: {e}")
        env.close()

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

if __name__ == "__main__":
    # Configuration options
    mode = "train"           # Options: "train", "random" 
    render_enable = False    # Set to True to visualize, False for faster training
    frame_skip = 4           # Number of frames to skip per action
    run_name = "run1"        # Name for this training run (used for logging)
    
    # Set render mode based on configuration
    render_mode = "human" if render_enable else None
    
    if mode == "train":
        train_dqn(render_mode=render_mode, frame_skip=frame_skip, run_name=run_name)
    elif mode == "random":
        test_random_agent(render_mode=render_mode)
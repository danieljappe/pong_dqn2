import gym
import numpy as np
import cv2
from collections import deque

class PongEnvironment:
    def __init__(self, render_mode=None):
        # Create the Pong environment
        if render_mode is None:
            self.env = gym.make("PongNoFrameskip-v4")
        else:
            self.env = gym.make("PongNoFrameskip-v4", render_mode=render_mode)
            
        # Map Pong's 6 actions to the 3 that matter: NOOP (0), UP (2), DOWN (3)
        self.action_mapping = [0, 2, 3]  
        self.action_space = gym.spaces.Discrete(len(self.action_mapping))
        self.observation_space = self.env.observation_space
        
        # Track previous observations and paddle position
        self.previous_observation = None
        self.paddle_y = None  # Track the agent's paddle position
        self.debug_mode = False  # Set to False for less output
        self.steps_since_print = 0
        
        # Initialize frame stack
        self.frame_stack = None
        
        # Print environment info only once
        print(f"Environment created. Action space: {self.action_space}, Observation space: {self.observation_space.shape}")

    def reset(self):
        """Reset the environment and return the initial observation."""
        observation = self.env.reset()
        
        # Handle different gym versions that might return (obs, info) or just obs
        if isinstance(observation, tuple) and len(observation) == 2:
            observation, _ = observation
        
        # Reset tracking variables
        self.previous_observation = None
        self.paddle_y = None
        self.steps_since_print = 0
        self.frame_stack = None
        
        return observation
    
    def step(self, action):
        """Take a step in the environment with the mapped action."""
        # Map the action
        env_action = self.action_mapping[action]
        
        # Take action
        result = self.env.step(env_action)
        
        # Handle different gym versions that might return different formats
        if len(result) == 5:  # gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        else:  # gym format
            return result
    
    def detect_paddle_and_ball(self, observation):
        """
        Detect the positions of the paddle and ball in the observation.
        Returns:
            paddle_y: Y-coordinate of the center of the paddle
            ball_x, ball_y: Coordinates of the ball
            ball_detected: Whether the ball was detected
        """
        if observation is None:
            return None, None, None, False
        
        # Convert to grayscale for simpler processing
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        
        # Find all bright pixels (ball and paddles are white)
        white_pixels = np.where(gray > 200)
        
        if len(white_pixels[0]) == 0:  # No white pixels found
            return None, None, None, False
        
        # Separate right side (our paddle)
        right_side_x = 140  # Approximate x-coordinate dividing the court
        right_indices = np.where(white_pixels[1] >= right_side_x)[0]
        
        # Find our paddle
        if len(right_indices) > 0:
            paddle_ys = white_pixels[0][right_indices]
            paddle_y = int(np.mean(paddle_ys))  # Paddle center point
        else:
            paddle_y = None
        
        # Find the ball (in the middle area, away from paddles)
        left_paddle_x = 20  # Approximate x-coordinate of opponent's paddle
        middle_indices = np.where((white_pixels[1] > left_paddle_x) & 
                                  (white_pixels[1] < right_side_x))[0]
        
        if len(middle_indices) > 0:
            ball_ys = white_pixels[0][middle_indices]
            ball_xs = white_pixels[1][middle_indices]
            
            # Ball is the centroid of white pixels in the middle
            ball_y = int(np.mean(ball_ys))
            ball_x = int(np.mean(ball_xs))
            ball_detected = True
        else:
            ball_x, ball_y = None, None
            ball_detected = False
        
        # Debug information
        if self.debug_mode and paddle_y is not None and ball_detected:
            print(f"  DEBUG: Paddle at y={paddle_y}, Ball at x={ball_x}, y={ball_y}")
        
        return paddle_y, ball_x, ball_y, ball_detected
    
    def step_with_skip(self, action, skip=4):
        """
        Take action and repeat it for several frames, adding positioning rewards.
        """
        total_reward = 0
        env_reward = 0  # Track environmental reward separately
        pos_reward = 0  # Track positioning reward separately
        done = False
        info = None
        next_observation = None
        
        # Tracking alignment for stats
        alignment_count = 0
        total_alignments = 0
        
        for i in range(skip):
            # Take step
            next_observation, reward, done, info = self.step(action)
            
            # Track environmental reward separately
            env_reward += reward
            total_reward += reward
            
            # If the agent scored, add a significant bonus
            if reward == 1:
                bonus = 0.5
                total_reward += bonus
                print(f"  🏆 Agent scored! (+{1.0 + bonus:.1f})")
            elif reward == -1:
                print(f"  ❌ Opponent scored")
            
            # If we get a regular environmental reward, we're done with this step
            if reward != 0:
                self.previous_observation = next_observation
                if done:
                    break
                continue
            
            # If no environmental reward, see if we should give a small positioning reward
            if i == skip - 1 and not done:  # Only on last step of skip sequence
                paddle_y, ball_x, ball_y, ball_detected = self.detect_paddle_and_ball(next_observation)
                
                if paddle_y is not None and ball_y is not None:
                    # Count how often we're well-aligned
                    total_alignments += 1
                    
                    # Give a small reward for paddle being close to ball's vertical position
                    vertical_diff = abs(paddle_y - ball_y)
                    if vertical_diff < 15:  # Within 15 pixels is good alignment
                        # Scale reward based on ball position - more reward when ball is closer
                        if ball_x is not None:
                            # Define the boundaries for left paddle and right side
                            left_paddle_boundary = 20
                            right_side_boundary = 140
                            
                            # Normalize to 0-1 range and invert so closer to paddle = higher reward
                            ball_closeness = 1.0 - ((right_side_boundary - ball_x) / (right_side_boundary - left_paddle_boundary))
                            positioning_reward = 0.01 * (1.0 + 2.0 * ball_closeness)  # More reward when ball is closer
                        else:
                            positioning_reward = 0.01
                        
                        total_reward += positioning_reward
                        pos_reward += positioning_reward
                        alignment_count += 1
                        
                        # Only log alignment rewards occasionally to reduce console spam
                        if self.steps_since_print >= 20:  # Print every 20 steps
                            self.steps_since_print = 0
                            if self.debug_mode:
                                print(f"  🎯 Good paddle alignment: {alignment_count}/{total_alignments} times")
            
            self.previous_observation = next_observation
            self.steps_since_print += 1
            
            if done:
                break
        
        # Update info dict with reward breakdown
        info = info or {}
        info['env_reward'] = env_reward
        info['pos_reward'] = pos_reward
        
        return next_observation, total_reward, done, info
    
    def close(self):
        """Close the environment."""
        self.env.close()
        
    def preprocess_observation(self, observation, stack_size=4):
        """
        Preprocess the observation and stack frames to capture motion:
        1. Convert to grayscale
        2. Resize to 84x84
        3. Normalize pixel values
        4. Stack last 4 frames
        """
        # Initialize frame stack if it doesn't exist
        if self.frame_stack is None:
            self.frame_stack = deque(maxlen=stack_size)
            for _ in range(stack_size):
                self.frame_stack.append(np.zeros((84, 84), dtype=np.float32))
        
        # Process current frame
        if observation is None:
            processed = np.zeros((84, 84), dtype=np.float32)
        else:
            # Handle different return types
            if isinstance(observation, tuple):
                observation = observation[0]
                
            # Convert RGB to grayscale
            gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            
            # Resize to 84x84
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1]
            processed = resized / 255.0
        
        # Add to stack
        self.frame_stack.append(processed)
        
        # Stack frames to form a single state
        stacked_state = np.stack(list(self.frame_stack), axis=2)
        return stacked_state  # Shape will be (84, 84, 4)
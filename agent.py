import numpy as np
import random
from model import build_dqn_model
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
        self,
        state_shape,
        action_size,
        learning_rate=0.00015,  # Increased from 0.0001
        gamma=0.99,
        epsilon=0.368,  # Starting from where long_run_final left off
        epsilon_min=0.05,
        epsilon_decay=0.997,  # Slightly faster decay to speed up transition to exploitation
        buffer_capacity=100000,
        batch_size=32
    ):
        # Environment parameters
        self.state_shape = state_shape
        self.action_size = action_size
        
        # Learning parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Create Q-networks: main and target
        self.model = build_dqn_model(state_shape, action_size, learning_rate)
        self.target_model = build_dqn_model(state_shape, action_size, learning_rate)
        
        # Initialize target network with same weights as main network
        self.update_target_network()
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Learning step counter and whether to print progress
        self.learning_step = 0
        self.print_progress = False
    
    def update_target_network(self):
        """Update the target network with weights from the main network."""
        self.target_model.set_weights(self.model.get_weights())
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        # Exploration: random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: best action according to Q-values
        state_tensor = np.expand_dims(state, axis=0)
        q_values = self.model.predict_on_batch(state_tensor)
        return np.argmax(q_values[0])
    
    def train(self):
        """Train the agent using Double DQN algorithm with experience replay."""
        # Skip if buffer doesn't have enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch of experiences
        samples = self.replay_buffer.sample(self.batch_size)
        if samples is None:
            return
            
        states, actions, rewards, next_states, dones = samples
        
        # Get current Q-values for all actions in current states
        current_q_values = self.model.predict_on_batch(states)
        
        # DOUBLE DQN: Get actions from main network
        next_actions = np.argmax(self.model.predict_on_batch(next_states), axis=1)
        
        # Get Q-values from target network
        next_q_values = self.target_model.predict_on_batch(next_states)
        
        # Update Q-values for the actions taken
        for i in range(self.batch_size):
            if dones[i]:
                # If done, no future rewards
                current_q_values[i, actions[i]] = rewards[i]
            else:
                # Double DQN: Use main network to select action, target network to evaluate it
                current_q_values[i, actions[i]] = rewards[i] + self.gamma * next_q_values[i, next_actions[i]]
        
        # Train the model with updated Q-values
        self.model.train_on_batch(states, current_q_values)
        
        # Update target network periodically (every 1000 steps)
        self.learning_step += 1
        if self.learning_step % 500 == 0: # Target Network Update Frequency 1000 -> 500
            self.update_target_network()
            print(f"Step {self.learning_step}: Target network updated. Epsilon: {self.epsilon:.4f}")
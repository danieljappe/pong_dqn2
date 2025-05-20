# Create venv:
´py -3.10 -m venv venv´
# Activate venv
´.\venv\Scripts\Activate´
# isntall libaries in environment
´pip install -r requirements.txt´


pong3/
│
├── main.py                # Main script to run the training
├── environment.py         # Environment setup and wrappers
├── agent.py               # DQN agent implementation 
├── model.py               # Neural network architecture
├── replay_buffer.py       # Experience replay memory
└── utils.py               # Helper functions for preprocessing, etc.

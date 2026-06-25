# Lunar Lander RL

This project trains a Deep Q-Network (DQN) agent to play the LunarLander environment using TensorFlow and Keras.

## Project Overview

The repository demonstrates a reinforcement learning workflow for the OpenAI Gym LunarLander environment. The agent learns through a DQN-style training loop with experience replay and target network updates.

## Demo Gallery

The following animations show how the agent behaves at different stages of training.

### Episode 500
<p align="center">
  <img src="./ep500.gif" alt="Episode 500 demo" width="70%" />
</p>

### Episode 1000
<p align="center">
  <img src="./ep1000.gif" alt="Episode 1000 demo" width="70%" />
</p>

### Episode 2000
<p align="center">
  <img src="./ep2000.gif" alt="Episode 2000 demo" width="70%" />
</p>

### Training History
<p align="center">
  <img src="./training_history.png" alt="Training reward curve" width="80%" />
</p>

## Project Files
- [train.py](train.py) — main training script that builds the environment, trains the agent, and saves the model.
- [utils.py](utils.py) — helper functions for action selection, replay buffer sampling, video generation, and reward plotting.
- [requirements.txt](requirements.txt) — Python dependencies required to run the project.
- [training_history.png](training_history.png) — reward curve generated during training.

## Key Features
- DQN-style training with experience replay
- Compatibility with modern Gym/Gymnasium reset and step APIs
- Automatic model saving and training visualization

## Environment Setup

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Training Run

Run the training script with:

```bash
python train.py
```

The script will train the agent, save the model file, and generate training visuals in the project directory.

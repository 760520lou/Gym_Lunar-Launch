# Lunar Lander RL

This project trains a Deep Q-Network (DQN) agent to play the LunarLander environment using TensorFlow and Keras.

## Features
- DQN training loop with experience replay
- Compatible environment reset/step handling for modern Gym APIs
- Video generation and reward history plotting

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run training

```bash
python train.py
```

The script will train the agent, save a model file, and generate videos and training plots in the project directory.

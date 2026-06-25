# Lunar Lander RL

This project trains a Deep Q-Network (DQN) agent to play the LunarLander environment using TensorFlow and Keras.

## Demo Gallery

### Episode 500
<video controls width="100%">
  <source src="./ep500.mp4" type="video/mp4">
</video>

### Episode 1000
<video controls width="100%">
  <source src="./ep1000.mp4" type="video/mp4">
</video>

### Episode 2000
<video controls width="100%">
  <source src="./ep2000.mp4" type="video/mp4">
</video>

### Training History
![Training History](./training_history.png)

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

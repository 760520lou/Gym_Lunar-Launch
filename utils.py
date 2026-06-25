import random
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV is used for assembling video
import tensorflow as tf

# Set a constant random seed for reproducibility.
SEED = 42


def get_action(q_values, epsilon):
    """
    Returns an action using the epsilon-greedy strategy.
    """
    num_actions = q_values.shape[-1]
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    else:
        if hasattr(q_values, "numpy"):
            return int(np.argmax(q_values.numpy()))
        else:
            return int(np.argmax(q_values))


def check_update_conditions(t, update_frequency, memory_buffer):
    """
    Returns True if there are enough samples in the replay buffer and the update frequency condition is met.
    """
    BATCH_SIZE = 64  # Batch size used in get_experiences
    return len(memory_buffer) >= BATCH_SIZE and (t % update_frequency == 0)


def get_experiences(memory_buffer):
    """
    Randomly samples a batch of experiences from the memory buffer.

    Returns:
      A tuple (states, actions, rewards, next_states, dones).
    """
    BATCH_SIZE = 64
    experiences = random.sample(memory_buffer, BATCH_SIZE)
    states = np.array([e.state for e in experiences])
    actions = np.array([e.action for e in experiences])
    rewards = np.array([e.reward for e in experiences], dtype=np.float32)
    next_states = np.array([e.next_state for e in experiences])
    dones = np.array([e.done for e in experiences], dtype=np.float32)
    return states, actions, rewards, next_states, dones


def get_new_eps(epsilon, decay=0.995, min_epsilon=0.01):
    """
    Decays the epsilon value but does not fall below a minimum threshold.
    """
    return max(epsilon * decay, min_epsilon)


def update_target_network(q_network, target_q_network):
    """
    Updates the target network by assigning weights from the Q-network.
    This uses TensorFlow's assign method so it works within tf.function.
    """
    for src, tgt in zip(q_network.weights, target_q_network.weights):
        tgt.assign(src)


def create_video(filename, env, model, fps=30):
    """
    Generates a video of one episode using the provided model.
    This implementation uses OpenCV (cv2) instead of imageio.
    """
    frames = []
    state = env.reset()
    done = False

    while not done:
        # Use the 'rgb_array' render mode to obtain the frame
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        state_q = np.expand_dims(state, axis=0)
        q_values = model(state_q)
        if hasattr(q_values, "numpy"):
            action = int(np.argmax(q_values.numpy()))
        else:
            action = int(np.argmax(q_values))
        state, reward, done, _ = env.step(action)

    # Close the environment if needed.
    env.close()

    # Get frame dimensions from the first frame.
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        # Convert the frame from RGB to BGR (as OpenCV uses BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    video_writer.release()


def plot_history(point_history):
    """
    Plots the training progress (episode vs total points) and saves the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(point_history, label="Total Points")
    plt.xlabel("Episode")
    plt.ylabel("Points")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_history.png")
    plt.show()
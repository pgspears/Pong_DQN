# DQN Pong Agent in TensorFlow.js

This project is a browser-based implementation of a Deep Q-Network (DQN) agent learning to play Pong from scratch using reinforcement learning.

The AI agent starts with no knowledge and learns entirely through trial and error by playing against a hard-coded opponent that plays at a superhuman level. Over time, the agent improves its strategy by optimizing for long-term reward, eventually becoming competitive and capable of winning.

## Features

- Live simulation of AI learning to play Pong
- Built with TensorFlow.js and rendered using HTML Canvas
- Visualizations include:
  - Reward per episode
  - Cumulative wins (Player vs. Opponent)
- Interactive controls for training, saving/loading models, and speed adjustments
- Works entirely in the browser—no server required

## Technologies Used

- JavaScript (vanilla)
- TensorFlow.js
- HTML5 Canvas
- Custom DQN implementation (with experience replay, target network, and epsilon decay)

## How to Use

1. Click **Start Training** to begin learning.
2. Watch the AI improve over time.
3. Use **Save Agent** and **Load Agent** to persist progress.
4. Try **Run Trained Agent** to see performance after training.

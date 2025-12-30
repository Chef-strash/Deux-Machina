import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()

total_reward = 0
done = False

while not done:
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = obs

    # --- Heuristic controller ---
    # 0.5 works better than 1.0 for more control
    target_angle = x*0.5 + vx

    angle_error = target_angle - angle
    angle_todo = angle_error*0.5 - ang_vel

    # faster descent when higher
    target_vy = -0.5 * (y - 0.2) # 0.2 ~ landing pad height
    vy_error = target_vy - vy
    hover_todo = vy_error * 0.5

    action = 0

    # Main engine
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        action = 2

    # prevent Side engines firing too much by ignoring small angles
    elif angle_todo < -0.05:
        action = 3  # fire right engine
    elif angle_todo > 0.05:
        action = 1  # fire left engine

    # Cut engines if landed
    # if left_leg and not right_leg:
    #     action = 1 
    # if right_leg and not left_leg:
    #     action = 3
    if left_leg and right_leg:
        action = 0

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print("Total reward:", total_reward)
env.close()

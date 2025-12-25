import gymnasium as gym


def angular_vel_control(angle, angular_vel, angle_threshold=0.05, angular_vel_threshold=0.05):
    """Controls the angular position and velocity of the lander."""
    if abs(angle) > angle_threshold or abs(angular_vel) > angular_vel_threshold:
        if angle > 0 and angular_vel > 0:
            return 3
        elif angle < 0 and angular_vel < 0:
            return 1
    return 0


def pos_control(x, x_vel, x_threshold=0.05, x_vel_threshold=0.05):
    """Controls the horizontal position and velocity of the lander."""
    if abs(x) > x_threshold or abs(x_vel) > x_vel_threshold:
        if x > 0 and x_vel > 0:
            return 1
        elif x < 0 and x_vel < 0:
            return 3
    return 0


def custom_policy(observation):
    """Custom policy for controlling the Lunar Lander."""
    x, y, x_vel, y_vel, angle, angular_vel, right_contact, left_contact = observation

    # Stabilize the lander if one leg has made contact
    if right_contact:
        return 1
    elif left_contact:
        return 3

    # Fire main engine if descending too quickly
    if y_vel < -0.8 and y < 0.8:
        return 2

    # Fire main engine if horizontal position and angle are near zero
    if abs(x) < 0.05 and abs(angle) < 0.05:
        if y_vel < -0.3:
            return 2

    pos_action = pos_control(x, x_vel)
    angle_action = angular_vel_control(angle, angular_vel)

    # If horizontal position or angle are not near zero
    if abs(x) >= 0.05 or abs(angle) >= 0.05:
        # Fire rigth or left engine based on dominant deviation
        if abs(x_vel) > abs(angular_vel):
            if pos_action != 0:
                return pos_action
        else:
            if angle_action != 0:
                return angle_action

    return 0


# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset()
run = True
total_reward = 0
while (run):
    # this is where you would insert your policy
    action = custom_policy(observation)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        run = False

print("Total reward: ", total_reward)
env.close()
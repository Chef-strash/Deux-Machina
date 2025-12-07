import gymnasium as gym

# Gains / thresholds
ANGLE_KP = 6.0
ANGLE_KD = 2.0
HORIZ_KP = 2.0
HORIZ_KD = 1.0
MAIN_VY_THRESHOLD = -0.35
NEAR_GROUND_Y = 0.25

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

total_reward = 0.0
done = False

while not done:
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = observation

    # default
    action = 0

    # if any leg touches, stop firing
    if left_contact or right_contact:
        action = 0
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if done:
            break
        else:
            continue

    # safety: slow descent with main engine
    if vy < MAIN_VY_THRESHOLD:
        action = 2
    else:
        # PD for rotation (want angle -> 0)
        # NOTE: corrected mapping:
        #   angle > 0  => tilted clockwise => fire RIGHT thruster (action=3) to correct
        #   angle < 0  => tilted counter-clockwise => fire LEFT thruster (action=1) to correct
        torque_cmd = ANGLE_KP * angle + ANGLE_KD * ang_vel

        # PD for horizontal correction (want x -> 0)
        horiz_cmd = HORIZ_KP * x + HORIZ_KD * vx

        # rotation correction has priority
        if abs(torque_cmd) > 0.12:
            if torque_cmd > 0:
                action = 3   # fire right thruster to produce counter-clockwise torque
            else:
                action = 1   # fire left thruster to produce clockwise torque
        else:
            # small rotation error -> attend to horizontal drift
            if abs(horiz_cmd) > 0.08:
                # corrected mapping: to move left (x>0) fire right thruster (action=3)
                if horiz_cmd > 0:
                    action = 3
                else:
                    action = 1
            else:
                action = 0

        # near ground: be conservative and prefer main engine if descending
        if y < NEAR_GROUND_Y:
            if vy < -0.15:
                action = 2
            else:
                # if small angle/vx remain, do nothing to avoid overcorrection
                if abs(vx) < 0.12 and abs(angle) < 0.12:
                    action = 0

    # apply action
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print("Episode finished. Total reward:", total_reward)
env.close()
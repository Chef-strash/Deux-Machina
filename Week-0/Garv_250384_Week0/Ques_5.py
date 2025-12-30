import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

run = True
total_reward = 0
while(run):
    
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = observation
    
    action = 0  # Default: do nothing
    
    # Angular stabilization
    if abs(angle) > 0.3 or abs(ang_vel) > 0.45 :
        if angle > 0.1:
            action = 1 
        elif 0<=abs(angle)<=0.1:
            action=0 
        else:
            action = 3 
    # Descent rate
    elif vy < -0.5 or (y < 1.0 and vy < -0.2):
        action = 2  
    # Hor. positioning
    elif abs(x) > 0.15 or (abs(vx) > 0.3 and abs(x) > 0.05):
        if x > 0 or (vx > 0 and x > 0.02):
            action = 1  
        else:
            action = 3  
  
    elif y < 0.3 and abs(x) < 0.1 and abs(vx) < 0.2:
        if vy < -0.1:
            action = 2  
    
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        observation, info = env.reset()
        run = False

print("Total Reward:", total_reward)
env.close()
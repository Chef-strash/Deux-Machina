import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset(seed=42)
x=observation[0]
y=observation[1]
y_vel=observation[3]
land_angle=observation[4]
run = True
total_reward = 0
while(run):
    if x<-0.1:
        if land_angle<0.1:
            action=3
        else:
            action=2
    elif x>0.17:
        if land_angle>-0.05:
            action=1
        else:
            action=2
    else:
        if y>0.5 and y_vel<-0.4:
            action=2
        else:
            action=0
   
    observation, reward, terminated, truncated, info = env.step(action)
    x=observation[0]
    y=observation[1]
    total_reward += reward
   
    if terminated or truncated:
        observation, info = env.reset()
        run = False
        
print("Total Reward:", total_reward)
env.close()

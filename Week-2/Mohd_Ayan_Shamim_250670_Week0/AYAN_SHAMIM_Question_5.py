import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=9999)
x_co=observation[0]
y_co=observation[1]
vx=observation[2]
vy=observation[3]
ld_angle=observation[4]
ang_vel=observation[5]
left_leg_contact=observation[6]
right_leg_contact=observation[7]
run = True
total_reward = 0
while(run):
    # this is where you would insert your policy
    action = env.action_space.sample()
    if x_co < -0.01:
        action = 3  # Fire right engine
    elif x_co > 0.01:
        action = 1  # Fire left engine
    else:
        action = 0  # Do nothing
    if vy < -0.1:
        action = 2  # Fire main engine for making the landing soft
    if ld_angle>0.5:
        action = 3 #Fire right engine
    elif ld_angle<-0.5:
        action =1 #Fire left engine
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    x_co=observation[0]
    y_co=observation[1]
    vx=observation[2]
    vy=observation[3]
    ld_angle=observation[4]
    ang_vel=observation[5]
    left_leg_contact=observation[6]
    right_leg_contact=observation[7]
    #some observations are not used ignore the same
    #I think the best strategy is to not let the probe tilt to a large angle while maintaing a less vy so that landing can be done softly
    #Also at the same time I am maintaining the x coordinate so that it lands between the flags
    total_reward += reward
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        run = False
        
print("Total Reward:", total_reward)
env.close()

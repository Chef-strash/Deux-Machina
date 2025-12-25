import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Get the first observation -> Initial State
observation, info = env.reset()
# Here observation is the current state of the environment (i.e, the position and velocity of the lander)
# observation is a numpy array of 8 floats representing the x coordinate, y coordinate, x velocity, y velocity, lander angle, angular velocity, left leg contact, right leg contact
# You can use a subset of these values to create your own strategy for landing the lunar lander.
# For example, I am using the x coordinate and y coordinate to create a simple strategy.
x_coord = observation[0]
y_coord = observation[1]
angle=observation[4]
y_vel=observation[3]
action=0
print("Initial Observation:", observation)
run = True
total_reward = 0
while(run):
    if y_coord>0.7:
        if angle>0.65:
            action=3
        elif angle<-0.65:
            action=1
        elif x_coord<-0.1:
            action=3
        elif x_coord>0.1:
            action=1
        else:
            action=0
    
    #phase near
    else:
        if y_vel<-0.2:
            action=2

        elif x_coord<-0.1:
            action=3
        elif x_coord>0.1:
            action=1
        elif angle>0.65:
            action=3
        elif angle<-0.65:
            action=1
        elif y_vel<-1:
            action=2
        else:
            action=0
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    x_coord = observation[0]
    y_coord = observation[1]
    angle=observation[4]
    y_vel=observation[3]
    total_reward += reward
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        run = False

print("Total Reward:", total_reward)
env.close()

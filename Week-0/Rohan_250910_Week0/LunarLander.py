import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Get the first observation -> Initial State
observation, info = env.reset()
# Here observation is the current state of the environment (i.e, the position and velocity of the lander)
# observation is a numpy array of 8 floats representing the x coordinate, y coordinate, x velocity, y velocity, lander angle, angular velocity, left leg contact, right leg contact
# You can use a subset of these values to create your own strategy for landing the lunar lander.
# For example, I am using the x coordinate and y coordinate to create a simple strategy.

print("Initial Observation:", observation)
run = True
total_reward = 0
while(run):
    x       = observation[0]
    y       = observation[1]
    x_vel   = observation[2]
    y_vel   = observation[3]
    ang      = observation[4]
    ang_v     = observation[5]
    left_leg  = observation[6]
    right_leg = observation[7]
    # it takes into account the angular tilt with angle and angular velocity adn tries to minimize it.
    if 0.1<5*ang+2*ang_v<-0.1:
        action=0
    if 5*ang+2*ang_v>0.1:
        action=1
    if 5*ang+2*ang_v<-0.1:
        action=3
    if x < -0.1:
        action = 3  
    elif x > 0.1:
        action = 1  
    else:
        action = 0  

    if y<0.5 and y_vel>0.001:
        action=2
    action = env.action_space.sample()
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        print("Final Obs",observation)
        observation, info = env.reset()
        run = False
print("Total Reward:", total_reward)
env.close()

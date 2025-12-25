import gymnasium as gym
import random


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
lang= observation[4]
vang=observation[5]
yvel=observation[3]
xvel=observation[2]

print("Initial Observation:", observation)
run = True
total_reward = 0
while(run):
    # this is where you would insert your strategy
    # print(x_coord, y_coord)
    if x_coord < -0.05:
        action = 3  # Fire left engine

    elif x_coord > 0.05:
        action = 1  # Fire right engine
    
    else:
        action = 0  # Do nothing
    
    if (lang > 0.04 and not (observation[7]==1.0 and observation[6]==1.0)):
        action=3    # Fire left engine
        if (x_coord>0.6 and random.random()>0.75): action=2
    elif (lang < -0.04 and not (observation[7]==1.0 and observation[6]==1.0)):
        action=1    # Fire right engine
        if (x_coord<-0.6 and random.random()>0.75): action=2
    if (xvel>0.3):
        action = 1  # Fire right engine
    elif (xvel<-0.3):
        action = 3  # Fire left engine

    if (vang> 0.2 and not (observation[7]==1.0 and observation[6]==1.0)) :
        action=3   # Fire left engine
    elif (vang<-0.2 and not (observation[7]==1.0 and observation[6]==1.0)):
        action=1 # Fire right engine
    if (x_coord < -0.01 and action==0 ):
        action = 3  # Fire left engine

    elif (x_coord > 0.01 and action==0) :
        action = 1  # Fire right engine
        
    #if (observation[7]==1.0 or observation[8]==1.0):
        #action =0
    if (yvel<-0.2 ): 
        action=2
    if ((y_coord<0.001)and not (observation[7]==1.0 and observation[6]==1.0)):
        action = 2
        
        
    if (action==0 and yvel<2):action=0
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    x_coord = observation[0]
    y_coord = observation[1]
    lang= observation[4]
    vang=observation[5]
    xvel=observation[2]
    yvel=observation[3]
    total_reward += reward
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        print("Final Observation",observation)
        observation, info = env.reset()
        run = False

print("Total Reward:", total_reward)
env.close()
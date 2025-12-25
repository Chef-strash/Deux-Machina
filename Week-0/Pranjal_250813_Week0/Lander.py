import gymnasium as gym
'''0: do nothing

1: fire left orientation engine

2: fire main engine

3: fire right orientation engine'''

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
    # this is where you would insert your strategy
    # print(x_coord, y_coord)

    x, y, vx, vy, angle, w, leg1, leg2 = observation

    #predicted angle is the angle to which the lander will tilt after 0.7 seconds of rotation
    #I am using predicted angle out here because when lander fires it sides engines, it over tilts so we are looking at future 0.7 seconds later
    pred_angle = angle + 0.7*w

    #To move it near to 0,0 we use desired tilt.
    #To move it, we fire the main engines. But first we tilt our lander in such a way that its nose faces the desired position, or the tilt lies in a close proximity of desired tilt and then we fire main engines
    desired_tilt = (x * 0.5) + (vx * 1.0)
    
    #To make sure that as it gets clode to surface, it must be in upright position to make a soft landing
    if y < 0.2:
        desired_tilt = 0

    # Here I am capping my desired_tilt angle because I don't wnat it to till 90 degree even if required
    if desired_tilt > 0.4: desired_tilt = 0.4
    if desired_tilt < -0.4: desired_tilt = -0.4

    
    angle_error = pred_angle - desired_tilt
    
    
    #Condition to check if the predicted angle lies within close range of desired angle
    if angle_error > 0.03:
        action = 3
    elif angle_error < -0.03:
        action = 1
    #if it lies, then I just have to control the descent of lander.
    # I have designed a two system check if y > 0.5, it is is allowed to descend at a much higher speed but after that threshold it must descend at a lower speed.
    # To decrease the speed it must fire its main engines upward 
    else :
        if y > 0.5:
            if vy < -0.8:
                action = 2
            else : action = 0
        else :
            if vy < -0.4:
                action = 2
            
            else : action = 0
     
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    x_coord = observation[0]
    y_coord = observation[1]
    total_reward += reward
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        run = False

print("Total Reward:", total_reward)
env.close()

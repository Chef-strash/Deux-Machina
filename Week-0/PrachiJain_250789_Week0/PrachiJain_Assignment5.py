import time
import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Get the first observation -> Initial State
observation, info = env.reset()
# Here observation is the current state of the environment (i.e, the position and velocity of the lander)
# observation is a numpy array of 8 floats representing the x coordinate, y coordinate, x velocity, y velocity, lander angle, angular velocity, left leg contact, right leg contact
x_coord, y_coord, x_vel, y_vel, angl, angl_vel, l_ctct, r_ctct = observation
run = True
total_reward = 0
while(run):
    # this is my strategy, the higher prority variables are put lower,
    # in general angle has higher priority, because the lander topples very quickly,
    # and y_velocity has to be decreased as the lander approaches ground to prevent crashing
    # rest of the values and positions of variables were kind of fine-tuned to improve landing
    if x_coord < -0.1 or x_vel<-1.5:
        action = 3  # Fire right engine
    elif x_coord > 0.1 or x_vel>1.5:
        action = 1  # Fire left engine
    else:
        action = 0  # Do nothing
    if y_vel<-1:
        action = 2  # Fire main engine
    if angl<-0.20:
        action = 1  # Fire left engine
    elif angl>0.20:
        action = 3  # Fire right engine
    if(y_vel<-0.5 and y_coord<0.5):
        action = 2  # Fire main engine
    if angl_vel>1.5:
        action = 3  # Fire right engine
    elif angl_vel<-1.5:
        action = 1  # Fire left engine
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    print(observation," ",action)
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.05)   # slowing the lander to study how the action is affecting it
    x_coord, y_coord, x_vel, y_vel, angl, angl_vel, l_ctct, r_ctct = observation
    total_reward += reward
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        run = False

print("Total Reward:", total_reward)
env.close()
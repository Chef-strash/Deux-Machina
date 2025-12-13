import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v2", render_mode="human")

# Get the first observation -> Initial State
observation, info = env.reset()
# Here observation is the current state of the environment (i.e, the position and velocity of the lander)
# observation is a numpy array of 8 floats representing the x coordinate, y coordinate, x velocity, y velocity, lander angle, angular velocity, left leg contact, right leg contact
# You can use a subset of these values to create your own strategy for landing the lunar lander.
# For example, I am using the x coordinate and y coordinate to create a simple strategy.
x_coord = observation[0]
y_coord = observation[1]
xv=observation[2]
yv=observation[3]
lander_angle=observation[4]
angular_velocity=observation[5]
left_leg_contact=observation[6]
right_leg_contact=observation[7]

print("Initial Observation:", observation)
run = True
total_reward = 0
count=0
while(run):
    if(left_leg_contact or right_leg_contact):
        if(lander_angle>.2):
            action=3
        elif(lander_angle<-.2):
            action=1
        else:
            action=0
    elif(y_coord>.1):
        
        if(count%6==0 or count%6==3 or count%6==1):
            if(y_coord>0.05 and y_coord<1 and yv<-.3):
                action =2
            else:
                if(lander_angle>.2):
                    action=3
                elif(lander_angle<-.2):
                    action=1
                else:
                    if(x_coord>0.1 ):
                        action=1
                    elif(x_coord<-0.1):
                        action=3 
                    else:
                        action =2
        elif(count%6==4 or count%6==2):
            if(lander_angle>.2):
                action=3
            elif(lander_angle<-.2):
                action=1
            else:
                if(x_coord>0.1 ):
                    action=1
                elif(x_coord<-0.1):
                    action=3 
                else:
                    action =0
        elif(count%6==5 ):
            if(x_coord>0.1 ):
                action=1
            elif(x_coord<-0.1):
                action=3 
            else:
                if(lander_angle>.2):
                    action=3
                elif(lander_angle<-.2):
                    action=1
        else:
            if(count%12==1):
                action =1
            else:
                action=3
    else:
        if(count%4==0 or count%4==1 or count%4==3):
            action =2
        else:
            
            if(lander_angle>.2):
                action=3
            elif(lander_angle<-.2):
                action=1
            else:
                if(x_coord>0.1 ):
                    action=1
                elif(x_coord<-0.1):
                    action=3 
                else:
                    action =2
    

   
    
   

    
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    x_coord = observation[0]
    y_coord = observation[1]
    xv=observation[2]
    yv=observation[3]
    lander_angle=observation[4]
    angular_velocity=observation[5]
    left_leg_contact=observation[6]
    right_leg_contact=observation[7]
    total_reward += reward
    # If the episode has ended then we can reset to start a new episode
    print("Initial Observation:", observation)

    if terminated or truncated:
        observation, info = env.reset()
        run = False
    count+=1

print("Total Reward:", total_reward)
env.close() 
import gymnasium as gym
from numpy import angle

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()

x = observation[0]        # Horizontal position (took x not x_coord as in example provided to us for ease)
y = observation[1]        # Height or Vertical position (took y not y_coord as in example provided to us for ease)
x_vel = observation[2]    # Horizontal speed
y_vel = observation[3]    # Vertical speed
angle = observation[4]    # Lander's angle

print("--> Initial Observation:", observation)
run = True
total_reward = 0

print("--> Starting simulation..")

while(run):
             # action 0: Do nothing
             # action 1: Fire left orientation engine
             # action 2: Fire main engine
             # action 3: Fire right orientation engine

    ### Now, trying to stay centered with extra monitoring from constraints of x_vel:
        # Also insuring that left and right orientation engine are not fired always when abs(x) > 0.1
        # But only when it has less x_vel to come inwards
        # i.e if x > 0.15 but x_vel < -0.1, we don't fire left orientation engine because it's already coming inwards
        # and similarly with x < -0.15, we only fire right orientation engine when x_vel < 0.1 because at that point it has less velocity to come inwards

    if x > 0.2:
        if x_vel >= -0.1:
            action = 1            # Fire left orientation engine
    elif x < -0.2:
        if x_vel <= 0.1:
            action = 3            # Fire right orientation engine
    else:
        action = 0                # Do nothing
        
    ### Slowing down while descending:
        # Previous given example didn't had height based control due to which it was crashing at high speed
        # Hence, now added extra constraints by monitoring of y_vel

    if y >= 0.7:
        if y_vel < -2.0:
            action = 2        # Firing main engine only when falling too fast according to it's height
        else:
            action = 0        # Else free fall until it moving too fast
    elif 0.3 <= y < 0.7:
        if y_vel < -1.0:
            action = 2        # Firing main engine only when falling too fast according to it's height
        else:
            action = 0        # Else free fall until it moving too fast
    elif y < 0.3:
        if y_vel < -0.5:
            action = 2        # Firing main engine for soft landing
        else:
            action = 0        # No main engined fired if already landing softly

    # We controlled x and y movements but didn't monitor the tilt of lander due to firing of left and right orientation engines
    # As observed :
                # right orientation engine also tilts lander to left
                # left orientation engine also tilts lander to right
    ### Adding tilt monitors:
    if y < 0.5:
        if angle > 0.12:       # tilted to right
            action = 3         # firing right orientation engine to rotate to left
    elif angle < -0.12:        # tilted to left
        action = 1             # firing left orientation engine to rotate to right

    observation, reward, terminated, truncated, info = env.step(action)
    
    x = observation[0]        # Horizontal position
    y = observation[1]        # Height or Vertical position
    x_vel = observation[2]    # Horizontal speed
    y_vel = observation[3]    # Vertical speed
    angle = observation[4]    # Lander's angle

    total_reward += reward

    if terminated or truncated:
        run = False
        
print("--> Total Reward:", total_reward)
print("--> Final Observation:", observation)
env.close()

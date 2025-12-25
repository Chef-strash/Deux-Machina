import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
state, info = env.reset()
total = 0
run = True
episode = 0
no_of_episode_to_run = 5
while run and episode< no_of_episode_to_run :
    x, y, vx, vy, angle,anglev,lc ,rc = state
    action = 0
    #if the lander falling too fast then to slow down the fall
    if vy < -0.4: action = 2 
    if vy > 0.4 : action = 0
    # First we see the postion of the lander, if it spinning too fast we activate the other engine 
    # Otherwise we activatre the engine normally
    elif x > 0.1 or (x > 0 and vx > 0.1):  
        if  abs(anglev) >0.5:
            action = 3
        else:
            action = 1   
    elif x < -0.1 or (x < 0 and vx < -0.1):
        if  abs(anglev) >1:
            action = 1
        else:
            action = 3                   
    
    observation, reward, terminated, truncated, info = env.step(action)
    total += reward
    if terminated or truncated:
        print(f"Episode {episode + 1} ended! Total Reward: {total:.2f}")
        episode += 1
        if episode < no_of_episode_to_run:
            observation, info = env.reset()
            total = 0
        else:
            run = False

env.close()
print(f"Score: {(total):.1f}")
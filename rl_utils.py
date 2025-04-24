# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:35:05 2025

@author:  onno.niemann@uam.es  alberto.suarez@uam.es

Adapted from ...
"""
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import numpy as np

def generate_greedy_episode(environment, Qtable):
    # Run one episode and store frames
    frames = []
    state, info = environment.reset()
    done = False

    while not done:
        frames.append(environment.render())  # Capture current frame (RGB array)
        action = np.argmax(Qtable[state])
        state, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        if done:
            frames.append(environment.render())

    environment.close()
    return frames
    
def show_episode(frames, interval=100):
    # Create animation from frames
    fig = plt.figure()
    im = plt.imshow(frames[0])
    
    def update(frame):
        im.set_array(frame)
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=frames, 
                                  interval=interval, repeat=False)
    plt.close()  # Prevents duplicate static image
    
    # Display animation
    return HTML(ani.to_jshtml())
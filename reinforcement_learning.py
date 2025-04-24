# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:06:34 2025
@author:  onno.niemann@uam.es  alberto.suarez@uam.es

Adapted from ...

"""
# from tqdm import tqdm

from tqdm.notebook import tqdm
import numpy as np
import random

# [To-Do]: fill in how to select the action 
def greedy_policy(Qtable, state):
    # Exploitation only: take the action with the highest (state, action) value
    action = np.argmax(Qtable[state])
    return action

# [To-Do]: Define the epsilon-greedy policy
def epsilon_greedy_policy(Qtable, state, epsilon, environment):

    if np.random.rand() < epsilon:
        action = environment.action_space.sample()
    else:
        action = greedy_policy(Qtable, state)
    return action


# [To-Do]: Impelemt Q-learning
def q_learning(
    environment, 
    n_training_episodes,
    max_steps,
    learning_rate,
    gamma,
    min_epsilon, 
    max_epsilon, 
    decay_rate,  
    Qtable,
):
    
    for episode in tqdm(range(n_training_episodes)):
     
        # Reduce epsilon (reduce exploration as learning progresses)
        epsilon = (
            min_epsilon
            + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
        )

        # Reset the environment
        state, info = environment.reset()

        # Episode loop    
        episode_over = False
        n_steps = 0

        while not episode_over and n_steps < max_steps:
            n_steps += 1

            # Choose action (a) at state (s) using an epsilon-greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon, environment)
            # Take the action (a) and observe the new state(s') and reward (r)
            next_state, reward, terminated, truncated, info = environment.step(action)
            # Update Q-table and state
            Qtable[state, action] += learning_rate * (
                reward + gamma * np.max(Qtable[next_state]) - Qtable[state, action]
            )
            state = next_state
            # Determine whether the episode is over
            episode_over = terminated or truncated
    return Qtable


# [To-Do]: Implement SARSA 
def sarsa_learning(
    environment, 
    n_training_episodes, 
    learning_rate,
    gamma,
    min_epsilon, 
    max_epsilon, 
    decay_rate, 
    max_steps, 
    Qtable,
):
    # Función para calcular epsilon basado en el episodio actual
    def get_epsilon(episode):
        return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    
    # Política epsilon-greedy para seleccionar acciones
    def get_action(state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return environment.action_space.sample()  # Exploración
        else:
            return np.argmax(Qtable[state, :])  # Explotación
    
    # Entrenamiento por episodios
    for episode in tqdm(range(n_training_episodes)):
        # Calcular epsilon para este episodio
        epsilon = get_epsilon(episode)
        
        # Reiniciar el entorno
        state, _ = environment.reset()
        
        # Seleccionar primera acción usando política epsilon-greedy
        action = get_action(state, epsilon)
        
        # Recorrer pasos del episodio
        for step in range(max_steps):
            # Tomar acción y observar resultado
            new_state, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            
            # Seleccionar siguiente acción usando política epsilon-greedy
            new_action = get_action(new_state, epsilon)
            
            # Actualizar Q-table usando la regla de SARSA (on-policy)
            Qtable[state, action] = Qtable[state, action] + learning_rate * (
                reward + gamma * Qtable[new_state, new_action] * (not done) - Qtable[state, action]
            )
            
            # Terminar episodio si llegamos al final
            if done:
                break
                
            # Actualizar estado y acción para el siguiente paso
            state = new_state
            action = new_action
            
    return Qtable
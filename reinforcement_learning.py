# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:06:34 2025
@author:  onno.niemann@uam.es  alberto.suarez@uam.es

Adapted from ...

"""
from tqdm.notebook import tqdm
import numpy as np


# fill in how to select the action
def greedy_policy(Qtable, state):
    # Exploitation only: take the action with the highest (state, action) value
    action = np.argmax(Qtable[state])
    return action

# Define the epsilon-greedy policy
def epsilon_greedy_policy(Qtable, state, epsilon, environment):

    if np.random.rand() < epsilon:
        action = environment.action_space.sample()
    else:
        action = greedy_policy(Qtable, state)
    return action


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

        epsilon = (
            min_epsilon
            + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        )

        state, info = environment.reset()

        episode_over = False
        n_steps = 0

        while not episode_over and n_steps < max_steps:
            n_steps += 1

            # Elegir una acción (a) en el estado (s) usando una política epsilon-greedy
            action = epsilon_greedy_policy(Qtable, state, epsilon, environment)

            # Ejecutar la acción (a) y observar el nuevo estado (s') y la recompensa (r)
            next_state, reward, terminated, truncated, info = environment.step(action)

            # Actualizar la Q-table y el estado
            Qtable[state, action] += learning_rate * (
                reward + gamma * np.max(Qtable[next_state]) - Qtable[state, action]
            )
            state = next_state
            # Determinar si el episodio ha terminado
            episode_over = terminated or truncated

    return Qtable



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
    """
    Implementación del algoritmo SARSA.
    """
    for episode in tqdm(range(n_training_episodes)):
        # Calcular epsilon para este episodio
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        
        # Reiniciar el entorno
        state = environment.reset()
        
        # Manejar diferentes formatos de retorno de reset()
        if isinstance(state, tuple):
            state = state[0]
            
        # Elegir primera acción con política epsilon-greedy
        if np.random.random() < epsilon:
            action = environment.action_space.sample()
        else:
            action = np.argmax(Qtable[state, :])
        
        for step in range(max_steps):
            # Tomar acción y observar resultado con compatibilidad para ambos formatos
            try:
                # Nueva API (5 valores)
                next_state, reward, terminated, truncated, info = environment.step(action)
                done = terminated or truncated
            except ValueError:
                # Antigua API (4 valores)
                next_state, reward, done, info = environment.step(action)
                
            # Elegir siguiente acción con política epsilon-greedy
            if np.random.random() < epsilon:
                next_action = environment.action_space.sample()
            else:
                next_action = np.argmax(Qtable[next_state, :])
            
            # Actualizar Q-table con la regla de SARSA
            Qtable[state, action] = Qtable[state, action] + learning_rate * (
                reward + gamma * Qtable[next_state, next_action] - Qtable[state, action]
            )
            
            # Preparar para el siguiente paso
            state = next_state
            action = next_action
            
            # Terminar episodio si llegamos al final
            if done:
                break
                
    return Qtable

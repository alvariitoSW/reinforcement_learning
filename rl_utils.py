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
from tqdm.notebook import tqdm

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

##########################################################################


# Codigo que compara los algoritmos SARSA y Q-learning
def run_experiments(large_environment, q_learning_param, sarsa_param):
    """
    Función que compara los algoritmos SARSA y Q-learning en un entorno dado,
    mostrando el promedio de métricas por cada 100 episodios.
    
    Parámetros:
    -----------
    large_environment : gym.Env
        Entorno de OpenAI Gym donde se ejecutarán los algoritmos
    q_learning_param : dict
        Diccionario con parámetros para el algoritmo Q-learning
    sarsa_param : dict
        Diccionario con parámetros para el algoritmo SARSA
    """
    
    # Extraemos los parámetros para mejor legibilidad
    q_episodes = q_learning_param['n_training_episodes']
    q_steps = q_learning_param['max_steps']
    q_lr = q_learning_param['learning_rate']
    q_gamma = q_learning_param['gamma']
    q_max_eps = q_learning_param['max_epsilon']
    q_min_eps = q_learning_param['min_epsilon']
    q_decay = q_learning_param['decay_rate']
    q_table = q_learning_param['Q_table']
    
    sarsa_episodes = sarsa_param['n_training_episodes']
    sarsa_steps = sarsa_param['max_steps']
    sarsa_lr = sarsa_param['learning_rate']
    sarsa_gamma = sarsa_param['gamma']
    sarsa_max_eps = sarsa_param['max_epsilon']
    sarsa_min_eps = sarsa_param['min_epsilon']
    sarsa_decay = sarsa_param['decay_rate']
    sarsa_table = sarsa_param['Q_table_sarsa']
    
    # Definir el tamaño del bloque para promediar
    avg_block_size = 50
    
    # Crear listas para acumular valores por cada bloque
    q_block_rewards = []
    q_block_lengths = []
    q_block_errors = []
    current_q_rewards = []
    current_q_lengths = []
    current_q_errors = []
    
    sarsa_block_rewards = []
    sarsa_block_lengths = []
    sarsa_block_errors = []
    current_sarsa_rewards = []
    current_sarsa_lengths = []
    current_sarsa_errors = []
    
    print("Entrenando algoritmo Q-learning...")
    
    # Entrenar Q-learning y recopilar métricas
    for episode in tqdm(range(q_episodes)):
        # Reiniciar el entorno
        state = large_environment.reset()[0]  # OpenAI Gym retorna (obs, info)
        
        # Inicializar métricas del episodio
        total_reward = 0
        step_count = 0
        cumulative_error = 0
        
        # Calcular epsilon para este episodio
        epsilon = q_min_eps + (q_max_eps - q_min_eps) * np.exp(-q_decay * episode)
        done = False
        
        # Ejecutar un episodio completo
        while not done and step_count < q_steps:
            # Seleccionar acción usando política epsilon-greedy
            if np.random.random() < epsilon:
                action = large_environment.action_space.sample()  # Exploración
            else:
                action = np.argmax(q_table[state, :])  # Explotación
            
            # Tomar la acción y obtener nueva información
            next_state, reward, terminated, truncated, _ = large_environment.step(action)
            done = terminated or truncated
            
            # Actualizar Q-table usando Q-learning (Actualización off-policy)
            next_best_action = np.argmax(q_table[next_state, :])
            td_target = reward + q_gamma * q_table[next_state, next_best_action] * (not done)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += q_lr * td_error
            
            # Actualizar métricas
            total_reward += reward
            step_count += 1
            cumulative_error += abs(td_error)
            
            # Actualizar estado
            state = next_state
        
        # Agregar métricas del episodio actual al bloque actual
        current_q_rewards.append(total_reward)
        current_q_lengths.append(step_count)
        current_q_errors.append(cumulative_error / max(1, step_count))  # Error promedio
        
        # Si completamos un bloque, calcular promedios y reiniciar
        if (episode + 1) % avg_block_size == 0:
            q_block_rewards.append(np.mean(current_q_rewards))
            q_block_lengths.append(np.mean(current_q_lengths))
            q_block_errors.append(np.mean(current_q_errors))
            
            # Reiniciar acumuladores para el siguiente bloque
            current_q_rewards = []
            current_q_lengths = []
            current_q_errors = []
    
    print("Entrenando algoritmo SARSA...")
    
    # Entrenar SARSA y recopilar métricas
    for episode in tqdm(range(sarsa_episodes)):
        # Reiniciar el entorno
        state = large_environment.reset()[0]
        
        # Inicializar métricas del episodio
        total_reward = 0
        step_count = 0
        cumulative_error = 0
        
        # Calcular epsilon para este episodio
        epsilon = sarsa_min_eps + (sarsa_max_eps - sarsa_min_eps) * np.exp(-sarsa_decay * episode)
        
        # Inicializar la primera acción usando política epsilon-greedy
        if np.random.random() < epsilon:
            action = large_environment.action_space.sample()  # Exploración
        else:
            action = np.argmax(sarsa_table[state, :])  # Explotación
        
        done = False
        
        # Ejecutar un episodio completo
        while not done and step_count < sarsa_steps:
            # Tomar la acción y obtener nueva información
            next_state, reward, terminated, truncated, _ = large_environment.step(action)
            done = terminated or truncated
            
            # Seleccionar siguiente acción usando política epsilon-greedy (para SARSA)
            if np.random.random() < epsilon:
                next_action = large_environment.action_space.sample()  # Exploración
            else:
                next_action = np.argmax(sarsa_table[next_state, :])  # Explotación
            
            # Actualizar Q-table usando SARSA (Actualización en política)
            td_target = reward + sarsa_gamma * sarsa_table[next_state, next_action] * (not done)
            td_error = td_target - sarsa_table[state, action]
            sarsa_table[state, action] += sarsa_lr * td_error
            
            # Actualizar métricas
            total_reward += reward
            step_count += 1
            cumulative_error += abs(td_error)
            
            # Actualizar estado y acción
            state = next_state
            action = next_action
        
        # Agregar métricas del episodio actual al bloque actual
        current_sarsa_rewards.append(total_reward)
        current_sarsa_lengths.append(step_count)
        current_sarsa_errors.append(cumulative_error / max(1, step_count))  # Error promedio
        
        # Si completamos un bloque, calcular promedios y reiniciar
        if (episode + 1) % avg_block_size == 0:
            sarsa_block_rewards.append(np.mean(current_sarsa_rewards))
            sarsa_block_lengths.append(np.mean(current_sarsa_lengths))
            sarsa_block_errors.append(np.mean(current_sarsa_errors))
            
            # Reiniciar acumuladores para el siguiente bloque
            current_sarsa_rewards = []
            current_sarsa_lengths = []
            current_sarsa_errors = []
    
    # Si quedan episodios sin promediar (último bloque incompleto), calculamos el promedio con lo que hay
    if current_q_rewards:
        q_block_rewards.append(np.mean(current_q_rewards))
        q_block_lengths.append(np.mean(current_q_lengths))
        q_block_errors.append(np.mean(current_q_errors))
    
    if current_sarsa_rewards:
        sarsa_block_rewards.append(np.mean(current_sarsa_rewards))
        sarsa_block_lengths.append(np.mean(current_sarsa_lengths))
        sarsa_block_errors.append(np.mean(current_sarsa_errors))
    
    # Crear ejes x para las gráficas (representando el final de cada bloque de 100 episodios)
    q_episodes_x = np.arange(avg_block_size, avg_block_size * (len(q_block_rewards) + 1), avg_block_size)[:len(q_block_rewards)]
    sarsa_episodes_x = np.arange(avg_block_size, avg_block_size * (len(sarsa_block_rewards) + 1), avg_block_size)[:len(sarsa_block_rewards)]
    
    # Configurar el estilo de matplotlib
    plt.style.use('ggplot')
    plt.figure(figsize=(18, 12))
    
    # 1. Gráfica de recompensa por episodio (promedio por bloques de 100)
    plt.subplot(3, 1, 1)
    plt.plot(q_episodes_x, q_block_rewards, label='Q-learning', color='blue', linestyle='-', marker='o')
    plt.plot(sarsa_episodes_x, sarsa_block_rewards, label='SARSA', color='red', linestyle='-', marker='x')
    plt.xlabel(f'Episodio (promedio por cada {avg_block_size})')
    plt.ylabel('Recompensa Media')
    plt.title(f'Comparación de Recompensa Media por cada {avg_block_size} Episodios')
    plt.legend()
    plt.grid(True)
    
    # 2. Gráfica de longitud de episodio (promedio por bloques de 100)
    plt.subplot(3, 1, 2)
    plt.plot(q_episodes_x, q_block_lengths, label='Q-learning', color='blue', linestyle='-', marker='o')
    plt.plot(sarsa_episodes_x, sarsa_block_lengths, label='SARSA', color='red', linestyle='-', marker='x')
    plt.xlabel(f'Episodio (promedio por cada {avg_block_size})')
    plt.ylabel('Número de Pasos Medio')
    plt.title(f'Comparación de Longitud Media de Episodio por cada {avg_block_size} Episodios')
    plt.legend()
    plt.grid(True)
    
    # 3. Gráfica de error de entrenamiento (promedio por bloques de 100)
    plt.subplot(3, 1, 3)
    plt.plot(q_episodes_x, q_block_errors, label='Q-learning', color='blue', linestyle='-', marker='o')
    plt.plot(sarsa_episodes_x, sarsa_block_errors, label='SARSA', color='red', linestyle='-', marker='x')
    plt.xlabel(f'Episodio (promedio por cada {avg_block_size})')
    plt.ylabel('Error TD Promedio')
    plt.title(f'Comparación de Error de Entrenamiento Medio por cada {avg_block_size} Episodios')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('comparacion_sarsa_qlearning_promediado.png', dpi=300)
    plt.show()
    
    print("Experimentos completados y gráficas generadas.")
    
    # estadísticas finales
    print(f"\nResumen de resultados (promedios de los últimos {avg_block_size} episodios):")
    print(f"Q-learning - Recompensa media: {q_block_rewards[-1]:.2f}, Longitud media: {q_block_lengths[-1]:.2f}, Error medio: {q_block_errors[-1]:.4f}")
    print(f"SARSA - Recompensa media: {sarsa_block_rewards[-1]:.2f}, Longitud media: {sarsa_block_lengths[-1]:.2f}, Error medio: {sarsa_block_errors[-1]:.4f}")
    
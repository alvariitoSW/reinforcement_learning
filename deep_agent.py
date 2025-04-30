from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from collections import deque
from tensorflow import keras
import gymnasium as gym
import tensorflow as tf
import numpy as np
import random
import time
import os

# Activar eager execution para debugging, pero usar tf.function para optimizar
tf.config.run_functions_eagerly(False)

class DQNAgent:
    """
    Agente para DQN con enfoque en eficiencia computacional.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hiperparámetros ultra-optimizados
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.97
        self.learning_rate = 0.002
        self.update_target_freq = 20
        
        # Memoria de experiencia reducida
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Contador de entrenamiento optimizado
        self.train_every = 4  
        self.train_counter = 0
        
        # Precalcular estados one-hot
        self.state_map = {}
        for i in range(state_size):
            one_hot = np.zeros(state_size)
            one_hot[i] = 1
            self.state_map[i] = one_hot

        # Modelo principal y objetivo
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Compilar funciones TensorFlow para optimizar
        self._compile_tf_functions()
    
    def _build_model(self):
        """
        Construye una red neuronal DQN optimizada
        """
        model = keras.Sequential([
            Input(shape=(self.state_size,)),
            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        # Configuración de compilación optimizada
        model.compile(
            loss='mse', 
            optimizer=Adam(learning_rate=self.learning_rate),
            run_eagerly=False  # Ejecutar de manera optimizada
        )
        return model
    
    def _compile_tf_functions(self):
        """
        Precompila funciones TensorFlow para mayor velocidad
        """
        # Convertir predict en función optimizada de TensorFlow
        self.predict_model = tf.function(lambda x: self.model(x, training=False))
        self.predict_target = tf.function(lambda x: self.target_model(x, training=False))
    
    def update_target_model(self):
        """Copia los pesos del modelo principal al modelo objetivo"""
        self.target_model.set_weights(self.model.get_weights())
    
    def preprocess_state(self, state):
        """Recupera estado one-hot precalculado"""
        return self.state_map[state]
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena experiencia en la memoria"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Selecciona una acción usando política epsilon-greedy ultra-optimizada"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Usar la función TensorFlow optimizada
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        q_values = self.predict_model(state_tensor)[0].numpy()
        return np.argmax(q_values)
    
    def replay(self):
        """Entrena la red con experiencias de forma ultra-eficiente"""
        # Incrementar contador y comprobar si debemos entrenar en este paso
        self.train_counter += 1
        if self.train_counter % self.train_every != 0:
            return 0
        
        if len(self.memory) < self.batch_size:
            return 0
        
        # Muestrear aleatoriamente experiencias
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Preparar datos en bloque
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Convertir a tensores de TensorFlow para mayor eficiencia
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        
        # Obtener valores Q actuales y próximos de forma optimizada
        targets = self.predict_model(states_tensor).numpy()
        next_q_values = self.predict_target(next_states_tensor).numpy()
        
        # Vectorizar más operaciones
        max_next_q = np.max(next_q_values, axis=1)
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Actualizar solo los valores Q para las acciones tomadas
        for i in range(self.batch_size):
            targets[i, actions[i]] = target_q[i]
        
        # Entrenar en un solo paso
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        loss = history.history['loss'][0]
        
        # Reducir epsilon más rápidamente
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
        
    def load(self, name):
        """Carga pesos del modelo desde archivo"""
        self.model.load_weights(name)
        self.target_model.load_weights(name)  # Asegurar que ambos modelos están sincronizados
        
    def save(self, name):
        """Guarda pesos del modelo en archivo"""
        self.model.save_weights(name)


def train_ultra_optimized_dqn(env_name='FrozenLake-v1', map_name='4x4', is_slippery=False,
                    episodes=500, render=False):
    """
    Entrena un agente DQN ultra-optimizado
    """
    # Crear entorno
    env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery, 
                   render_mode='rgb_array' if render else None)
    
    # Obtener dimensiones
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # Inicializar agente
    agent = DQNAgent(state_size, action_size)
    
    # Listas para resultados
    scores = []
    avg_scores = []
    
    # Límite de pasos reducido drásticamente
    max_steps = 50 if map_name == '4x4' else 100
    
    # Early stopping
    best_avg_score = -np.inf
    patience = 20  # Episodios sin mejora antes de parar
    patience_counter = 0
    min_episodes = 200  # Mínimo de episodios a entrenar
    
    print(f"Entrenando durante máximo {episodes} episodios (con early stopping)...")
    
    for e in tqdm(range(1, episodes + 1)):
        # Reiniciar entorno
        state, _ = env.reset()
        
        # Preprocesar estado
        state_processed = agent.preprocess_state(state)
        
        # Variables para este episodio
        done = False
        score = 0
        step = 0
        
        while not done and step < max_steps:
            # Seleccionar acción
            action = agent.act(state_processed)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Preprocesar siguiente estado
            next_state_processed = agent.preprocess_state(next_state)
            
            # Recompensa simplificada
            modified_reward = reward if reward > 0 else (-1.0 if done else -0.01)
            
            # Guardar experiencia
            agent.remember(state_processed, action, modified_reward, next_state_processed, done)
            
            # Entrenar
            agent.replay()
            
            # Preparar para siguiente iteración
            state_processed = next_state_processed
            score += reward
            step += 1
        
        # Actualizar red objetivo menos frecuentemente
        if e % agent.update_target_freq == 0:
            agent.update_target_model()
            
        # Registrar resultados mínimos
        scores.append(score)
        
        # Promedio de últimos 50 episodios
        window = min(50, len(scores))
        avg_score = np.mean(scores[-window:])
        avg_scores.append(avg_score)
        
        # Mostrar progreso brevemente
        if e % 50 == 0 or e == 1:
            print(f"Episodio {e}/{episodes}, Promedio: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Early stopping
        if e > min_episodes:
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping en episodio {e} - No hay mejora durante {patience} episodios")
                break
    
    env.close()
    
    # Guardar modelo
    agent.save("dqn_frozenlake_ultra.weights.h5")
    
    print(f"Entrenamiento completado en {e} episodios")
    return agent, scores, avg_scores


def simple_plot(scores, avg_scores):
    """
    Gráfica mínima de resultados para ahorrar recursos
    """
    plt.figure(figsize=(8, 4))
    plt.plot(scores, 'b-', alpha=0.3, label='Recompensa')
    plt.plot(avg_scores, 'r-', label='Promedio')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('Entrenamiento DQN Ultra-Optimizado')
    plt.legend()
    plt.savefig('dqn_training_ultra.png', dpi=150)
    plt.close()  # Cerrar figura para liberar memoria


def evaluate_agent(agent, env_name='FrozenLake-v1', map_name='4x4', 
                  is_slippery=False, episodes=50, render=False):
    """
    Evalúa un agente DQN entrenado (versión ultra-rápida)
    """
    # Crear entorno
    env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery, 
                  render_mode='human' if render else None)
    
    # Obtener tamaño del estado
    state_size = env.observation_space.n
    
    scores = []
    success_count = 0
    
    # Añadir pequeña exploración
    eval_epsilon = 0.02
    
    for e in tqdm(range(episodes)):
        state, _ = env.reset()
        state = agent.preprocess_state(state)
        
        done = False
        score = 0
        steps = 0
        max_steps = 50 if map_name == '4x4' else 100
        
        while not done and steps < max_steps:
            # Seleccionar acción
            if np.random.rand() < eval_epsilon:
                action = random.randrange(env.action_space.n)
            else:
                action = agent.act(state, training=False)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = agent.preprocess_state(next_state)
            state = next_state
            score += reward
            steps += 1
            
            if done and reward > 0:
                success_count += 1
            
            # Pausa muy breve para visualización
            if render:
                time.sleep(0.02)
        
        scores.append(score)
    
    env.close()
    
    avg_score = np.mean(scores)
    success_rate = success_count / episodes
    
    print(f"Evaluación: Puntuación media: {avg_score:.2f}, Éxito: {success_rate:.2%} ({success_count}/{episodes})")
    
    return scores, success_rate


def ultra_optimized_main():
    """Función principal ultra-optimizada"""
    # Configuración básica
    env_name = 'FrozenLake-v1'
    map_name = '4x4'
    is_slippery = False
    
    model_file = "dqn_frozenlake_ultra.weights.h5"
    
    state_size = 16 if map_name == '4x4' else 64
    action_size = 4
    
    agent = DQNAgent(state_size, action_size)
    
    # Entrenamiento o carga de modelo en caso de existir
    if os.path.exists(model_file):
        print(f"Cargando modelo existente '{model_file}'...")
        agent.load(model_file)
    else:
        print(f"Iniciando entrenamiento ultra-optimizado...")
        agent, scores, avg_scores = train_ultra_optimized_dqn(
            env_name=env_name,
            map_name=map_name,
            is_slippery=is_slippery,
            episodes=500,
            render=False
        )
        
        agent.save(model_file)
        # Gráfica simple
        simple_plot(scores, avg_scores)
    
    # Evaluación rápida
    fast_eval_scores, fast_eval_success = evaluate_agent(
        agent, env_name, map_name, is_slippery, episodes=10, render=False
    )
    
    # Entrenamiento adicional si es necesario, pero breve
    if fast_eval_success < 0.5:
        print("\nRendimiento bajo. Entrenando brevemente...")
        agent, scores, avg_scores = train_ultra_optimized_dqn(
            env_name=env_name,
            map_name=map_name,
            is_slippery=is_slippery,
            episodes=100,
            render=False
        )
        agent.save(model_file)

    # Evaluación final
    print("\nEvaluando agente final...")
    eval_scores, success_rate = evaluate_agent(
        agent,
        env_name=env_name,
        map_name=map_name,
        is_slippery=is_slippery,
        episodes=50,  # Menos episodios
        render=True
    )

    print(f"Evaluación final: Tasa de éxito: {success_rate:.2%}")

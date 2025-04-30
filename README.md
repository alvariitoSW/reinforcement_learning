# Proyecto de Aprendizaje por Refuerzo: Entrenamiento de Agentes en el Entorno FrozenLake

## Descripción General

Este proyecto implementa y compara diferentes algoritmos de aprendizaje por refuerzo (RL) utilizando el entorno FrozenLake de Gymnasium. Se exploran tres algoritmos fundamentales: Q-learning, SARSA (State-Action-Reward-State-Action) y Deep Q-Learning (DQN), analizando su comportamiento, rendimiento y diferencias en diferentes configuraciones del entorno.

El objetivo principal es comprender los fundamentos del aprendizaje por refuerzo a través de la implementación práctica de algoritmos clásicos y modernos, evaluando su eficacia en entornos con diferentes niveles de complejidad y estocacidad.

## Requisitos y Dependencias

Para ejecutar este proyecto se requieren las siguientes dependencias:

```
numpy
matplotlib
gymnasium
tensorflow (para Deep Q-Learning)
tqdm
imageio
pygame
IPython
```

Puede instalar estas dependencias mediante:

```bash
pip install numpy matplotlib gymnasium tensorflow tqdm imageio pygame
```

## Estructura del Proyecto

El proyecto está organizado en los siguientes archivos:

- **reinforcement_learning.py**: Implementación de los algoritmos Q-learning y SARSA.
- **rl_utils.py**: Utilidades para visualización, generación de episodios y comparación experimental de algoritmos.
- **deep_agent.py**: Implementación de Deep Q-Learning con optimizaciones.
- **Notebooks (.ipynb)**: Cuadernos Jupyter con explicaciones, experimentos y visualizaciones.

## Algoritmos Implementados

### 1. Q-Learning (Off-Policy)

Algoritmo de aprendizaje por refuerzo que aprende la función de valor Q óptima independientemente de la política seguida. La actualización de la función Q se realiza mediante:

```
Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
```

Características:
- **Off-policy**: Aprende la política óptima independientemente de las acciones de exploración tomadas
- Utiliza el valor máximo del siguiente estado para actualizaciones
- Tiende a encontrar rutas más cortas y directas

### 2. SARSA (On-Policy)

Algoritmo que aprende una función de valor Q para la política que está siguiendo actualmente. La actualización se realiza según:

```
Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
```

Características:
- **On-policy**: Aprende la política que está siguiendo, incluyendo acciones exploratorias
- Considera la siguiente acción real que se tomará
- Tiende a ser más conservador, prefiriendo rutas más seguras pero potencialmente más largas

### 3. Deep Q-Learning (DQN)

Extensión de Q-learning que utiliza redes neuronales para aproximar la función Q, permitiendo trabajar con espacios de estados grandes o continuos.

Características implementadas:
- Redes neuronales como aproximadores de función
- Experience replay (memoria de experiencias pasadas)
- Red objetivo separada para estabilidad
- Estrategia epsilon-greedy para exploración-explotación

## Entornos Utilizados

El proyecto utiliza el entorno **FrozenLake-v1** de Gymnasium, un entorno tipo rejilla donde:
- El agente debe navegar desde un punto inicial hasta un objetivo evitando agujeros.
- Se implementan dos variantes:
  - **No resbaladizo**: El agente siempre se mueve en la dirección deseada.
  - **Resbaladizo**: El agente puede deslizarse en direcciones no deseadas (entorno estocástico).
- Se utilizan dos tamaños de rejilla: 4x4 (simple) y 8x8 (más complejo).

## Cómo Ejecutar el Código

1. **Preparación del entorno**:
   ```python
   import gymnasium as gym
   import numpy as np
   from reinforcement_learning import q_learning, sarsa_learning
   import rl_utils
   ```

2. **Creación y configuración del entorno**:
   ```python
   environment = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False, render_mode='rgb_array')
   ```

3. **Entrenamiento de un agente Q-learning**:
   ```python
   Qtable = np.zeros((environment.observation_space.n, environment.action_space.n))
   Qtable = q_learning(
       environment, 
       n_training_episodes=1000,
       max_steps=100,
       learning_rate=0.7,
       gamma=0.95,
       min_epsilon=0.05, 
       max_epsilon=1.0, 
       decay_rate=0.0005,  
       Qtable=Qtable,
   )
   ```

4. **Visualización de un episodio**:
   ```python
   frames = rl_utils.generate_greedy_episode(environment, Qtable)
   rl_utils.show_episode(frames, interval=250)
   ```

5. **Comparación de algoritmos**:
   ```python
   rl_utils.run_experiments(environment, q_learning_param, sarsa_param)
   ```

6. **Para ejecutar DQN**:
   ```python
   from deep_agent import ultra_optimized_main
   ultra_optimized_main()
   ```

## Resultados y Comparaciones

El proyecto analiza y compara los algoritmos en términos de:

### 1. Recompensa de Episodio
Mide la recompensa total acumulada durante un episodio. Los resultados muestran que SARSA tiende a obtener recompensas más altas que Q-learning, especialmente en entornos estocásticos, debido a su naturaleza más conservadora.

### 2. Longitud de Episodio
Mide el número de pasos hasta completar un episodio. Q-learning tiende a encontrar rutas más cortas (aproximadamente 15 pasos) mientras que SARSA opta por trayectorias más largas pero seguras (35-40 pasos).

### 3. Error de Entrenamiento
Mide la diferencia entre las predicciones del agente y los valores objetivo. SARSA muestra errores más altos, indicando una adaptación continua a la política actual, mientras que Q-learning converge más rápidamente a valores estables.

### Conclusiones Principales:

- **Q-learning**: Prioriza rutas cortas y directas, pero puede ser menos estable en entornos estocásticos.
- **SARSA**: Prefiere rutas seguras aunque sean más largas, adaptándose mejor a entornos con riesgo.
- **DQN**: Demuestra capacidad para manejar entornos más complejos, aunque requiere más recursos computacionales.

## Aplicaciones y Extensiones

El proyecto puede extenderse para:
- Implementar otros algoritmos como Expected SARSA o algoritmos basados en política (Policy Gradient)
- Aplicar estos algoritmos a entornos más complejos como Atari o entornos continuos
- Explorar técnicas avanzadas como aprendizaje por refuerzo profundo con representación de estado mediante redes convolucionales

## Referencias

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
- Documentación de Gymnasium: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- Tutoriales de Gymnasium: [https://gymnasium.farama.org/introduction/basic_usage/](https://gymnasium.farama.org/introduction/basic_usage/)
- Hasselt, H. V., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1).
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. International Conference on Learning Representations (ICLR).
- Singh, S., Jaakkola, T., Littman, M. L., & Szepesvári, C. (2000). Convergence results for single-step on-policy reinforcement-learning algorithms. Machine Learning, 38(3), 287-308.

## Autores

Este proyecto es parte de un laboratorio para una asignatura académica. Ha sido implementado por estudiantes con la guía de instructores académicos. Las bases del código han sido adaptadas de recursos educativos de aprendizaje por refuerzo.

---

**Nota**: Este proyecto tiene fines educativos y busca facilitar la comprensión de los conceptos fundamentales del aprendizaje por refuerzo a través de implementaciones prácticas y visuales.
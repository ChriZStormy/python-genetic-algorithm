import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def flip1(individuo):
    idx = np.random.randint(0, len(individuo))
    individuo[idx] = 1 - individuo[idx]
    return individuo

def un_punto(p1, p2):
    n = len(p1)
    punto = np.random.randint(1, n-1)
    h1 = np.concatenate([p1[:punto], p2[punto:]])
    h2 = np.concatenate([p2[:punto], p1[punto:]])
    return h1, h2

def dos_puntos(p1, p2):
    n = len(p1)
    p1_, p2_ = sorted(np.random.choice(range(1, n-1), 2, replace=False))
    h1 = np.concatenate([p1[:p1_], p2[p1_:p2_], p1[p2_:]])
    h2 = np.concatenate([p2[:p1_], p1[p1_:p2_], p2[p2_:]])
    return h1, h2

def torneo_binario(pop, fitness):
    i, j = np.random.choice(len(pop), 2, replace=False)
    return pop[i] if fitness[i] < fitness[j] else pop[j]

def ruleta(pop, fitness):
    inv_fit = 1 / (1 + np.array(fitness))
    probs = inv_fit / np.sum(inv_fit)
    idx = np.random.choice(len(pop), p=probs)
    return pop[idx]

def create_pop(n_individuos, n_bits, dim):
    total_bits = n_bits * dim 
    return np.random.randint(0, 2, (n_individuos, total_bits))


def binario_a_real(pop, n_vars, num_bits, li, ls):
    n_individuos = len(pop)
    pop_real = np.zeros((n_individuos, n_vars))

    for i in range(n_individuos):
        for v in range(n_vars):
       
            inicio = v * num_bits
            fin = (v + 1) * num_bits
            bits = pop[i, inicio:fin]
        
            decimal = int("".join(map(str, bits)), 2)
   
            real = li + (decimal / (2**num_bits - 1)) * (ls - li)
            pop_real[i, v] = real
    
    return pop_real
 

def sphere(x):
    return np.sum(x**2)

def ackley(x):
    x = np.array(x)
    a, b, c = 20, 0.2, 2*np.pi
    d = len(x)
    return -a * np.exp(-b*np.sqrt(np.sum(x**2)/d)) - np.exp(np.sum(np.cos(c*x))/d) + a + np.e

def rastrigin(x):
    x = np.array(x)
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def sphere_function(x):
    x = np.array(x)
    return np.sum(x**2)

def bukin6(x):
    x1, x2 = x[:2]
    return 100 * np.sqrt(abs(x2 - 0.01*x1**2)) + 0.01 * abs(x1+10)

def cross_in_tray(x):
    x1, x2 = x[:2]
    fact = np.exp(abs(100 - np.sqrt(x1**2 + x2**2)/np.pi))
    return -0.0001 * (abs(np.sin(x1)*np.sin(x2)*fact) + 1)**0.1

def himmelblau(x):
    x1, x2 = x[:2]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def eggholder(x):
    x1, x2 = x[:2]
    return -(x2+47)*np.sin(np.sqrt(abs(x1/2 + (x2+47)))) - x1*np.sin(np.sqrt(abs(x1-(x2+47))))



def algoritmo_genetico(config):
    num_bits = config["num_bits"]
    pop_size = config["pop_size"]
    generaciones = config["generaciones"]  
    dim = config["dim"]
    funcion_obj = config["funcion_obj"]
    seleccion = config["seleccion"]
    cruza = config["cruza"]
    mutacion = config["mutacion"]
    elitismo = config["elitismo"]
    li = config["limite_inferior"]
    ls = config["limite_superior"]
    
    pop = create_pop(pop_size, num_bits, dim)
    pop_real = binario_a_real(pop, dim, num_bits, li, ls)
    fitness = [funcion_obj(ind) for ind in pop_real]
    best_fitness = []
    best_solution = pop_real[np.argmin(fitness)].copy()
    
    history = []  # guardamos todas las poblaciones
    best_history = []  # guardamos el mejor de cada generación
    for g in range(generaciones):
        nueva_poblacion = []
        
        if elitismo:
            idx_best = np.argmin(fitness)
            mejor_individuo = pop[idx_best].copy()
            nueva_poblacion.append(mejor_individuo)
        
        while len(nueva_poblacion) < pop_size:
            p1 = seleccion(pop, fitness)
            p2 = seleccion(pop, fitness)
            h1, h2 = cruza(p1, p2)
            h1 = mutacion(h1)
            h2 = mutacion(h2)
            nueva_poblacion.append(h1)
            if len(nueva_poblacion) < pop_size:
                nueva_poblacion.append(h2)
        
        pop = np.array(nueva_poblacion)
        pop_real = binario_a_real(pop, dim, num_bits, li, ls)
        fitness = [funcion_obj(ind) for ind in pop_real]

        idx_best = np.argmin(fitness)
        if fitness[idx_best] < funcion_obj(best_solution):
            best_solution = pop_real[idx_best].copy()
        best_fitness.append(funcion_obj(best_solution))
        history.append(pop_real.copy())
        best_history.append(best_solution.copy())
    
    return best_solution, best_fitness, history, best_history
    
config = {
    "num_bits": 12,         
    "pop_size": 30,
    "generaciones": 100,
    "dim": 3,              
    "funcion_obj": sphere,  
    "seleccion": ruleta,
    "cruza": dos_puntos,
    "mutacion": flip1,
    "elitismo":0,
    "limite_superior" : 10,
    "limite_inferior" : -10       
}

b_s, b_f, history, best_history = algoritmo_genetico(config)

print("Mejor solución:", b_s)
print("Fitness:", b_f[-1])

fig, ax = plt.subplots()
ax.set_xlim(config["limite_inferior"], config["limite_superior"])
ax.set_ylim(config["limite_inferior"], config["limite_superior"])
ax.set_title(f"Algoritmo Genético sobre {config['funcion_obj'].__name__}")

scat = ax.scatter([], [], c="blue", s=30, alpha=0.6)
best_marker, = ax.plot([], [], marker="*", color="red", markersize=15, label="Mejor")

def update(frame):
    pop = history[frame]
    best = best_history[frame]
    scat.set_offsets(pop[:, :2])  
    best_marker.set_data([best[0]], [best[1]])  
    return scat, best_marker

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=100, blit=True, repeat=False)
ax.legend()
plt.show()
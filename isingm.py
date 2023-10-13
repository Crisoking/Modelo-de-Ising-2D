#%% LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

#%% PARÁMETROS DEL MODELO DE ISING 2D
# Parámetros de la simulación
num_steps = 300000
intervalo_muestreo = 100 # Número de pasos entre cada muestreo
Temperatura = np.arange(0.5,5,0.05) # Valores de temperatura
datos_por_temperatura = [] # Parámetros obtenidos para cada temperatura
muestras = [] # Lista de muestras de la red
energias = [] # Lista de energías
J=1.0
N=20
tam_red= [N,N]

#%% INICIALIZACIÓN DE LA RED
config_espin = np.random.choice([-1,1], size=tam_red)

def energia(config):
    # Cálculo de la energía de la configuración de la red
    return -J * np.sum(config * (np.roll(config, 1, axis=0) + 
                                 np.roll(config, 1, axis=1)))

def metropolis_hastings_algoritmo(config, beta):
    # Algoritmo de Metropolis-Hastings
    # beta = 1 / (kb * T), kb = constante de Boltzmann
    i, j = np.random.randint(0, tam_red[0]), np.random.randint(0, tam_red[1])
    delta_energia = 2 * J * config[i, j] * (config[(i - 1) % tam_red[0], j] +
                                            config[(i + 1) % tam_red[0], j] +
                                            config[i, (j - 1) % tam_red[1]] +
                                            config[i, (j + 1) % tam_red[1]]
    )
    if delta_energia < 0 or np.random.random() < np.exp(-beta * delta_energia):
        config[i, j] *= -1 # Cambia el espín

#%% SIMULACIÓN PARA ENERGÍA VS TIEMPO DE SIMULACIÓN
# Simulación de las muestras de la red

Beta = 1/ (J * 2) # Factor beta para la temperatura T = 2.6

for step in range(num_steps):
    metropolis_hastings_algoritmo(config_espin, Beta)
    
    # Cálculo de la energía de la red en cada paso
    energia_actual = energia(config_espin)
    energias.append(energia_actual)
    
    # Muestra de la configuración de la red en el intervalo especificado
    if step % intervalo_muestreo == 0:
        muestras.append(config_espin.copy())

# Cálculo de la energía promedio
energia_promedio = np.mean(energias)

print("Energía Promedio:", energia_promedio)

# Gráfica de la energía vs tiempo de simulación
time_steps = range(0, num_steps)
plt.plot(time_steps, energias, label='Energía')
plt.xlabel('Tiempo de simulación', fontsize=15)
plt.ylabel('Energía', fontsize=15)
plt.minorticks_on()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
plt.show()

#%% SIMULACIÓN PARA MAGNETIZACIÓN PROMEDIO VS TEMPERATURA

magnetizaciones = [] # Lista de magnetizaciones en cada paso para la temperatura T

for T in Temperatura:
    beta = 1 / (J * T) # Cálculo de beta para la temperatura T

    # Reset de la configuración de la red para cada temperatura
    config_espin = np.random.choice([-1, 1], size=tam_red)

    # Simulación y estabilización de la red
    magn_step=[]
    for step in range(num_steps):
        metropolis_hastings_algoritmo(config_espin, beta)
        # Cálculo de la magnetización de la red en cada paso
        magnetizacion_actual = np.sum(config_espin)
        magn_step.append(magnetizacion_actual)

    magnetizaciones.append(magn_step)
    
magn_promedio=[]
for i in magnetizaciones:
    magn_promedio.append(i[-100000].mean()/(N**2))
    
# Gráfica de la magnetización promedio vs temperatura
plt.scatter(Temperatura, magn_promedio, marker='o', s= 10)
plt.xlabel("Temperatura", fontsize=15)
plt.ylabel("Magnetización promedio", fontsize=15)
plt.title("Magnetización promedio vs temperatura", fontsize=15)
plt.minorticks_on()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

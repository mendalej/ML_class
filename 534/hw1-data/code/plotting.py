import pandas as pd
import matplotlib.pyplot as plt

k_values = list(range(1, 100, 2))
smart = [
    23.9, 19.5, 18.4, 16.6, 15.8, 16.3, 16.5, 15.8, 16.0, 16.6,
    16.3, 15.5, 15.4, 15.7, 15.0, 15.1, 15.6, 15.0, 14.8, 14.7,
    14.3, 14.8, 15.0, 15.1, 15.3, 15.7, 15.3, 15.4, 15.6, 15.6,
    15.5, 15.4, 15.5, 15.5, 15.2, 15.6, 15.2, 14.9, 15.0, 15.1,
    15.3, 15.5, 15.5, 15.4, 15.5, 15.5, 15.2, 15.3, 15.5, 15.8
]
smartscaling = [
    26.9, 24.0, 23.4, 23.3, 22.2, 21.9, 21.7, 21.5, 22.1, 21.4, 
    21.7, 22.3, 22.1, 22.0, 21.3, 21.4, 20.7, 21.3, 21.7, 21.8, 
    21.8, 22.3, 22.3, 22.7, 22.1, 22.4, 22.3, 21.8, 21.9, 
    22.4, 22.7, 22.6, 22.2, 22.2, 22.9, 22.1, 22.5, 22.0, 22.3, 
    21.9, 22.4, 21.9, 21.9, 22.2, 22.6, 22.5, 22.8, 22.4, 22.5, 
    23.1
]
naive= [
    21.2, 18.7, 18.3, 16.7, 16.3, 16.6, 16.2, 15.9, 15.7, 15.4, 
    16.0, 16.4, 16.4, 16.8, 16.4, 15.8, 16.2, 16.4, 16.4, 16.2, 
    16.4, 15.8, 15.5, 16.1, 15.9, 16.1, 16.1, 16.2, 16.0, 16.1, 
    16.1, 15.4, 15.2, 16.0, 15.9, 15.9, 16.0, 16.0, 15.8, 16.0, 
    16.1, 16.0, 16.2, 15.6, 16.1, 15.7, 16.0, 16.0, 
    16.2, 16.1
]


plt.figure(figsize=(10, 6))
plt.plot(k_values, smartscaling, label='Smart + Scaling', color='blue', linestyle='dashed', marker='o')
plt.plot(k_values, smart, label='Smart', color='purple', linestyle='dashed', marker='o')
plt.plot(k_values, naive, label='Naive',color='green',linestyle='dashed', marker='o')

plt.xlabel('k')
plt.ylabel('Dev Error Rate')
plt.title('KNN Dev Error Rate vs. k')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

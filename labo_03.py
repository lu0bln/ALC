import numpy as np
import math
import matplotlib.pyplot as plt

"EJERCICIO 1"
"a)Una funcion norma(x,p) que reciba un vector x (un objeto iterable) y una norma p y retorne su norma."

def norma(vector,p):
    res = 0
    for i in range(0,vector.size):
        res += vector[i] ** p
    res = res ** (1/p)
    return res

x = np.array([1,2,3])
#print(norma(x,1))

"b)Una funcion normaliza(X,p) que reciba una lista de vectores X y una norma p y retorne una lista Y donde cada elemento corresponde a normalizarlos elementos de X con la norma p"
def normaliza(matriz,n):
    Y = []
    for i in range(0,matriz.shape[0]): # quiero que itere sobre filas
        fila_n = []
        norma_vector = norma(matriz[i],n)
        for j in range(0,matriz.shape[1]):
            fila_n.append(matriz[i][j] * 1/(norma_vector))
        Y.append(fila_n)
    return np.array(Y)

M = np.array([[1,2,3],[4,5,6]])
print(normaliza(M,2))

print(norma(normaliza(M,2)[0],2))

p = [1,2,5,10,100,200]

plt.plot ([0,1,0,-1,0],[1,0,-1,0,1],marker="*")
#plt.show()

np.random.randn(200,2)



## EJERCICIO 4 (P1)
# ...
# Aca, crear la matriz y resolver el sistema para calcular a,b y c.
# Obtuve que:
a = -3/2
b = 11/2
c= -3
# ...
xx = np.array ([1 ,2 ,3])
yy = np.array ([1 ,2 ,0])
x = np. linspace(0,4,100) #genera 100 puntos equiespaciados entre 0 y 4.
f = lambda t: a*t**2+b*t+c #esto genera una funcion f de t.
plt . plot (xx ,yy, '*' )
plt . plot (x, f (x))
#plt .show()
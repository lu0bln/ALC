import numpy as np

## EJERCICIO 3 (P1)
1 + 3
a = 7
b = a + 1
print ("b = ", b)
# Vectores
v = np.array([1,2 ,3 ,-1]) 
w = np.array([2 ,3 ,0 ,5])
print ("v + w = ", v + w)
print ("2∗v = ", 2*v)
print ("v∗∗2 = ", v**2)
# Matrices (ejecutar los comandos uno a uno para ver los resultados)
A = np.array([[1 ,2 ,3 ,4 ,5] ,[0 ,1 ,2 ,3 ,4] ,[2 ,3 ,4 ,5 ,6] ,[0 ,0 ,1 ,2 ,3] ,[0 ,0 ,0 ,0 ,1]])
print (A)
print (A[0].size)
A[0:2 ,3:5]
A[:2 ,3:]
A[[0 ,2 ,4] ,:]
ind = np.array ([0 ,2 ,4])
A[ ind , ind ]
A[ ind , ind [: ,None ]]
# Numeros complejos
1j *1j
(1+2j )*1j

## EJERCICIO 4 (P1)


## EJERCICIO 21 (P1) <-- HACER EN ARCHIVO A PARTE!

## FUNCIONES REUTILIZABLES

def filas(matriz) -> int:
    if (np.array(matriz).size == 0):
        return 0
    else:
        return np.array(matriz).size//columnas(matriz)

def columnas(matriz) ->int:
    if (np.array(matriz).size == 0):
        return 0
    else:
        return np.array(matriz[0]).size

## EJERCICIO 1
def esCuadrada(matriz):
    res = False
    if (filas(matriz) == columnas(matriz) and matriz.size !=0):
        res = True
    return res

matriz_test = np.array([[1,2,3],[2,5,4],[3,4,7]])
print(esCuadrada(matriz_test))
print(matriz_test[0])

## EJERCICIO 2
def triangSup(matriz):
    f = filas(matriz)
    c = columnas(matriz)
    if (c < 2 and f > 1):
        return matriz
    U = []
    #Copio la matriz original a U
    for i in range(0,f):
        U.append(matriz[i].copy()) 
    for i in range(0,f):
        for j in range(0,c):
            if (i >= j):
                U[i][j] = 0
    return np.array(U)
  
print("triangular superior:")
print(triangSup(matriz_test))
print("y esta es la original:")
print(matriz_test)

## EJERCICIO 3
def triangInf(matriz):
    f = filas(matriz)
    c = columnas(matriz)
    if (c < 2 and f > 1):
        return matriz
    L = []
    #Copio la matriz original a U
    for i in range(0,f):
        L.append(matriz[i].copy()) 
    for i in range(0,f):
        for j in range(0,c):
            if (i <= j):
                L[i][j] = 0
    return np.array(L)

print("triangular inferior:")
print(triangInf(matriz_test))
print("y esta es la original:")
print(matriz_test)

## EJERCICIO 4
def diagonal(matriz):
    f = filas(matriz)
    c = columnas(matriz)
    if (c < 2 and f > 1):
        return matriz
    D = []
    #Copio la matriz original a U
    for i in range(0,f):
        D.append(matriz[i].copy()) 
    for i in range(0,f):
        for j in range(0,c):
            if (i != j):
                D[i][j] = 0
    return np.array(D)

print("diagonal:")
print(diagonal(matriz_test))
print("y esta es la original:")
print(matriz_test)

## Ejercicio 5
def traza(matriz):
    traza = 0
    i = 0
    while i < min(filas(matriz),columnas(matriz)):
        traza += matriz[i][i]
        i += 1
    return traza

print("traza:")
print(traza(matriz_test))
print("y esta es la original:")
print(matriz_test)

## EJERCICIO 6
def traspuesta(matriz):
    T = []
    f = filas(matriz)
    c = columnas(matriz)
    for i in range(0,f):
        for j in range(0,c):
            if (len(T) < c):
                T.append([matriz[i][j].copy()])
            else:
                T[j].append(matriz[i][j].copy())
    return np.array(T)

print("Traspuesta:")
print(traspuesta(matriz_test))
print("y esta es la original:")
print(matriz_test)

## EJERCICIO 7 (Solo valida para matrices cuadradas!!)
def esSimetrica(matriz):
    res = True
    A_t = traspuesta(matriz)
    if (not esCuadrada(matriz)):
        res = False
    else:
        for i in range(0,filas(matriz)):
            for j in range(0,columnas(matriz)):
                if(A_t[i][j] != matriz[i][j]):
                    res = False
    return res

print("La matriz es simetrica:")
print(esSimetrica(matriz_test))

## EJERCICIO 8 (MULTIPLICACION VECTORIAL MATRIZ . VECTOR )
# Recibo matriz de tamano nxm y un vector x de tamano m; calcularAx devuelve vector b de tamano n 
#filaxcolumna
def calcularAx(matriz,vector):
    
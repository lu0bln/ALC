import numpy as np

def norma(vector,p):
    res = 0
    for i in range(0,vector.size):
        res += vector[i] ** p
    res = res ** (1/p)
    return res


def traspuesta(matriz):
    T = []
    for i in range(0,matriz.shape[0]):
        for j in range(0,matriz.shape[1]):
            if (len(T) < matriz.shape[1]):
                T.append([matriz[i][j].copy()])
            else:
                T[j].append(matriz[i][j].copy())
    return np.array(T)

#EJERCICIO 1
def matriz_ceros(filas,colummnas):
    M = []
    for i in range(0,filas):
        M.append([0])
        for j in range(0,colummnas-1):
            M[i].append(0)
    return np.array(M)

def QR_con_GS(A):
    cant_ops = 0
    Q = matriz_ceros(A.shape[0],A.shape[1]).astype(float)
    R = matriz_ceros(A.shape[1],A.shape[1])
    Q[:,0] = A[:,0]*(1/np.linalg.norm(A[:,0]))
    

    return Q

test = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(QR_con_GS(test))


def normaliza(matriz,n):
    Y = []
    for i in range(0,matriz.shape[0]): # quiero que itere sobre filas
        fila_n = []
        norma_vector = norma(matriz[i],n)
        for j in range(0,matriz.shape[1]):
            fila_n.append(matriz[i][j] * 1/(norma_vector))
        Y.append(fila_n)
    return np.array(Y)

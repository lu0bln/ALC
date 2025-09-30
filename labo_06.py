import numpy as np

A_test = np.array([[1,2,3],[4,5,6]])
B_test = np.array([[1],[2],[3]])

def matriz_ceros(filas,colummnas):
    M = []
    for i in range(0,filas):
        M.append([0])
        for j in range(0,colummnas-1):
            M[i].append(0)
    return np.array(M)

def multiplicar_matrices(A,B):
    C = matriz_ceros(A.shape[0],B.shape[1])
    for i in range(0,A.shape[0]):
        for j in range(0,B.shape[1]):
            n = 0
            for k in range(0,B.shape[0]):
                n += A[i][k]*B[k][j]
            C[i][j] = n
    return np.array(C)

print(f"Multiplicar la matriz A =\n{A_test} \npor la matriz B=\n{B_test}\nda como resultado C=\n{multiplicar_matrices(A_test,B_test)} ")

""" EJERCICIO 1"""
# A)


# B)
def metpot2k(A,tol=1e-15,K=1000):
    v = np.random.rand(A.shape[1],1)
    v*_ = 2



import numpy as np
####### librerias.py ##########
def norma(x,p):
    res = 0
    for i in range(0,x.size):
        res += abs(x[i]) ** p
    res = res ** (1/p)
    return res

def normaliza(X,p):
    Y = np.array(X)
    for i in range(len(X)):
        Y[i] = X[i] * 1/norma(X[i],p)
    return Y.T  # pues ingreso como .T, para devolverla como antes
##############################}

def diagRH(A,tol =1e-15 ,K=1000):
    A = np.array(A)
    S = np.zeros(A.shape(0),A.shape(1)) #Matriz de autovectores
    D = np.ones(A.shape(0),A.shape(1)) #Matriz de autovalores
    return S,D

def svd_reducida(A,tol=1e-15):
    A = np.array(A)
    A_t = A.T
    if A.shape(1) > A.shape(0): #Cuando columnas>filas
        #Obtener la matriz V' (de nxr) primero
        hat_V,hat_E = diagRH(A_t@A)
        hat_U = normaliza((A@hat_V).T,2) # U se obtiene normalizando las columnas de B = A@hat_V       
    else: #Cuando filas>= columnas
        #Obtener la matriz U' primero, y hacer el mismo procedimiento pero sobre A_t
        hat_U,hat_E = diagRH(A_t@A) #Duda: A_t@A o A@A_t?
        hat_V = normaliza((A_t@hat_V).T,2) #Mismo proc que antes pero con A_t
    return hat_U,hat_E,hat_V


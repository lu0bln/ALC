import numpy as np
import librerias as lib
"----------------LABORATORIO 01 --------- NUMEROS DE MAQUINA"

def error(x,y):
    return abs(np.float64(x)-np.float64(y))

def error_relativo(x,y):
    if x == 0:
        print("infinito")
        return np.inf # Representacion del infinito
    else: 
        return error(x,y)/abs(x)
    
def sonIguales(x,y,atol=1e-08): # Funcion de regalo en los test :)
    return np.allclose(error(x,y),0,atol=atol)

def matricesIguales(A,B):
    A = np.array(A) # Convierto a A y B en matrices numpy por si vienen listas de listas
    B = np.array(B)
    res = True
    if (A.shape[0] == B.shape[0] and A.shape[1] == B.shape[1]):
        for f in range(0,A.shape[0]):
            for c in range(0,B.shape[1]):
                if not sonIguales(A[f][c],B[f][c]):
                    res = False
    else:
        res = False # Cuando las matrices no tengan las mismas dimensiones, no seran iguales de entrada...
    return res

"--------------LABORATORIO 02-------------TRANSFORMACIONES LINEALES---------------------"

def rota(theta):
    matriz_r = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return matriz_r

#print(rota(np.pi/2))

def escala(s):
    matriz_escala = np.zeros((len(s),len(s)))
    for i in range(len(s)):
            matriz_escala[i,i] = s[i]
    return matriz_escala

def rota_y_escala(theta,s):
    return lib.multiplicar_matrices(rota(theta),escala(s))

def afin(theta,s,b):
    matriz_afin = np.eye(3) # matriz identidad
    matriz_re = rota_y_escala(theta,s)
    matriz_afin[:2,:2] = matriz_re  #de la fila 0 a 1, columna 0 a 1, le asigno a la matriz rota y escala
    matriz_afin[:2,2] = b #de la fila 0 a 1, en la columna 3, asigno el vector b
    return matriz_afin

def trans_afin(v,theta,s,b):
    nuevo_v = np.ones(3) # creo vector de 1's = [1,1,1]
    nuevo_v[:2] = v # donde nuevo_v sera = [v1,v2,1] para poder realizar la multiplicacion matricial
    w = lib.calcularAx(afin(theta,s,b),nuevo_v)  # arreglar calcularAx para que acepte vectores fila tambn
    return w[:2]

print(trans_afin(np.array([1,0]) , np.pi/2,[3,2] ,[4,5]))

"------------LABO O3-----------NORMAS Y NC--------------"
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
    return Y

def normaMatMC(A,q,p,Np):
    x_max = 0
    max_norma = 0
    for _ in range(Np):
        xs = np.random.randn(A.shape[1]) #Genera vectores aleatorios de tamano n 
        xs = normaliza(xs,p)    #Normalizo xs
        ys = lib.calcularAx(A,xs)   # Hago A*xs
        y_norma = norma(ys,q)   #Calculo la norma de A*xs en q
        if y_norma > max_norma: #Me fijo si la nueva norma es mayor al maximo actual y lo guardo
            x_max = ys
            max_norma = y_norma
    return x_max, max_norma

def normaExacta(A,p=[1,'inf']):
    A = np.array(A)
    max_list = []
    if p!= 1 or p!= 'inf':
        return None
    elif p == 'inf':    #Si me piden la norma inf sumo los valors abs de cada vector vi y los voy agregando a una lista para luego obtener el max de ellos
        for i in range(A.shape[0]):
            max.append(sumatoria_fila(A[i]))
    elif p == 1:        #Si me piden la norma 1 hago lo mismo que antes pero a la matriz traspuesta, asi es mas facil sumar las columnas como si fueran filas.
        A = A.T
        for i in range(A.shape[0]):
            max.append(sumatoria_fila(A[i]))
    max = max(max_list)
    return max

"Devuelve el numero de condicion de A usando la norma inducida p"
def condMC(A,p):
    A = np.array(A)
    A_ = np.linalg.solve(A,np.eye(A.shape[0]))
    k = normaMatMC(A,p,p,1000)[0] * normaMatMC(A_,p,p,1000)[0] # Np = 1000 Que valor deberia tener? Aleatorio?
    return k

"Que devuelve el numero de condicion de A a partir de la formula de la ecuacion (1) usando la norma p."
def condExacto(A, p) :
    A = np.array(A)
    A_ = np.linalg.solve(A,np.eye(A.shape[0]))
    k = normaExacta(A,p) * normaExacta(A_,p)
    return k

## Defino funcion para sumar filas (reutilizo)
def sumatoria_fila(x):
    res = 0
    for i in range(len(x)):
        res += abs(x[i])
    return res

"------------LABO O4-----------FACTORIZACION LU--------------"
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

#print(trans_afin(np.array([1,0]) , np.pi/2,[3,2] ,[4,5]))

"------------LABO O3-----------NORMAS Y NC--------------"
## Defino funcion para sumar filas (reutilizo)
def sumatoria_fila(x):
    res = 0
    for i in range(len(x)):
        res += abs(x[i])
    return res

def abs_vector(x):
    y=[]
    for elemento in x:
        y.append(abs(elemento))
    return np.array(y)

def norma(x,p):
    x = np.array(x)
    res = 0
    if p == 'inf':
        return max(abs_vector(x))
    for i in range(0,x.size):
        res += abs(x[i]) ** p
    res = res ** (1/p)
    return res

def normaliza(X,p):
    Y = []
    for i in range(len(X)):
        Y.append(X[i] * 1/norma(X[i],p))
    return Y

def normaMatMC(A,q,p,Np):
    A = np.array(A)
    x_max = 0
    max_norma = 0
    for _ in range(Np):
        xs = np.random.randn(A.shape[1]) #Genera vectores aleatorios de tamano n 
        xs = xs / norma(xs,p)  #Normalizo xs
        ys = A@xs.T   # Hago A*xs <------------- ARREGLAR FUNCIONES DE librerias.py calcularAx() por @ y traspuesta() por .T
        y_norma = norma(ys,q)   #Calculo la norma de A*xs en q
        if y_norma > max_norma: #Me fijo si la nueva norma es mayor al maximo actual y lo guardo
            x_max = ys
            max_norma = y_norma
    return max_norma,x_max
#print(normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)[0])

def normaExacta(A,p=[1,'inf']):
    e = [1,'inf']
    A = np.array(A)
    normas = []
    if p != e and p not in e: #Agrego que si p no esta en e (pues sino falla al ser 1 o inf)
        return None
    for norma in e:
        maximos=[]
        if norma == 'inf':    #Si me piden la norma inf sumo los valors abs de cada vector vi y los voy agregando a una lista para luego obtener el max de ellos
            for i in range(A.shape[0]):
                maximos.append(sumatoria_fila(A[i]))
            normas.append(max(maximos))
        if norma == 1:        #Si me piden la norma 1 hago lo mismo que antes pero a la matriz traspuesta, asi es mas facil sumar las columnas como si fueran filas.
            At = A.T
            for i in range(At.shape[0]):
                maximos.append(sumatoria_fila(At[i]))
            normas.append(max(maximos))
    return normas

#print((normaExacta(np.array([[1,-2],[-3,-4]]))))

"Devuelve el numero de condicion de A usando la norma inducida p"
def condMC(A,p):
    A = np.array(A)
    A_ = np.linalg.solve(A,np.eye(A.shape[0])) # Se puede usar el np.linalg.solve? Es para calcular la inversa...
    k = normaMatMC(A,p,p,1000)[0] * normaMatMC(A_,p,p,1000)[0] # Np = 1000 Que valor deberia tener? Aleatorio?
    return k

"Que devuelve el numero de condicion de A a partir de la formula de la ecuacion (1) usando la norma p."
def condExacta(A, p) :
    A = np.array(A)
    A_ = np.linalg.solve(A,np.eye(A.shape[0]))
    if p == 1:
        k = normaExacta(A,p)[0] * normaExacta(A_,p)[0]
    elif p == 'inf':
        k = normaExacta(A,p)[1] * normaExacta(A_,p)[1]
    return k


"------------LABO O4-----------FACTORIZACION LU--------------"


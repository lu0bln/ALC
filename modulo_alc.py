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

def calculaLU(A):
    if A is None:
        return None,None,0
    
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return None,None,0
    ## desde aqui -- CODIGO A COMPLETAR
    for i in range(n):
        fila_i = Ac[i][i:n] # guardo la primera fila (la fila i que solo es usada como "pivote", no es cambiada solo se usa para calcular)
        for j in range(i+1,m):
            fila_j = Ac[j][i:n]
            if fila_i[0] == 0:
                print("No se puede hacer descomposicion LU pivote nulo")
                return None,None,0  # Si fui triangulando y tuve nops>0, pero justo luego me topo con un pivote nulo -> nops == 0 ? (segun el test es asi)
            elif fila_j[0] == 0:
                continue
            pivote = fila_j[0] / fila_i[0]
            cant_op+=1
            Ac[j][i:n] = fila_j - (pivote)*fila_i
            cant_op+= 2*len(fila_j)-2
            fila_j[0] = pivote

    L = lib.triangInf(Ac) + np.eye(A.shape[0],A.shape[1])
    U = lib.triangSup(Ac) 
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre la matriz Ac
    return L, U, cant_op

def esTriangSup(L):
    res = True
    for i in range(0,L.shape[0]):
        for j in range(0,L.shape[1]):
            if (i > j and L[i][j] != 0):
                res = False
    return res


def res_tri(L,b,inferior = True):
    n = L.shape[0]
    x = np.zeros(n)
    if esTriangSup(L):  # Por los tests, si me dan inferior = False no basta, tengo que verificar si es o no triangSup pues cambia todo
        for i in range(n-1,-1,-1):
            x_i = b[i]
            for j in range(i+1,n):  #Cambie los indices pues no depende del tamano de x, es mejor asi
                x_i -= L[i,j]*x[j]
            x[i] = (x_i*1/L[i,i])
        return np.array(x)
    if inferior == False:   # Por los test, si me dan inferior = False pero es triangInf -> lo resuelvo normal,
        L = lib.traspuesta(L)
    for i in range(0,n):
        x_i = b[i]
        for j in range(i):      #Cambie los indices pues no depende del tamano de x, es mejor asi
            x_i -= L[i,j]*x[j]
        x[i] = (x_i*1/L[i,i])
    return np.array(x)

def inversa(A):
    L, U, nops = calculaLU(A)
    n = A.shape[0] 
# Teniendo L y U sabemos que det(A) = det(L).det(U) donde L es triang inf con 1s en la diagonal y U triang sup -> det(L) = 1 y det(U) = U11...Unn
    for i in range(A.shape[0]): 
        if U[i,i] == 0: # -> Si algun elem de la diag(U) == 0 -> det(U) == 0 y A no es inversible  
            print("La matriz no es inversible")
            return None
# Si A es inversible -> Creamos una matriz identidad (que sera nuestra x_i en el procedimiento)    
    matriz_id = np.eye(n,n)
# Tambien para ir armando la A_inv cada sol x_i sera guardada como columna de A_i 
    A_inv = np.zeros((n,n))
    for i in range(n):
        y = res_tri(L,matriz_id[i],inferior=True)
        x = res_tri(U,y,inferior=False)
        A_inv[:,i] = x # Asigno a x como columna_i de A_inv
    return A_inv

def calculaLDV(A):
    L,U,nops = calculaLU(A)
# Sabemos que A = LDV con L triang inf,D diagonal, V triang sup -> V_t.D = U_t
    V,D,cops = calculaLU(lib.traspuesta(U))
    return L,D,lib.traspuesta(V)

def esSDP(A, atol=1e-8):
    if not lib.esSimetrica(A,atol):
        print("La matriz no es SDP pues no es simetrica")
        return None
    res = True
    L,D,V = calculaLDV(A)
    for i in range(D.shape[0]):
        if D[i,i] <= 0:
            res = False
    return res

'----------LABO 05------------'
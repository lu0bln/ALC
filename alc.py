import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm
    
#%% -----------LABO 00 ---------- LIBRERIAS
def matriz_ceros(f, c=None):
    if c is None:
        c = 1
    res = [[0.0] * c for _ in range(f)]
    return np.array(res, dtype=float)

def matriz_identidad(n:int):
    res = [[0]*n for _ in range(n)]
    for i in range(n):
        res[i][i]= 1.0
    return np.array(res)

def multiplicar_matrices(A, B): 
    '''
    Caso VECTOR x VECTOR --> MATRIZ
    (n,) x (m,) --> (n,m)
    '''
    if A.ndim == 1 and B.ndim == 1:
        p = len(A)
        p2 = len(B)
        C = matriz_ceros(p,p2)
        for i in range(p):
            C[i,:] = A[i] * B
        return C
    
    '''
    Caso VECTOR x MATRIZ --> VECTOR
    (n,) x (n,m) --> (m,)
    ''' 
    if A.ndim == 1:
        p = B.shape[1]
        if B.shape[0] != len(A):
            return None
        C = matriz_ceros(p)
        for i in range(p):
            C[i] = np.sum(A * B[:,i])
        return C
    
    '''
    Caso MATRIZ x VECTOR --> VECTOR
    (n,m) x (m,) --> (n,)
    '''
    if B.ndim == 1:
        n, m = A.shape
        if m != len(B):
            return None
        C = matriz_ceros(n)
        for i in range(n):
            C[i] = np.sum(A[i,:] * B)
        return C
    '''
    Caso MATRIZ x MATRIZ --> MATRIZ
    (n,m) x (m,p) --> (n,p)
    '''
    n, p = A.shape
    p2, m = B.shape
    
    C = matriz_ceros(n,m)

    if p != p2:
        return None

    for j in range(m):
        colB = B[:,j]
        for i in range(n):
            C[i,j] = np.sum(A[i] * colB)     

    return C

def producto_escalar(A, B):
    A = np.array(A)
    B = np.array(B)

    if A.size != B.size:
        return None

    return float(np.sum(A * B))

def esCuadrada(matriz):
    res = False
    if (matriz.shape[0] == matriz.shape[1] and matriz.size !=0):
        res = True
    return res

def triangSup(matriz):
    f = matriz.shape[0]
    c = matriz.shape[1]
    if (c < 2 and f > 1):
        return matriz
    U = []
    #Copio la matriz original a U
    for i in range(0,f):
        U.append(matriz[i].copy()) 
    for i in range(0,f):
        for j in range(0,c):
            if (i > j):
                U[i][j] = 0
    return np.array(U)

def triangInf(matriz):
    f = matriz.shape[0]
    c = matriz.shape[1]
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

def diagonal(matriz):
    f = matriz.shape[0]
    c = matriz.shape[1]
    if (c < 2 and f > 1):
        return matriz
    r = min(f,c)
    D = matriz_ceros(f,c)
    #Copio la matriz original a U
    for i in range(0,r):
        D[i][i] = matriz[i][i]
    return np.array(D)

def traza(matriz):
    traza = 0
    i = 0
    while i < min(matriz.shape[0],matriz.shape[1]):
        traza += matriz[i][i]
        i += 1
    return traza

def traspuesta(matriz):
    matriz = np.array(matriz)
    #Caso si entra vector fila
    if len(matriz.shape) == 1:
        T = matriz_ceros(matriz.shape[0], 1)
        for i in range(matriz.shape[0]):
            T[i][0] = matriz[i]
        return T
    T = matriz_ceros(matriz.shape[1],matriz.shape[0])
    for i in range(0,matriz.shape[0]):
        for j in range(0,matriz.shape[1]):
                T[j][i] = matriz[i][j]
    return np.array(T)
#-- Funcion traspuesta optimizada, utilizada para el tp
def traspuesta_(matriz):    
    M = np.array(matriz, dtype=float)

    # caso escalar → no hay traspuesta
    if M.ndim == 0:
        return M

    # caso vector 1D → convertir a vector fila
    if M.ndim == 1:
        M = M.reshape(1, -1)

    filas, columnas = M.shape
    T = matriz_ceros(columnas, filas)

    for i in range(filas):
        for j in range(columnas):
            T[j][i] = M[i][j]

    return T
#--
def esSimetrica(matriz,atol=1e-8):
    res = True
    A_t = traspuesta_(matriz)
    if (not esCuadrada(matriz)):
        res = False
    else:
        for i in range(0,matriz.shape[0]):
            for j in range(0,matriz.shape[1]):
                #Sin tol -> if(A_t[i][j] != matriz[i][j]):
                if(abs(A_t[i][j]-matriz[i][j])>atol):
                    res = False
    return res

def calcularAx(matriz,vector):
    B = []
    if vector.shape[0] == 1:
        vector = traspuesta(vector)
    for i in range(0,matriz.shape[0]):
        v_n = 0
        for j in range(0,matriz.shape[1]):
            v_n += matriz[i][j]*vector[j]
        B.append([v_n])
    return np.array(B)

def intercambiarFilas(matriz, i, j):
    a_la_fila_j = matriz[i].copy()
    matriz[i] = matriz[j]
    matriz[j] = a_la_fila_j 
    return matriz

def sumar_fila_multiplo(matriz, i, j, escalar):
    for c in range(0,matriz.shape[1]):
        matriz[i][c] = matriz[i][c] + matriz[j][c]*escalar
    return matriz

def esDiagonalmenteDominante(matriz):
    res = True
    for i in range(0,matriz.shape[0]):
        elemento_d = abs(matriz[i][i])
        suma_fila_i = -(elemento_d)
        for j in range(0,matriz.shape[1]):
            suma_fila_i += abs(matriz[i][j])
        if (elemento_d < suma_fila_i):
            res = False
    return res

def matrizCirculante(vector):
    C = [vector[0]]
    size = vector.size
    for i in range(1,size):
        C.append([C[i-1][size - 1]])
        for j in range(1,size):
            C[i].append(C[i-1][j-1])
    return np.array(C)

def matrizVandermonde(vector):
    V = []
    for i in range (0,vector.shape[0]):
        elem_v = vector[i][0]
        V.append([elem_v ** 0])
        for j in range(1,vector.shape[0]):
            V[i].append(elem_v ** j)
    return np.array(V)


#%% -----------LABO 01 --------- NUMEROS DE MAQUINA

def error(x,y):
    return abs(np.float64(x)-np.float64(y))

def error_relativo(x,y):
    if x == 0:
        print("infinito")
        return np.inf # Representacion del infinito
    else: 
        return error(x,y)/abs(x)
    
def matricesIguales(A,B):
    A = np.array(A) # Convierto a A y B en matrices numpy por si vienen listas de listas
    B = np.array(B)
    res = True
    if (A.shape[0] == B.shape[0] and A.shape[1] == B.shape[1]):
        for f in range(0,A.shape[0]):
            for c in range(0,B.shape[1]):
                if error(A[f][c],B[f][c])>1e-08:
                    res = False
    else:
        res = False # Cuando las matrices no tengan las mismas dimensiones, no seran iguales de entrada...
    return res

#%%------------LABO 02---------- TRANSFORMACIONES LINEALES

def rota(theta):
    matriz_r = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return matriz_r

def escala(s):
    #supongo s es una lista
    n = len(s) 
    res = matriz_ceros(n,n)
    for i in range(len(s)):
        res[i][i] = s[i]
    return res 

def rota_y_escala(theta,s):
    return multiplicar_matrices(rota(theta),escala(s))

def afin(theta,s,b):
    matriz_afin = matriz_identidad(3) # matriz identidad
    matriz_re = rota_y_escala(theta,s)
    matriz_afin[:2,:2] = matriz_re  #de la fila 0 a 1, columna 0 a 1, le asigno a la matriz rota y escala
    matriz_afin[:2,2] = b #de la fila 0 a 1, en la columna 3, asigno el vector b
    return matriz_afin

def trans_afin(v,theta,s,b):
    nuevo_v = np.array([1,1,1]) # creo vector de 1's = [1,1,1]
    nuevo_v[:2] = v # donde nuevo_v sera = [v1,v2,1] para poder realizar la multiplicacion matricial
    w = calcularAx(afin(theta,s,b),nuevo_v)  # arreglar calcularAx para que acepte vectores fila tambn
    return traspuesta(w[:2])

#%%------------LABO O3---------- NORMAS Y NC
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
        ys = calcularAx(A,xs)   # Hago A*xs 
        y_norma = norma(ys,q)   #Calculo la norma de A*xs en q
        if y_norma > max_norma: #Me fijo si la nueva norma es mayor al maximo actual y lo guardo
            x_max = ys
            max_norma = y_norma
    return max_norma,x_max

def norma_matriz_inf(A):
    #Si me piden la norma inf sumo los valors abs de cada vector vi y los voy agregando a una lista para luego obtener el max de ellos
    maximos = []
    for i in range(A.shape[0]):
        maximos.append(sumatoria_fila(A[i]))
    return max(maximos)

def norma_matriz_uno(A):
    #Si me piden la norma 1 hago lo mismo que antes pero a la matriz traspuesta, asi es mas facil sumar las columnas como si fueran filas.
    At = traspuesta(A)
    maximos = []
    for i in range(At.shape[0]):
        maximos.append(sumatoria_fila(At[i]))
    return max(maximos)

def normaExacta(A,p=[1,'inf']):
    e = [1,'inf']
    A = np.array(A)
    normas = []
    if p != e and p not in e: #Agrego que si p no esta en e (pues sino falla al ser 1 o inf)
        return None
    
    if p == 1:
        return norma_matriz_uno(A)
    
    if p == 'inf':
        return norma_matriz_inf(A)
    
    uno = norma_matriz_uno(A)
    inf = norma_matriz_inf(A)
    return [uno, inf]


"Devuelve el numero de condicion de A usando la norma inducida p"
def condMC(A,p):
    A = np.array(A)
    A_ = inversa(A)
    k = normaMatMC(A,p,p,10000)[0] * normaMatMC(A_,p,p,10000)[0] # Np = 1000 Que valor deberia tener? Aleatorio?
    return k

"Que devuelve el numero de condicion de A a partir de la formula de la ecuacion (1) usando la norma p."
def condExacta(A, p) :
    A = np.array(A)
    A_ = inversa(A)
    k = normaExacta(A,p) * normaExacta(A_,p)
    return k

#%%------------LABO O4---------- FACTORIZACION LU

def calculaLU(A):
    if A is None:
        return None,None,0
    
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy().astype(float)
    
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
            Ac[j,i:n] = fila_j - (pivote)*fila_i
            cant_op+= 2*len(fila_j)-2
            fila_j[0] = pivote

    L = triangInf(Ac) + matriz_identidad(n)
    U = triangSup(Ac) 
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre la matriz Ac
    return L, U, cant_op

def esTriangSup(L):
    res = True
    for i in range(0,L.shape[0]):
        for j in range(0,L.shape[1]):
            if (i > j and L[i][j] != 0):
                return False
    return res

'Resuelve T x = b donde T es triangular inferior o superior. b es un vector fila. Devuelve un vector fila'
def res_tri(T, b, inferior=True):

    #-- b a vector columna
    b = np.array(b, dtype=float)

    #-- Si b es vector fila (1,n), pasarlo a (n,1)
    if b.ndim == 2 and b.shape[0] == 1:
        b = b.T
    # Si es vector 1D (n,), pasarlo a (n,1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    n = T.shape[0]
    x = matriz_ceros(n, 1)

    if inferior:
        # L x = b
        for i in range(n):
            suma = b[i,0]
            for j in range(i):
                suma -= T[i,j] * x[j,0]
            x[i,0] = suma / T[i,i]

    else:
        # U x = b
        for i in range(n-1, -1, -1):
            suma = b[i,0]
            for j in range(i+1, n):
                suma -= T[i,j] * x[j,0]
            x[i,0] = suma / T[i,i]

    # Devuelvo vector fila
    return traspuesta_(x)

#-- Funcion res_triangular optimizada, para usar en el tp
def res_triangular(L, b, inferior=True):
    n = L.shape[0]
    x = matriz_ceros(n)

    if not inferior:
        for i in range(n-1, -1, -1):
            s = b[i]
            row = L[i]
            for j in range(i+1, n):
                s -= row[j] * x[j]
            x[i] = s / row[i]
        return x

    for i in range(n):
        s = b[i]
        row = L[i]
        for j in range(i):
            s -= row[j] * x[j]
        x[i] = s / row[i]

    return x
#---------------------------------------------------------------
def inversa(A):
    L, U, nops = calculaLU(A)
    n = A.shape[0] 
# Teniendo L y U sabemos que det(A) = det(L).det(U) donde L es triang inf con 1s en la diagonal y U triang sup -> det(L) = 1 y det(U) = U11...Unn
    for i in range(n): 
        if U[i,i] == 0: # -> Si algun elem de la diag(U) == 0 -> det(U) == 0 y A no es inversible  
            print("La matriz no es inversible")
            return None
# Si A es inversible -> Creamos una matriz identidad (que sera nuestra x_i en el procedimiento)    
    matriz_id = matriz_identidad(n)
# Tambien para ir armando la A_inv cada sol x_i sera guardada como columna de A_i 
    A_inv = matriz_ceros(n,n)
    for i in range(n):
        y = res_tri(L,matriz_id[i],inferior=True)
        x = res_tri(U,traspuesta(y),inferior=False)
        A_inv[:,i] = x # Asigno a x como columna_i de A_inv
    return A_inv

def calculaLDV(A):
    L,U,nops = calculaLU(A)
# Sabemos que A = LDV con L triang inf,D diagonal, V triang sup -> V_t.D = U_t
    V,D,cops = calculaLU(traspuesta_(U))
    return L,D,traspuesta_(V)

def esSDP(A, atol=1e-8):
    if not esSimetrica(A,atol):
        print("La matriz no es SDP pues no es simetrica")
        return None
    res = True
    L,D,V = calculaLDV(A)
    for i in range(D.shape[0]):
        if D[i,i] <= 0:
            res = False
    return res

#%%------------LABO 05---------- A=QR
def norma_(x,p):
    x = np.array(x)
    res = 0
    if p == 'inf':
        return np.max(np.abs(x))
    res = np.sum(np.abs(x)**p)**(1/p)
    return res

def QR_con_GS(A, tol=1e-12, bar=True):
    A = np.array(A).astype(float)
    Q = matriz_ceros(A.shape[0],A.shape[1]).astype(float)
    q = matriz_ceros(A.shape[0],A.shape[1]).astype(float)
    R = matriz_ceros(A.shape[1],A.shape[1]).astype(float)
    
    Q[:,0] = A[:,0]*(1/norma_(A[:,0], 2))
    R[0,0] = norma_(A[:,0], 2)
    
    iterador = tqdm(range(1,A.shape[1]), desc="Resolviendo QR con Gram Schmidt") if bar else range(1,A.shape[1])

    for j in iterador:
        q[:,j] = A[:,j]
        
        for k in range(0,j):
            R[k,j] = producto_escalar(Q[:,k], q[:,j])
            q[:,j] = q[:,j] - R[k,j]*Q[:,k]
        
        q_norma = norma_(q[:,j], 2)
        
        if q_norma < tol:
            R[j,j] = 0.0
            Q[:,j] = 0.0
        else:
            R[j,j] = q_norma
            Q[:,j] = q[:,j] / R[j,j]

    return Q, R

def QR_con_HH(A, tol=1e-12, bar=True):
    n,m = A.shape

    R = A.copy()
    Q = matriz_identidad(n)

    iterador = tqdm(range(min(n, m)), desc="Resolviendo QR con House Holder") if bar else range(min(n, m))

    for k in iterador:
        x = R[k:, k]
        sign = 1.0 if x[0]>= 0 else -1.0
        a = -sign * norma_(x, 2)
        u = x.copy()
        u[0] -= a 
        norm_u = norma_(u,2)
        
        if norm_u < tol:
            continue
        
        u = u / norm_u
        R[k:, k:] -= 2.0 * multiplicar_matrices(u, multiplicar_matrices(u, R[k:, k:]))
        Q[:, k:] -= 2.0 * multiplicar_matrices(multiplicar_matrices(Q[:,k:], u), u)

    return Q, R

def calculaQR(A, metodo = 'RH', tol = 1e-12, bar=True):
    metodos = ['RH', 'GS']
    if(metodo not in metodos):
        return None
    
    if(metodo == 'RH'):
        return QR_con_HH(A, tol, bar)
    
    if(metodo == 'GS'):
        return QR_con_GS(A, tol, bar)

#%%------------LABO 06---------- Autovalores

def multiplicar(x,v):
    x= np.array(x)
    v = np.array(v)
    if len(x)!=len(v):
        print("Sytaxis error en multiplicar")
    res = 0
    for i in range(len(x)):
        res += x[i][0]*v[i][0]
    
    return res

def f(A, v):
    w = multiplicar_matrices(A, v)
    norma_w = np.sqrt(multiplicar(w, w))
    if norma_w != 0:
        return w / norma_w
    return matriz_ceros(A.shape[0], 1)

def metpot2k(A, tol=1e-15, n=1000):
    vec = np.random.rand(A.shape[0], 1)
    w_monio = f(A, f(A, vec))
    e = multiplicar(w_monio, vec)
    k = 0 
    
    while np.abs(e - 1) > tol and k < n:
        vec = w_monio
        w_monio = f(A, f(A, vec))
        e = multiplicar(w_monio, vec)
        k += 1
        
    Av = multiplicar_matrices(A, w_monio)
    lam = multiplicar(w_monio, Av)
    return w_monio, lam, k

def diagRH_aux(A, u, k):
    """Calcula B = (I - k u u.T) A (I - k u u.T) manualmente."""
    n = A.shape[0]
    
    # 1. v = A * u
    v = matriz_ceros(n, 1)
    for i in range(n):
        v[i, 0] = np.sum(A[i, :] * u[:, 0])
        
    # 2. w = v - (k/2) * (u.T * v) * u
    ut_v = multiplicar(u, v)
    beta = ut_v * (k / 2.0)
    
    w = matriz_ceros(n, 1)
    for i in range(n):
        w[i, 0] = v[i, 0] - beta * u[i, 0]
    
    # 3. B = A - k * (u * w.T + w * u.T)
    B = matriz_ceros(n, n)
    for i in range(n):
        for j in range(n):
            term = u[i, 0] * w[j, 0] + w[i, 0] * u[j, 0]
            B[i, j] = A[i, j] - k * term
    return B

def aplicar_H_a_matriz(M, u, k):
    """Calcula H * M columna por columna manualmente."""
    n, m = M.shape
    res = matriz_ceros(n, m)
    
    for j in range(m):
        col_j = M[:, j] # Vector columna (n,)
        
        # dot = u.T * col
        dot = np.sum(u[:, 0] * col_j)
        
        # res = col - k * dot * u
        factor = k * dot
        for i in range(n):
            res[i, j] = col_j[i] - factor * u[i, 0]
            
    return res
def diagRH(A, tol=1e-15, K=1000, k_max="max"):
    if not esSimetrica(A):
        print("La matriz no es simetrica")
        return None, None
    n = A.shape[0] 
    if n == 1: return np.array([[1.0]]), A 

    # Manejo k_max para SVD truncada
    if k_max != "max":
        if k_max <= 0: return matriz_identidad(n), A
        k_next = k_max - 1
    else:
        k_next = "max"

    # 1. MetPot
    v1, lam, _ = metpot2k(A, tol, min(K, 500))

    # 2. Householder
    e1 = matriz_ceros(n, 1); e1[0,0] = 1.0
    u = e1 - v1
    norma_u2 = multiplicar(u, u)
    
    if abs(norma_u2) < 1e-15:
        k_val = 0.0; B = A.copy()
    else:
        k_val = 2.0 / norma_u2
        B = diagRH_aux(A, u, k_val)

    # 4. Deflación
    A_monio = B[1:, 1:]

    # 5. Recursión
    S_monio, D_monio = diagRH(A_monio, tol, K, k_max=k_next)

    # 6. Reconstrucción D
    D_final = matriz_ceros(n, n)
    D_final[0, 0] = lam
    filas_d, cols_d = D_monio.shape
    for i in range(filas_d):
        for j in range(cols_d):
            D_final[i+1, j+1] = D_monio[i, j]

    # 7. Reconstrucción S
    S_bloque = matriz_ceros(n, n)
    S_bloque[0, 0] = 1.0
    filas_s, cols_s = S_monio.shape
    for i in range(filas_s):
        for j in range(cols_s):
            S_bloque[i+1, j+1] = S_monio[i, j]
            
    if k_val > 0:
        S_final = aplicar_H_a_matriz(S_bloque, u, k_val)
    else:
        S_final = S_bloque

    return S_final, D_final

#%%------------LABO 07---------- Trancisiones_y_rala

def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """
    res = matriz_ceros(n,n)
    for j in range(n):
        # ahroa vec es un vec aleatorio de n filas por 1 col
        vec = np.abs(np.random.rand(n,1))
        vec = (vec)/np.sum(vec)#normalizo
        for i in range(n):
            #vec es un vec columna entoces necesito indicar la col [0]
            res[i,j]= vec[i][0]
    return res

def tranciciciones_uniformes_aux(n,thres):
    #devuelve un vec normalizado con al menos v[i]!=0
    res = matriz_ceros(n,1)
    suma_parcial = 0
    for i in range(n):
        if np.random.rand() <= thres:
            res[i] = 1
            suma_parcial+=1

    if suma_parcial==0:
        #mala idea hacerlo con recursion y probalidad
        indice_random       = np.random.randint(0,n)#tomo el indice y cambio ese
        res[indice_random]  = 1.0
        return res

    return res/suma_parcial

def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """
    # por lo que tengo entendido en esta funcion incremento la probabilidad que sea 0
    # entocnes si el numero al azar es menor que thres pongo 1 sino 0
    # despues normalizo asi quedan todos iguales los que no son 0
    if thres >=1:
        print(f"Imposible thres mal dado")
        return None
    res = matriz_ceros(n,n)
    for j in range(n):# son las col por que vec es un vector col
        vec = tranciciciones_uniformes_aux(n,thres)
        for i in range(n)            :
            res[i,j] = vec[i][0]
    return res

def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), 
    usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    # como A*v=Lam*v si lam tiende a 0 entonces A*v = 0 entonces v es nucleo 
    #buscamos lam que sean mas chicos que la tol
    A_t = traspuesta(A)
    At_A = multiplicar_matrices(A_t,A)
    #print(At_A,At_A.shape)
    #if esSimetrica(At_A):
    S,D = diagRH(At_A,tol)
    n = S.shape[0]
    indices = []
    for i in range(len(S)):
        autovalor = D[i,i]
        if np.abs(autovalor)<=tol:
            indices.append(i)
            #ahora tengo la lista de indecis que cumplen
    k = len(indices)
    if k==0:
        print(f"No exsiste Nucleo para \n {A}")
        return matriz_ceros(0,0)# tuve que poner esto para que pase el A.shape[0]==0
    nucleos = []
    for j in indices:
        autovec =S[:,j]    
        nucleos.append(autovec)
    res = traspuesta(np.array(nucleos))

    return res

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. 
    Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a 
    valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. 
    Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    #basicamente puse el caso borde y le di lo que pedia
    if len(listado)==0:
        return [{},((m_filas,n_columnas))]
    else:
        indices_i   =   listado[0]
        indices_j   =   listado[1]
        valores     =   listado[2]
        dict_res    =   {}
        # asumo que todas las listas de las listas tienen la misma dimencion
        for k in range(len(valores)):
            i   = indices_i[k]
            j   = indices_j[k]
            val = valores[k]
            if np.abs(val)>tol:
                # agrego val al diccionario
                dict_res[(i,j)]= val
    return [dict_res ,(m_filas,n_columnas)] #era una lista lo que tenia que devolver

def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    rala = A[0]
    filas , cols = A[1]
    if len(v)!=cols:
        print("dimenciones mal")
    w    = np.array([0.0]*filas)
    for tupla in rala.keys():
        i ,j= tupla
        val = rala[tupla]
        # es para hacer fila y col como no se todos los datos de la fila pongo la condicion del if y voy sumando
        if i < filas and j < cols:
            w[i]+= val*v[j]
    return w 
#%%------------LABO 08---------- SVD
def svd_reducida(A, k="max", tol=1e-15):
    print(f"Iniciando SVD reducida de matriz {A.shape}")
    m_orig, n_orig = A.shape
    transp = False
    if m_orig < n_orig:
        A = traspuesta_(A); transp = True
    m, n = A.shape
    
    # 1. Matriz Covarianza
    print("  - Calculando AtA")
    At = traspuesta_(A)
    AtA = multiplicar_matrices(At, A)
    
    k_int = "max"
    if k != "max" and str(k).upper() != "MAX":
        k_int = int(k)
        
    print(f"  - Diagonalizando ({n}x{n}) con k={k_int}")
    S_v, D_v = diagRH(AtA, tol, k_max=k_int)
    
    # 2. Filtrar
    autovalores = []
    for i in range(D_v.shape[0]): autovalores.append(D_v[i,i])
    
    indices = []
    for i in range(len(autovalores)):
        if abs(autovalores[i]) > tol: indices.append(i)
        else: break
    
    r = len(indices)
    k_final = r
    if k_int != "max": k_final = min(k_int, r)
    
    if k_final == 0:
        return matriz_ceros(m_orig, 0), np.array([]), matriz_ceros(n_orig, 0)
    
    # 3. Construir
    hatSig_vec = []
    for i in range(k_final):
        hatSig_vec.append(np.sqrt(autovalores[indices[i]]))
    hatSig = np.array(hatSig_vec)
    
    hatV = matriz_ceros(n, k_final)
    for j in range(k_final):
        idx = indices[j]
        for i in range(n): hatV[i, j] = S_v[i, idx]
    
    # 4. Construir U
    print(f"  - Construyendo U (k={k_final})")
    hatU = matriz_ceros(m, k_final)
    for j in range(k_final):
        v_j = matriz_ceros(n, 1)
        for i in range(n): v_j[i, 0] = hatV[i, j]
            
        sigma = hatSig[j]
        if sigma > tol:
            u_j = multiplicar_matrices(A, v_j)
            inv_sig = 1.0 / sigma
            for i in range(m):
                hatU[i, j] = u_j[i, 0] * inv_sig
        
    if transp: return hatV, hatSig, hatU
    return hatU, hatSig, hatV

#%%----------TRABAJO PRACTICO------------Funciones------------------------

#%% 1.-- Lectura de datos. 
'Recibe la carpeta donde estan nuestros embeddings y nos devuelve X_train,Y_train,X_validate,Y_validate'
def cargarDatos(carpeta):
    dogs_t = carpeta + 'train/dogs/efficientnet_b3_embeddings.npy'
    cats_t = carpeta + 'train/cats/efficientnet_b3_embeddings.npy'

    dogs_v = carpeta + 'val/dogs/efficientnet_b3_embeddings.npy'
    cats_v = carpeta + 'val/cats/efficientnet_b3_embeddings.npy'

    dogs_t = np.load(dogs_t)
    cats_t = np.load(cats_t)

    dogs_v = np.load(dogs_v)
    cats_v = np.load(cats_v)

    y =[]
    for _ in range(1000):
        y.append([1,0])
    for _ in range(1000):
        y.append([0,1])
    
    Y_t = np.array(y).T

    y = []
    for _ in range(500):
        y.append([1,0])
    for _ in range(500):
        y.append([0,1])

    Y_v = np.array(y).T

    data_t = np.concatenate((cats_t,dogs_t),axis=1)
    data_v = np.concatenate((cats_v,dogs_v),axis=1)

    return data_t, Y_t, data_v, Y_v

#%% 2.-- Ecuaciones Normales 

'La funcion recibe una matriz A, y devuelve la L de la descomposicion de Cholesky / A=L.L_t'
def calculaCholesky(A):
    A = A.astype(float)
    print(A)
    if not esSDP(A,atol=1e15):
        return None
    L_,D_,V_ = calculaLDV(A)
    D = np.sqrt(D_)
    L = L_@D
    return L

'- Algorithm 1: Funcion que recibe X e Y matrices / depende el caso a,b o c el peso W se calcula de distinta manera. Devuelve el peso W'
def calculo_peso_W(X,Y):
    n,p = X.shape
    # El rango va a estar acotado entre min(n,p)
    rango = min(n,p)
    X_t = traspuesta_(X)
    #-- Caso a) rango(X) = p ^ n>p -> X_ = (X_t.X)^-1.X_t donde resuelvo (X_t.X)U=X_t aplicando cCholesky a X_t.X
    if rango == p and n>p:
    #-- Para hallar X_ -> resolvemos A.U = X_t aplicando cholesky a A -> L.L_t.U=X_t
        A = X_t@X
        L = calculaCholesky(A) # triang inf
    #-- Resolvemos L.y= X_t con L_t.U= y (y matriz de L_t.shape[0] y U.shape[1]) --> calculamos W
        W = pinvEcuacionesNormales(X,L,Y)
        return W
    #-- Caso b) rango(X) = n ^ n<p -> X_ = X_t(X.X_t)^-1 donde resuelvo V.(X.X_t)=X_t aplicando Cholesky a (X.X_t) 
    elif  rango == n and n<p:
        A = X@X_t
        L = calculaCholesky(A)
    #-- Resolvemos L.y= X con L_t.V_t= y (y matriz de L_t.shape[0] y V_t.shape[1])
        W = pinvEcuacionesNormales(X,L,Y)
        return W
    #-- Caso c) rango(X) = p ^ n=p -> X_ = (X)^-1 donde despejo W de WX=Y
    elif rango == n and n == p:
        #-- Para despejar W tomamos a X = L.L_t 
        L = calculaCholesky(X)
        W = pinvEcuacionesNormales(X,L,Y)
        return W
    return None

'Funcion pinvEcuacionesNormales() recibe X de entrenamiento, L la matriz de Cholesky (dependiendo el caso es distinta), e Y'
def pinvEcuacionesNormales(X,L,Y):
    n,p = X.shape
    rango = min(n,p)
    W = []

    if rango == p and n>p:
        #-- Resolvemos L.y= X_t con L_t.U= y (y matriz de L_t.shape[0] y U.shape[1]) --> obtenemos la pinv U
        pinv = calculo_pinv(traspuesta_(X),L)
        #-- Calculamos W 
        W = Y@pinv
    elif rango == n and n<p:
        #-- Resolvemos L.y= X con L_t.V_t= y (y matriz de L_t.shape[0] y V_t.shape[1])
        pinv = calculo_pinv(X,L)
        #-- Calculamos W, como pinv es V_t -> trasponemos para obtener V pseudoinversa  
        W = Y@traspuesta_(pinv)
    elif rango == n and n == p:
        #-- Para despejar W usamos X=L.L_t -> W.L.L_t = Y -> (W.L.L_t)_t = Y_t -> L.L_t.W_t = Y_t de donde tomamos L_t.W_t = y ^ L.y = Y_t
        W = calculo_pinv(traspuesta_(Y),L)
        #-- Hasta aca obtuve W_t, traspongo para tener la W
        W = traspuesta_(W)
    return W

'Funcion calculo_pinv() recibe X y L de Cholesky previamente calculado para cada caso'
def calculo_pinv(X,L):
    # L es triangular inferior (Cholesky)
    # X es matriz (puede ser X o X_t según el caso)

    L_t = traspuesta_(L)
    m, n = X.shape
    pinv = matriz_ceros(L.shape[1], n)

    for i in range(n):
        # -- Tomo la columna i de X como vector 1D pues res_tri recibe (matriz,vector fila)
        b = X[:, i].reshape(-1)

        # -- L.Y = b 
        y = res_tri(L, b, inferior=True)   # y ya es 1D

        # -- L_t.x = y
        x = res_tri(L_t, y, inferior=False)

        # -- Guardamos pinv
        pinv[:, i] = x.reshape(-1)

    return pinv
'''    
    L_t = traspuesta_(L) #triang sup
    pinv = matriz_ceros(L.shape[1],X.shape[1])

    for i in range(L.shape[0]):
            y = res_tri(L,traspuesta_(X[:,i]),inferior=True) # Pongo las columnas de X como filas pues res_tri recibe (matriz,vector fila)
            x = res_tri(L_t,traspuesta_(y),inferior=False)
            pinv[:,i] = x # Asigno a x como columna_i de U
    return pinv 
'''


#-- Pruebas -------------------------------
carpeta='cats_and_dogs/'
X_t, y_t,X_v, y_v = cargarDatos(carpeta)

W = calculo_peso_W(X_t,y_t)
print(W)
#------------------------------------------

'''
'La funcion recibe X, la matriz de embeddings, L la matriz de Cholesky, y Y la matriz de targets de entrenamiento. La funcion devuelve W'
def pinvEcuacionesNormales(X,L,Y):
    n,p = X.shape
    m,q = Y.shape
    # El rango va a estar acotado entre min(n,p)
    r = min(n,p)
    X_t = traspuesta(X)
    W = []
    #-- Caso a) rango(X) = p ^ n>p -> X_ = (X_t.X)^-1.X_t donde resuelvo (X_t.X)U=X_t aplicando cCholesky a X_t.X
    if r == p and n>p:
    #-- Para hallar X_ -> resolvemos A.U = X_t aplicando cholesky a A -> L.L_t.U=X_t
        A = multiplicar_matrices(X_t,X)
        L = calculaCholesky(A) # triang inf
        L_t = traspuesta(L) # triang sup
    #-- Resolvemos L.y= X_t con L_t.U= y (y matriz de L_t.shape[0] y U.shape[1])
        U = matriz_ceros(n,n) #revisar dimensiones
        for i in range(n):
            y = res_tri(L,traspuesta(X_t[:,i]),inferior=True) # Uso las filas de X pues son las columnas de X_t
            x = res_tri(L_t,traspuesta(y),inferior=False)
            U[:,i] = x # Asigno a x como columna_i de U
        W = multiplicar_matrices(Y,U)
    #-- Caso b) rango(X) = n ^ n<p -> X_ = X_t(X.X_t)^-1 donde resuelvo V.(X.X_t)=X_t aplicando cCholesky a (X.X_t)
    elif  r == n and n<p:
        A = multiplicar_matrices(X,X_t)
        L = calculaCholesky(A)
        L_t = traspuesta(L)
    #-- Resolvemos L.y= X con L_t.V_t= y (y matriz de L_t.shape[0] y V_t.shape[1])
        V = matriz_ceros(n,n) #revisar dimensiones
        for i in range(n):
            y = res_tri(L,traspuesta(X[:,i]),inferior=True) # Uso las filas de X_t pues son las columnas de X
            x = res_tri(L_t,traspuesta(y),inferior=False)
            V[:,i] = x # Asigno a x como columna_i de V
        # Esta V es V_t pues (V.(X.X_t)=X_t)_t -> (X.X_t)V_t = X
        W = multiplicar_matrices(Y,traspuesta(V))
    #-- Caso c) rango(X) = p ^ n=p -> X_ = (X)^-1 donde despejo W de WX=Y
    elif r == n and n == p:
        L_t = traspuesta(L)
        W = matriz_ceros(n,n) #revisar dimensiones
        for i in range(n):
        #-- Para despejar W usamos X=L.L_t -> W.L.L_t = Y -> (W.L.L_t)_t = Y_t -> L.L_t.W_t = Y_t de donde tomamos L_t.W_t = y ^ L.y = Y_t
            y = res_tri(L,Y[i],inferior=True) # Uso las filas de Y pues son las columnas de Y_t
            x = res_tri(L_t,traspuesta(y),inferior=False)
            W[:,i] = x # Asigno a x como columna_i de W
        #-- Hasta aca obtuve W_t, traspongo para tener la W
        W = traspuesta(W)
    return W
'''
#%% 3.-- Descomposicion en valores singulares

def pinvSVD(U, S, V, Y):
    """W = Y * V * S_inv * U.T"""
    # 1. Y * V
    print("  - Paso 1: Y @ V")
    YV = multiplicar_matrices(Y, V)
    if YV is None: return None
    m, k = YV.shape
    
    # 2. Escalar
    print("  - Paso 2: Escalar Sigma")
    YV_scaled = matriz_ceros(m, k)
    for j in range(k):
        val_s = S[j]
        inv = 1.0/val_s if abs(val_s)>1e-15 else 0.0
        for i in range(m):
            YV_scaled[i, j] = YV[i, j] * inv
            
    # 3. Por U.T
    print("  - Paso 3: Resultado @ U.T")
    Ut = traspuesta_(U)
    W = multiplicar_matrices(YV_scaled, Ut)
    return W
#%% 4.-- Descomposicion QR
def pinvHouseHolder(Q, R, Y, bar=True):
    return pinvQR(Q, R, Y, metodo='House Holder', bar=bar)

def pinvGramSchmidt(Q, R, Y, bar=True):
    return pinvQR(Q, R, Y, metodo='Gram Schmidt', bar=bar)

def pinvQR(Q, R, Y, metodo, bar=True):
    n, m = R.shape
    r = min(n, m)
    Rr = R[:r, :r] #Tomo las filas y columnas que estan trianguladas
    Qr = Q[:, :r] #Tomo las columnas necesarias para dimensiones
    
    Rrt = traspuesta_(Rr)
    Qrt = traspuesta_(Qr)

    Vt = matriz_ceros(r, Qr.shape[0])

    iterador = tqdm(range(Qrt.shape[1]), desc=f"Resolviendo pinvQR, metodo: {metodo}") if bar else range(Qrt.shape[1])

    for i in iterador:
        Vt[:,i] = res_triangular(Rrt, Qrt[:, i], inferior=True)
    
    V = traspuesta_(Vt)

    return multiplicar_matrices(Y, V) #Retorno el W con los pesos
#%% 5.-- Pseudo-Inversa de Moore-Penrose
'Recibe dos ma-trices y devuelva True si verifican las condiciones de Moore-Penrose'
def esPseudoInversa(X,pX,tol=1e-8):
    res = False
    if condiciones_MP(X,pX,tol,1) and condiciones_MP(X,pX,tol,2) and condiciones_MP(X,pX,tol,3) and condiciones_MP(X,pX,tol,4):
        res = True
    return res

def condiciones_MP(X,pX,tol,condicion):
    res = False
    if condicion == 1:
        A = multiplicar_matrices(pX,X)
        Y = multiplicar_matrices(X,A)
        if matricesIguales(Y,X):
            res = True
    if condicion == 2:
        A = multiplicar_matrices(X,pX)
        Y = multiplicar_matrices(pX,A)
        if matricesIguales(Y,pX):
            res = True
    if condicion == 3:
        A = multiplicar_matrices(X,pX)
        if matricesIguales(A,traspuesta_(A)):
            res = True
    if condicion == 4:
        A = multiplicar_matrices(pX,X)
        if matricesIguales(A,traspuesta_(A)):
            res = True
    return res

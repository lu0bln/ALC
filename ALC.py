import numpy as np
#%% -----------LABO 00 ---------- LIBRERIAS
def matriz_ceros(filas, cols): # <--- CAMBIO: Aceptar filas y cols
    res = []
    for _ in range(filas):
        vec_zeros = []
        for _ in range(cols):
            vec_zeros.append(0.0) # <-- CAMBIO: Usar 0.0 (float)
        res.append(vec_zeros)
    return np.array(res)

def matriz_identidad(n:int):
    res = [[0]*n for _ in range(n)]
    for i in range(n):
        res[i][i]= 1.0
    return np.array(res)

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

def multiplicar_matrices(A,B):
    C = matriz_ceros(filas(A),columnas(B))
    for i in range(0,filas(A)):
        for j in range(0,columnas(B)):
            n = 0
            for k in range(0,filas(B)):
                n += A[i][k]*B[k][j]
            C[i][j] = n
    return np.array(C)

def producto_escalar(A, B):
    if(len(A) != len(B)):
        return None
    res = 0
    for i in range(len(A)):
        res = res + (A[i] * B[i])

    return res

def esCuadrada(matriz):
    res = False
    if (filas(matriz) == columnas(matriz) and matriz.size !=0):
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

def traza(matriz):
    traza = 0
    i = 0
    while i < min(filas(matriz),columnas(matriz)):
        traza += matriz[i][i]
        i += 1
    return traza

def traspuesta(matriz):
    matriz = np.array(matriz)
    T = matriz_ceros(matriz.shape[1],matriz.shape[0])
    for i in range(0,matriz.shape[0]):
        for j in range(0,matriz.shape[1]):
                T[j][i] = matriz[i][j]
    return np.array(T)

def esSimetrica(matriz,atol=1e-8):
    res = True
    A_t = traspuesta(matriz)
    if (not esCuadrada(matriz)):
        res = False
    else:
        for i in range(0,filas(matriz)):
            for j in range(0,columnas(matriz)):
                #Sin tol -> if(A_t[i][j] != matriz[i][j]):
                if(abs(A_t[i][j]-matriz[i][j])>atol):
                    res = False
    return res

def calcularAx(matriz,vector):
    B = []
    if vector.shape[0] == 1:
        vector = traspuesta(vector)
    for i in range(0,filas(matriz)):
        v_n = 0
        for j in range(0,columnas(matriz)):
            v_n += matriz[i][j]*vector[j]
        B.append([v_n])
    return np.array(B)

def intercambiarFilas(matriz, i, j):
    a_la_fila_j = matriz[i].copy()
    matriz[i] = matriz[j]
    matriz[j] = a_la_fila_j 
    return matriz

def sumar_fila_multiplo(matriz, i, j, escalar):
    for c in range(0,columnas(matriz)):
        matriz[i][c] = matriz[i][c] + matriz[j][c]*escalar
    return matriz

def esDiagonalmenteDominante(matriz):
    res = True
    for i in range(0,filas(matriz)):
        elemento_d = abs(matriz[i][i])
        suma_fila_i = -(elemento_d)
        for j in range(0,columnas(matriz)):
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
    for i in range (0,filas(vector)):
        elem_v = vector[i][0]
        V.append([elem_v ** 0])
        for j in range(1,filas(vector)):
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
                res = False
    return res

def res_tri(L,b,inferior = True):
    n = L.shape[0]
    x = matriz_ceros(n,1)
    if esTriangSup(L):  # Por los tests, si me dan inferior = False no basta, tengo que verificar si es o no triangSup pues cambia todo
        for i in range(n-1,-1,-1):
            x_i = b[i]
            for j in range(i+1,n):  #Cambie los indices pues no depende del tamano de x, es mejor asi
                x_i -= L[i,j]*x[j]
            x[i] = (x_i*1/L[i,i])
        return traspuesta(x)
    if inferior == False:   # Por los test, si me dan inferior = False pero es triangInf -> lo resuelvo normal,
        L = traspuesta(L)
    for i in range(0,n):
        x_i = b[i]
        for j in range(i):      #Cambie los indices pues no depende del tamano de x, es mejor asi
            x_i -= L[i,j]*x[j]
        x[i] = (x_i*1/L[i,i])
    return traspuesta(x)

def inversa(A):
    L, U, nops = calculaLU(A)
    n = A.shape[0] 
# Teniendo L y U sabemos que det(A) = det(L).det(U) donde L es triang inf con 1s en la diagonal y U triang sup -> det(L) = 1 y det(U) = U11...Unn
    for i in range(A.shape[0]): 
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
    V,D,cops = calculaLU(traspuesta(U))
    return L,D,traspuesta(V)

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

def QR_con_GS(A, tol=1e-12):
    if(A.shape[0] != A.shape[1]):
        return None
    
    cant_ops = 0
    Q = matriz_ceros(A.shape[0],A.shape[1]).astype(float)
    q = matriz_ceros(A.shape[0],A.shape[1]).astype(float)
    R = matriz_ceros(A.shape[1],A.shape[1]).astype(float)
    
    Q[:,0] = A[:,0]*(1/norma(A[:,0], 2))
    R[0,0] = norma(A[:,0], 2)
    
    for j in range(1,A.shape[0]):
        q[:,j] = A[:,j]
        
        for k in range(0,j):
            R[k,j] = producto_escalar(Q[:,k], q[:,j])
            q[:,j] = q[:,j] - R[k,j]*Q[:,k]
        
        q_norma = norma(q[:,j], 2)
        
        if q_norma < tol:
            R[j,j] = 0.0
            Q[:,j] = matriz_ceros(A.shape[1])
        else:
            R[j,j] = q_norma
            Q[:,j] = q[:,j] / R[j,j]
    
    non_zero = (Q != 0).any(axis = 0)
    Q = Q[:, non_zero]
    R = R[0:Q.shape[1], :]

    return Q, R

#Primer vector canonico de R^k
def e_1(k):
    e = np.zeros(k)
    e[0] = 1
    return e

#Creamos H_monio con dos matrices cuadradas (nxn) y (mxm) para formar una matriz (n+m x n+m)
def crear_H_monio(A, B):
    if(len(A.shape) == 1):
        filas_a, col_a = 0, 0
    else:
        filas_a, col_a = A.shape
    
    if(len(B.shape) == 1):
        filas_b, col_b = 0, 0
    else:
        filas_b, col_b = B.shape

    filas_res = filas_a + filas_b
    col_res = col_a + col_b

    res = matriz_ceros(filas_res, col_res)

    res[0:filas_a, 0:col_a] = A
    res[filas_a:filas_res, col_a:col_res] = B

    return np.array(res)

def QR_con_HH(A, tol=1e-12):
    filas = A.shape[0]
    col = A.shape[1]
    
    if(filas < col):
        return None

    R = A.copy()
    Q = matriz_identidad(filas)

    for k in range(0, col):
        x = R[k:filas, k]
        sign = 1 if x[0]>= 0 else -1
        a = -sign * norma(x, 2)
        u = x - a * e_1(filas-k)
        norm_u = norma(u, 2)
        if(norm_u > tol):
            u = u / norm_u
            u = np.array([u])
            H = matriz_identidad(filas-k) - 2*(multiplicar_matrices(traspuesta(u), u))
            H_monio = crear_H_monio(matriz_identidad(k), H)
            R = multiplicar_matrices(H_monio, R)
            Q = multiplicar_matrices(Q, traspuesta(H_monio))
    
    return Q, R

def calculaQR(A, metodo = 'RH', tol = 1e-12):
    metodos = ['RH', 'GS']
    if(metodo not in metodos):
        return None
    
    if(metodo == 'RH'):
        return QR_con_HH(A, tol)
    
    return QR_con_GS(A, tol)

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

def f(A,v):
    w = multiplicar_matrices(A,v)
    norma_w = np.sqrt(multiplicar(w,w))
    if norma_w!=0:
        return w/norma_w
    return matriz_ceros(A.shape[0],1)

def metpot2k(A, tol=1e-15, n=1000):
    vec = np.random.rand(len(A), 1)
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

def diagRH(A,tol = 1e-15,K = 1e5):
    if esSimetrica(A)== False:
        print("La matriz no es simetrica")
        return None, None
    n = A.shape[0] 
    v1,lam , _ = metpot2k(A,tol,K)
    #necesito que sea un vec col
    e1 = matriz_ceros(n,1)
    e1 [0][0]= 1
    u = e1 -v1
    noma_u = multiplicar(u,u) #estaba alcuadrado
    k =2/noma_u
    u_t = traspuesta(u)
    u_uT = multiplicar_matrices(u,u_t)# este tiene que dar una matriz
    # H = I -k* u*uT (u*uT es de n x n)
    H = matriz_identidad(n) - k* u_uT
    # como H es simetrica H=H.T
    HA = multiplicar_matrices(H,A)# es un H*A de paso
    #recordamos que H=H.T
    B  = multiplicar_matrices(HA,H) # ya es el paso final
    if n==2:
        return H,B
    # ahora ya con todo eso tenmos lsito para la recursion
    A_monio = []# saco la primera fila y col como en el pddf
    for fila in B[1:n]:
        fila_A = []
        for j in range(1,len(fila)):
            fila_A.append(fila[j])
        A_monio.append(fila_A)
    A_monio =np.array(A_monio)
    # vamos a la recursion con A monio
    S_monio,D_monio = diagRH(A_monio,tol,K)
    #Ahora reconstruyo s y d
    S_reconstrudio = np.array(matriz_identidad(n))
    D_reconstrudio = np.array(matriz_ceros(n,n))
    #recuen ahora me doy cuenta que puedo hacer con los arrayt
    # puedo usar coma A[1:,1:]   para reducir 
    D_reconstrudio[0,0] = lam
    D_reconstrudio[1:,1:] = D_monio
    S_reconstrudio[1:,1:] = S_monio
    S_final = multiplicar_matrices(H,S_reconstrudio)
    D_final = D_reconstrudio
    return S_final,D_final

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
#%%----------TRABAJO PRACTICO------------Funciones------------------------

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

    data_t = np.concatenate((cats_t,dogs_t))
    data_v = np.concatenate((cats_v,dogs_v))

    return data_t, Y_t, data_v, Y_v

if __name__ == "__main__":
    test = np.array([[1,3,1],[5,3,1],[15,9,3]])
    test2 = matriz_identidad(3)
    A = np.array([[3,2],[4,1]])
    
    path = 'cats_and_dogs/'

#-- 2. Ecuaciones Normales 

def esColumnaNula(x):
    res = True
    for elem in x:
        if abs(elem) > 1e-10:
            return False
    return res
def rango(X):
    n = 0
    L,U,ops = calculaLU(X)
    for j in range(U.shape[1]):
        columna = U[:,j]
        if not esColumnaNula(columna):
            n+=1
    return n
def calculaCholesky(A):
    if not esSDP(A,atol=1e-10):
        return None
    L_,D_,V_ = calculaLDV(A)
    D = np.sqrt(D_)
    L = multiplicar_matrices(L_,D)
    return L

'La funcion recibe X, la matriz de embeddings, L la matriz de Cholesky, y Y la matriz de targets de entrenamiento. La funcion devuelve W'
def pinvEcuacionesNormales(X,L,Y):
    n,p = X.shape
    m,q = Y.shape
    r = rango(X)

    #-- Caso a) rango(X) = p ^ n>p -> X_ = (X_t.X)^-1.X_t donde resuelvo (X_t.X)U=X_t aplicando cCholesky a X_t.X
    if r == p and n>p:

    #-- Caso b) rango(X) = n ^ n<p -> X_ = X_t(X.X_t)^-1 donde resuelvo V.(X.X_t)=X_t aplicando cCholesky a (X.X_t)
    elif  r == n and n<p:

    #-- Caso c) rango(X) = p ^ n=p -> X_ = (X)^-1 donde despejo W de WX=Y
    elif r == n and n == p:
    
    return None
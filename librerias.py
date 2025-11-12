import numpy as np
import matplotlib.pyplot as plt #libreria para graficar

## EJERCICIO 3 (P1)
'''
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
# Numeros complejos <-- NOTAR QUE EL j SE RESERVA PARA LOS NUMEROS IMAGINARIOS EN PYTHON
1j *1j
(1+2j )*1j
'''
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

## EJERCICIO 21 (P1) <-- HACER EN ARCHIVO A PARTE!


## FUNCIONES REUTILIZABLES
A_test = np.array([[1,1,1],[1,1,0]])
B_test = np.array([[1,3],[2,4],[3,2]])
matriz_test = np.array([[-10 ,2 ,3 ,4] ,[0 ,7 ,2 ,3] ,[2 ,3 ,12 ,5] ,[0 ,0 ,1 ,2]])
vector_columna = np.array([[1],[2],[3],[4],[5]])
vector_fila = np.array([[1,2,3]])

def matriz_ceros(filas, cols): # <--- CAMBIO: Aceptar filas y cols
    res = []
    for _ in range(filas):
        vec_zeros = []
        for _ in range(cols):
            vec_zeros.append(0.0) # <-- CAMBIO: Usar 0.0 (float)
        res.append(vec_zeros)
    return np.array(res)
#print(f"Matriz de ceros de n filas y m columnas:\n{matriz_ceros(5,3)}")

# Hago la matriz identidad
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

#print(f"Multiplicar la matriz A =\n{A_test} \npor la matriz B=\n{B_test}\nda como resultado C=\n{multiplicar_matrices(A_test,B_test)} ")

## EJERCICIO 1
def esCuadrada(matriz):
    res = False
    if (filas(matriz) == columnas(matriz) and matriz.size !=0):
        res = True
    return res

#print(f"La matriz:\n{matriz_test}\nes cuadrada?\n{esCuadrada(matriz_test)}")

# Triangular superior de la matriz A, con ceros fuera de la diagonal
## EJERCICIO 2
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
  
#print(f"Triangular superior:\n{triangSup(matriz_test)}\nY esta es la original:/n{matriz_test}")

## EJERCICIO 3
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

#print(f"Triangular inferior:\n{triangInf(matriz_test)}\nY esta es la original:\n{matriz_test}")

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

#print(f"Diagonal:\n{diagonal(matriz_test)}\nY esta es la original:\n{matriz_test}")

## Ejercicio 5
def traza(matriz):
    traza = 0
    i = 0
    while i < min(filas(matriz),columnas(matriz)):
        traza += matriz[i][i]
        i += 1
    return traza

#print(f"Traza:\n{traza(matriz_test)}\nY esta es la original:\n{matriz_test}")

## EJERCICIO 6
def traspuesta(matriz):
    matriz = np.array(matriz)
    T = np.zeros((matriz.shape[1],matriz.shape[0]))
    for i in range(0,matriz.shape[0]):
        for j in range(0,matriz.shape[1]):
                T[j][i] = matriz[i][j]
    return np.array(T)

#print(f"Traspuesta:\n{traspuesta(matriz_test)}\nY esta es la original:\n{matriz_test}")

## EJERCICIO 7 (Solo valida para matrices cuadradas!!)
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

#print(f"La matriz\n{matriz_test}\nEs simetrica?:{esSimetrica(matriz_test)}")

## EJERCICIO 8 (MULTIPLICACION VECTORIAL MATRIZ . VECTOR )
# Recibo matriz de tamano nxm y un vector x de tamano m(mas especificamente el vector columna); calcularAx devuelve vector b de tamano n (1 columna de n filas)
#filaxcolumna
# arreglar calcularAx para que acepte vectores fila tambn
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

#print(f"La multiplicacion vectorial de la matriz\n{matriz_test}\npor el vector columna\n{vector_columna}\nes:\n{calcularAx(matriz_test,vector_columna)}\nY la original sigue como antes:\n{matriz_test}")

## EJERCICIO 9  (Desarrollar una funcion intercambiarFilas(A, i, j), que intercambie las filas i y la j de la matriz A. El intercambio tiene que ser in-place.)
def intercambiarFilas(matriz, i, j):
    a_la_fila_j = matriz[i].copy()
    matriz[i] = matriz[j]
    matriz[j] = a_la_fila_j 
    return matriz

#print(f"Intercambio de filas en la matriz\n{matriz_test}\n Se cambia la fila 3 por la fila 1:\n{intercambiarFilas(matriz_test,3,1)}")

## EJERCICIO 10 (Desarrollar una funcion sumar_fila_multiplo(A, i, j, s)que a la fila i le sume la fila j multiplicada por un escalar s)
# Esta es una operacion elemental clave en la eliminacion gaussiana. La operacion debe ser in-place.)

def sumar_fila_multiplo(matriz, i, j, escalar):
    for c in range(0,columnas(matriz)):
        matriz[i][c] = matriz[i][c] + matriz[j][c]*escalar
    return matriz

#print(f"Suma de la fila_i + (escalar * fila_j) de la matriz:\n{matriz_test}\nEs:\n{sumar_fila_multiplo(matriz_test,1,3,2)}")

## EJERCICIO 11 
# Desarrollar una funcion esDiagonalmenteDominante(A) que devuelva True si una matriz cuadrada A es estrictamente diagonalmente dominante. 
#Esto ocurre si para cada fila, el valor absoluto del elemento en la diagonal es mayor que la suma de los valores absolutos de los demas elementos en esa fila
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

#print(f"Es la matriz:\n{matriz_test}\ndiagonalmente Dominante?:{esDiagonalmenteDominante(matriz_test)}")

## EJERCICIO 12
# Desarrollar una funcion matrizCirculante(v) que genere una matriz circulante a partir de un vector. 
# En una matriz circulante la primer fila es igual al vector v, y en cada fila se encuentra una permutacion cıclica de la fila anterior, moviendo los elementos un lugar hacia la derecha.
def matrizCirculante(vector):
    C = [vector[0]]
    size = vector.size
    for i in range(1,size):
        C.append([C[i-1][size - 1]])
        for j in range(1,size):
            C[i].append(C[i-1][j-1])
    return np.array(C)

#print(f"La matriz circulante del vector fila:\n{vector_fila}\nEs:\n{matrizCirculante(vector_fila)}")

## EJERCICIO 13
# vector ∈ R_n y se devuelve la matriz de Vandermonde V ∈ Rn×n cuya fila i-esima corresponde con la potencias (i − 1)-esima de los elementos de vector
''' LA PRIMERA POSICION EN CADA FILA EMPIEZA CON EL ELEM[i][0] ** 0'''
def matrizVandermonde(vector):
    V = []
    for i in range (0,filas(vector)):
        elem_v = vector[i][0]
        V.append([elem_v ** 0])
        for j in range(1,filas(vector)):
            V[i].append(elem_v ** j)
    return np.array(V)

#print(f"La matriz Vandermonde del vector\n{vector_columna}\n es:\n{matrizVandermonde(vector_columna)}")

## EJERCICIO 14
# Que estime el numero aureo ϕ como Fk+1/Fk, siendo Fk el k-esimo numero de la sucesion de Fibonacci.
# Para esto, formulen la sucesion de Fibonacci Fk+1 = Fk +Fk−1 de forma matricial, usando la semilla F0 = 0,F1 = 1.
def fibonacci_matricial(n):
    A = np.array([[1,1],[1,0]])
    F = np.array([[1,1],[1,0]])
    if (n == 0):
        return 0 
    if (n == 1):
        return 1
    for i in range(0,n-1):
        F = multiplicar_matrices(F,A)
    return F[0][1]

#print(f"El fibonaacci de 23 es: {fibonacci_matricial(23)}")
# Grafique el valor aproximado de ϕ en funcion del numero de pasos de la sucesion considerado.

def numeroAureo(n):
    if (n == 0):
        return print("infinito :p")
    return fibonacci_matricial(n+1)/fibonacci_matricial(n)

#Defino variables a usar en el grafico
f = []
a = []
n = 10
for i in range(0,n):
    f.append(int(fibonacci_matricial(i)))
    a.append(numeroAureo(i))

#print(f"fibonacci ={f}; numero aureo ={a}")
plt.figure(figsize=(8, 5))  # Creamos el lienzo para el gráfico
plt.plot(f, a, marker='o')
plt.title("Aproximación del número áureo con Fibonacci")
plt.xlabel("Paso n")
plt.ylabel("F(n+1)/F(n)")
plt.grid()
#plt.show()

''' EJERCICIO 15 '''
def matrizFibonacci(n):
    A = matriz_ceros(n,n)
    for i in range(0,n):
        for j in range(0,n):
            A[i][j] = fibonacci_matricial(i+j)
    return np.array(A)

#print(f"La matriz de fibonacci de n filas y columnas es:\n{matrizFibonacci(3)}")

''' EJERCICIO 16 '''
# Genera una matriz de Hilbert H de n×n, y cada h_ij = 1/i+j+1.
def matrizHilbert(n):
    H = matriz_ceros(n,n)
    for i in range(0,n):
        for j in range(0,n):
            H[i][j] = 1/(1+i+j)
    return np.array(H)

#print(f"La matriz de Hilbert de n filas y columnas es:\n{matrizHilbert(5)}")

''' EJERCICIO 17 '''
# Escriba una rutina que calcule los valores entre -1 y 1 de los siguientes polinomios:
# Funciones reutilizables
def crear_rango(pasos):
    rango_x = [-1]
    i = 0
    while i<pasos:
        if rango_x[i] >= 1:
            i = pasos
        else:
            rango_x += [rango_x[i]+(1/(pasos//2))]
            i += 1
    return rango_x

def valores_y(rango_x,polinomio):
    valores = []
    for i in range(0,len(rango_x)):
        x = rango_x[i]
        y = 0
        for e in range(0,len(polinomio)):
            y += polinomio[e]*(x**e)
        valores += [y]
    return valores

# I)     x5 −x4 +x3 −x2 +x −1
pasos = 10
rango_x = crear_rango(pasos)
polinomio_i = [-1,1,-1,1,-1,1]
y1 = valores_y(rango_x,polinomio_i)

# II)    x2 +3
polinomio_ii = [3,0,1]
y2 = valores_y(rango_x,polinomio_ii)

# III)   x10 −2
polinomio_iii = [-2,0,0,0,0,0,0,0,0,0,1]
y3 = valores_y(rango_x,polinomio_iii)
 
plt.plot(rango_x, y1, marker='o',label = "x5 −x4 +x3 −x2 +x −1")
plt.plot(rango_x, y2, marker='o',label = "x2 +3")
plt.plot(rango_x, y3, marker='o',label = "x10 −2")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7) # agrega linea en y=0
plt.title("Valores entre -1 y 1 del polinomio I")
plt.xlabel("Valores de x")
plt.ylabel("Polinomio en el valor x")
plt.grid()
#plt.show()

''' EJERCICIO 18 ??? (row_echelon esta en el video de la teo03, es una funcion para triangular una amtriz A)'''
''' Modificar la funcion row echelon de manera que evalue en cada pivot si no hay otro elemento de la misma columan con modulo 
mayor (en valor absoluto). En caso afirmativo hacer el swap de las filas. Esta operatoria permite tener mayor estabilidad numerica.'''





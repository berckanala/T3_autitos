import pandas as pd
import numpy as np

def furness(t, O, D, tol=1e-6, maxit=1000):  
    k = len(O)
    O = np.array(O)[:, np.newaxis].astype(float)
    D = np.array(D).astype(float)
    t = np.array(t)
    ai = np.ones((k, 1))
    bj = np.ones((1, k))
    iters = 0
    while iters < maxit:
        row_sums = t.sum(axis=1)
        col_sums = t.sum(axis=0)
        row_sums[row_sums == 0] = 1
        col_sums[col_sums == 0] = 1
        ai = O / row_sums[:, np.newaxis]
        t = t * ai
        col_sums = t.sum(axis=0)
        col_sums[col_sums == 0] = 1
        bj = D / col_sums
        t = t * bj
        row_sums_after = t.sum(axis=1)
        col_sums_after = t.sum(axis=0)
        
        if np.max(np.abs(row_sums_after - O.squeeze())) < tol and \
           np.max(np.abs(col_sums_after - D)) < tol:
            print(f"Convergencia alcanzada en {iters + 1} iteraciones.")
            break
        iters += 1
    if iters == maxit:
        print("Advertencia: se alcanzó el número máximo de iteraciones sin convergencia.")
    return t

# Crear los datos de las tablas
data = {
    "Zona O\\D": ["301", "302", "308", "314", "316", "318", "E1", "E2", "E3", "E4"],
    "301": [0, 0, 0, 0, 0, 0, 709, 0, 714, 821],
    "302": [284, 845, 0, 0, 1202, 0, 369, 894, 3514, 3671],
    "308": [0, 0, 107, 0, 0, 0, 39, 0, 1457, 1886],
    "314": [0, 0, 0, 0, 0, 0, 126, 25, 1208, 949],
    "316": [0, 171, 0, 0, 0, 0, 37, 97, 728, 344],
    "318": [0, 0, 0, 0, 108, 0, 107, 0, 529, 650],
    "E1": [811, 1622, 0, 0, 193, 0, 0, 0, 0, 0],
    "E2": [98, 836, 0, 25, 0, 0, 0, 0, 0, 0],
    "E3": [121, 1029, 1563, 0, 529, 0, 0, 0, 0, 0],
    "E4": [645, 1663, 3480, 0, 338, 0, 0, 0, 0, 0]
}

df1_1 = pd.DataFrame(data).set_index("Zona O\\D")
EOD = df1_1.values

data2 = {
    "Zona O\\D": ["301", "302", "308", "314", "316", "318", "E1", "E2", "E3", "E4"],
    "301": [0.5, 1.85, 1.53, 3.11, 2.22, 0.77, 9.71, 5.15, 8.85, 16.01],
    "302": [1.85, 1.31, 2.69, 2.22, 1.56, 1.32, 9.23, 6.13, 7.39, 14.78],
    "308": [1.53, 2.69, 1.25, 1.81, 1.81, 2.31, 9.02, 6.74, 6.05, 13.56],
    "314": [3.11, 2.22, 1.81, 0.65, 1.25, 2.95, 7.04, 5.55, 6.8, 13.12],
    "316": [2.22, 1.56, 1.81, 1.25, 0.99, 2.18, 7.74, 5.18, 7.43, 14.04],
    "318": [0.77, 1.32, 2.31, 2.95, 2.18, 0.5, 9.78, 5.97, 9.01, 15.82],
    "E1": [9.71, 9.23, 9.02, 7.04, 7.74, 9.78, np.inf, np.inf, np.inf, np.inf],
    "E2": [5.15, 6.13, 6.74, 5.55, 5.18, 5.97, np.inf, np.inf, np.inf, np.inf],
    "E3": [8.85, 7.39, 6.05, 6.8, 7.43, 9.01, np.inf, np.inf, np.inf, np.inf],
    "E4": [16.01, 14.78, 13.56, 13.12, 14.04, 15.82, np.inf, np.inf, np.inf, np.inf],
}

cost_df = pd.DataFrame(data2).set_index("Zona O\\D")
Cij = cost_df.values
data3 = {
    "Zona": ["Oi,2024", "Dj,2024"],
    "301": [7737, 1744],
    "302": [15089, 5736],
    "308": [6632, 4995],
    "314": [2308, 20],
    "316": [4425, 2301],
    "318": [2784, 1],
    "E1": [2464, 3191],
    "E2": [899, 2337],
    "E3": [3042, 15240],
    "E4": [5746, 15561],
}

data4={
    "Zona":["Oi,2012", "Dj,2012"],
    "301":[1960, 2245],
    "302":[6166,10780],
    "308":[5150, 3488],
    "314":[25, 2307],
    "316":[2371, 1377],
    "318":[1, 1395],
    "E1":[1387, 2626],
    "E2":[1016, 959],
    "E3":[8150, 3243],
    "E4":[8322, 6126]
}

# Convertir 'data' a una matriz numérica
matrix = pd.DataFrame(data).drop(columns="Zona O\\D").astype(float).values


# Extraer los vectores Oi y Dj de 'data3'
O2024 = pd.DataFrame(data3).set_index("Zona").loc["Oi,2024"].values
D2024 = pd.DataFrame(data3).set_index("Zona").loc["Dj,2024"].values


# Asegurarse de que las dimensiones coincidan
if matrix.shape[0] != len(O2024) or matrix.shape[1] != len(D2024):
    raise ValueError("Las dimensiones de la matriz y los vectores O y D no coinciden.")

# Aplicar el algoritmo Furness
resultado = furness(matrix, O2024, D2024)

# Convertir el resultado en un DataFrame para agregar la fila y columna de sumas
resultado_df = pd.DataFrame(resultado, index=df1_1.index, columns=df1_1.columns)

# Calcular las sumas de las filas y columnas
row_sums = resultado_df.sum(axis=1)
col_sums = resultado_df.sum(axis=0)

# Agregar las sumas como la última fila y columna
resultado_df['Oi,2024'] = row_sums
total_column = pd.concat([col_sums, pd.Series(row_sums.sum(), index=['Oi,2024'])])
resultado_df.loc['Dj, 2024'] = total_column

#--------------------------------------------------------------------


# Parámetros dados
beta = 0.2176

k = 0.1

# Ejemplo de DataFrames de entrada (asegúrate de que cost_df y df1_1 estén previamente definidos)
# cost_df: DataFrame con la matriz de costos (Cij)
# df1_1: DataFrame con la matriz EODij

# Inicializar una lista vacía para almacenar los resultados
Tij_list = []
O = pd.DataFrame(data4).set_index("Zona").loc["Oi,2012"].values
D = pd.DataFrame(data4).set_index("Zona").loc["Dj,2012"].values
# Recorrer cada combinación i, j para calcular Tij
for i in range(10):  # Asumiendo que hay 10 zonas (ajusta si es necesario)
    row = []
    for j in range(10):
        cij = cost_df.iloc[i, j]  # Costo de la celda i, j
        EODij = df1_1.iloc[i, j]  # Valor de EODij en la celda i, j
        Tij =  O[i]*D[j] * (cij ** -k) * np.exp(-beta * cij)  # Cálculo de Tij
        row.append(Tij)  # Añadir el resultado a la fila
    Tij_list.append(row)  # Añadir la fila a la lista principal

denominador=0
for i in range(10):  # Asumiendo que hay 10 zonas (ajusta si es necesario)
    for j in range(10):
        denominador += Tij_list[i][j]
numerador = O.sum()

alpha= numerador/denominador

for i in range(10):  # Asumiendo que hay 10 zonas (ajusta si es necesario)
    for j in range(10):
        Tij_list[i][j] = Tij_list[i][j]*alpha

# Convertir la lista a un DataFrame de pandas
Tij_df = pd.DataFrame(Tij_list, index=cost_df.index, columns=cost_df.columns)

row_sums = Tij_df.sum(axis=1)
col_sums = Tij_df.sum(axis=0)
Tij_df['Oi,2024'] = row_sums
total_column = pd.concat([col_sums, pd.Series(row_sums.sum(), index=['Oi,2024'])])
Tij_df.loc['Dj, 2024'] = total_column
# Mostrar el DataFrame resultante
print("Resultado después de matriz gravitacional:\n", Tij_df)



#--------------------------------------------------------------------------------
# Eliminar la última fila y columna de Tij_df
df = Tij_df.iloc[:-1, :-1]

# No intentes eliminar la columna "Zona O\\D" si ya no existe en el DataFrame
# Si solo deseas convertir el DataFrame a una matriz numérica para usar en el algoritmo Furness:
Tij = df.astype(float).values

# Ahora aplica el algoritmo Furness
resultado = furness(Tij, O2024, D2024)

# Mostrar el resultado en formato DataFrame para verificar
resultado_df = pd.DataFrame(resultado, index=df.index, columns=df.columns)


#--------------------------------------------------------------------------------
# Reemplazar NaN por 0 en Tij_df
Tij_df = Tij_df.fillna(0)
# Reemplazar NaN por 0 en df1_1 (por si acaso contiene NaN también)
df1_1 = df1_1.fillna(0)
# Ahora puedes continuar con el cálculo del MSE
mse = np.mean((Tij_df - df1_1) ** 2)/100

# Mostrar el resultado del MSE
print(f"El error cuadrático medio (MSE) es: {mse:.4f}")

#--------------------------------------------------------------------------------

row_sums = resultado_df.sum(axis=1)
col_sums = resultado_df.sum(axis=0)
resultado_df['Oi,2024'] = row_sums
total_column = pd.concat([col_sums, pd.Series(row_sums.sum(), index=['Oi,2024'])])
resultado_df.loc['Dj, 2024'] = total_column


print("Resultado después de aplicar Furness:\n", resultado_df)





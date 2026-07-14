
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from scipy.spatial.distance import cdist

from sklearn.preprocessing import MinMaxScaler


def bloques_4h(df):
    
    # Agrupamos los elementos del dataframe por fecha,
    filas = []
    for dia, grupo in df.groupby(df.fecha,sort=False):
        # Para cada día construimos el vector de 144 elementos que compondría 
        # el día completo.
        perfil_dia = grupo["consumo"].to_numpy(dtype=float)        
        timestamps_dia = grupo.index.to_numpy()
        # Dividimos este vector en 6 partes de manera que podamos construir los
        # 6 bloques de 4 horas que componen cada día.
        for b in range(6):
            inicio = b*24
            filas.append({"inicio": timestamps_dia[inicio],
                          "vector": perfil_dia[inicio:inicio+24] })        
        
    df_bloques = pd.DataFrame(filas)

    return df_bloques

def encontrar_vecinos(df, hora_bloque, k, metrica = "euclidean"):
    
    df_bloques =  df.copy()

    # Buscamos el índice del bloque anterior al que queremos predecir que es
    # el que se pasa como argumento.
    indice_bloque = df_bloques.index[df_bloques["inicio"] == hora_bloque][0]
    # Nos quedamos con el perfil de dicho bloque.
    perfil_consulta = df_bloques.loc[indice_bloque, "vector"]
    # Sacamos los consumos de todos los días del dataset.
    matriz_historico = np.vstack(df_bloques["vector"].to_numpy())    
    # Calculamos la distancia que se indique como parámetro de cada perfil al 
    # del bloque anterior del que queremos predecir.
    distancias = cdist(
        matriz_historico,
        perfil_consulta.reshape(1, -1),
        metric=metrica
    ).ravel()

    # Guaradamos estas distancias en el DataFrame.
    df_bloques["distancias"] = distancias
    
    # Eliminamos del DataFrame el perfil del bloque del que estamos buscando
    # los vecinos ya que él mismo será siempre su vecino mas cercano a una
    # distancia de 0.0 y el perfil del último bloque del dataset. No nos vale
    # para nada como vecino ya que no tenemos información del siguiente bloque.
    # Ordenamos en función del valor de la distanicia.
    df_sin_filas = df_bloques.sort_values("distancias").drop(  
                        index=[indice_bloque,df_bloques.shape[0]-1]
                        ).copy()
    
    # Sacamos la lista de índices presentes en el dataset.
    indices_df = df_sin_filas.index.tolist()
    vecinos = []
    # Iteramos sobre los diferentes vecinos para comprobar si es viable usarlos.
    for vecino in df_sin_filas.index:
        # Para cada vecino encontrado calculamos el índice del siguiente bloque.
        indice_siguiente = vecino + 1
        # Si este bloque siguiente está en el dataset nos sirve como vecino.
        if(indice_siguiente in indices_df):
            # Guardamos el índice del vecino, el del bloque siguiente, el perfil
            # del bloque siguiente que es el que nos interesa y la distancia del
            # vecino encontrado ya que la necesitamos para calcular el peso de
            # cada bloque al realizar la predicción.
            vecinos.append({
                "indice": vecino,
                "indice_siguiente": indice_siguiente,
                "vector": df_bloques.loc[indice_siguiente,"vector"],
                "distancias": df_bloques.loc[vecino,"distancias"]
                })
            # Si ya hemos guardado k vecinos dejamos de iterar sobre el conjunto
            # de estos.
            if(len(vecinos) == k):
                break
    #Tenemos los vecinos del bloque anterior al que queremos predecir. 
    #Necesitamos quedarnos con el bloque siguiente de cada vecino.
    #indices_siguientes = [i + 1 for i in df_vecinos.index.tolist()]
    df_siguientes = pd.DataFrame(vecinos)
    
    return df_siguientes

def calcular_pesos(df_vecinos):

    # Sacamos el listado (que está ordenado por distancia) de las distancias de
    # de los vecinos.
    distancias = df_vecinos["distancias"].to_numpy()
    
    # Guardamos las distancias del vecino mas lejano y del mas cercano.
    distancia_mas_lejano = distancias[len(distancias)-1]
    distancia_mas_cercano = distancias[0]
    
    # Calculamos los pesos que corresponde a cada vecino
    pesos = (distancia_mas_lejano - distancias)/(distancia_mas_lejano-distancia_mas_cercano)
    
    # Asignamos a cada entrada su peso.
    df_vecinos["peso"] = pesos

    return df_vecinos

def predecir_bloque_siguiente_kWNN(df, hora_bloque, k, metrica):

    df_bloques = df.copy()
    
    # Buscar los bloque siguientes de los k vecinos más próximos al bloque cuya 
    # hora de inicio se indica en el parámetro hora_consulta.
    df_vecinos = encontrar_vecinos(
        df_bloques,
        hora_bloque,
        k,
        metrica
    )

    # Calcular el peso asociado a cada vecino
    df_vecinos = calcular_pesos(df_vecinos)

    # Inicializar el vector que representa el bloque a predecir.
    prediccion = np.zeros(df_bloques["vector"].apply(len)[0])

    # Calculamos la predicción usando la media ponderada por los pesos de los 
    # consumos de cada vecino.
    prediccion = np.sum(df_vecinos["vector"].to_numpy() * df_vecinos["peso"].to_numpy(), axis=0)/np.sum(df_vecinos["peso"].to_numpy(),axis=0)

    return prediccion
   
def calcular_metricas(real, prediccion):

    # Pasamos a vectores de NumPy los bloques para facilitar los cálculos
    real = np.asarray(real, dtype=float)
    prediccion = np.asarray(prediccion, dtype=float)

    # Calculamos los errores.
    mae = mean_absolute_error(real, prediccion)
    mse = mean_squared_error(real, prediccion)
    mre = np.mean(
        np.abs(real - prediccion)
        / real
    )

    return {
        "mae": mae,
        "mse": mse,
        "mre": mre
    }

def calibrar_parametros(df_bloques, fecha_inicio_test, k_values, metricas):
    
    # División del dataset en el conjunto de entrenamiento (espacio de busqueda)
    # y el de test (cálculo de errores en las predicciones)
    df_train = df_bloques[df_bloques["inicio"] < fecha_inicio_test].copy()
    df_test = df_bloques[df_bloques["inicio"] >= fecha_inicio_test].copy()

    resultados = []
    # Iteramos sobre cada elemento del conjunto de test para predecir el bloque
    # que se correspondería con dicho timestamp.
    for _, fila_test in df_test.iterrows():
        # Sacamos las horas del bloque de conjunto de test que vamos a predecir,
        # la hora del bloque anterior a este que necesitamos para realizar la 
        # predicción y el perfil  real del valor a predecir para comprobar los
        # errores cometidos.
        hora_consulta = fila_test["inicio"]
        hora_bloque = df_train["inicio"].iloc[-1]
        real = fila_test["vector"]
        
        print(f"Estimado bloque: {hora_consulta}", end="\r", flush=True)
        # Iteramos sobre todos los pares de valores de los parámetros.
        for k in k_values:
            for metrica in metricas:
                # Predicción del bloque
                prediccion = predecir_bloque_siguiente_kWNN(df_train, hora_bloque, k, metrica)
                # Calculo de erores cometidos en la predicción.
                metricas_error = calcular_metricas(real, prediccion)
                # Se guardan los resultados de la evaluación.
                resultados.append({
                    "k": k,
                    "metrica": metrica,
                    "inicio": hora_consulta,
                    "fecha": hora_consulta.date,
                    "hora": hora_consulta.time,
                    "mae": metricas_error["mae"],
                    "mse": metricas_error["mse"],
                    "mre": metricas_error["mre"]
                })
        
        # Añadimos el bloque real al conjunto de train para no arrastrar el error
        # al predecir el siguiente bloque
        df_train = pd.concat([df_train, fila_test.to_frame().T], ignore_index=True)
    
    df_resultados = pd.DataFrame(resultados)
    print("Completado!                               ")
    return df_resultados

def mejores_calibraciones(resumen_parametros):
    mejores_resultados = []
    
    # Agrupamos los resultados por métricas e iteramos por ellos.
    for metrica, resumen in resumen_parametros.groupby("metrica", sort=False):
        # Buscamos la entrada del df de resultados que minimiza la metrica MAE
         mejores_resultados.append(
             pd.Series(
                 resumen.loc[resumen["mae"].idxmin()],
                 name=f"mejor_mae_{metrica}"
             )
         )
         # Buscamos la entrada del df de resultados que minimiza la metrica MSE
         mejores_resultados.append(
             pd.Series(
                 resumen.loc[resumen["mse"].idxmin()],
                 name=f"mejor_mse_{metrica}"
             )
         )
         # Buscamos la entrada del df de resultados que minimiza la metrica MRE
         mejores_resultados.append(
             pd.Series(
                 resumen.loc[resumen["mre"].idxmin()],
                 name=f"mejor_mre_{metrica}"
             )
         )
    
    mejores_resultados = pd.DataFrame(mejores_resultados)
    return mejores_resultados

def realizar_predicciones(df_bloques,k,metrica):
    
    # Consturimos el conjunto de entrenamiento y el de validación.
    fecha_inicio_test = pd.to_datetime("2015-06-8 00:00:00")
    fecha_fin_test = pd.to_datetime("2015-06-15 00:00:00")
    # Como espacio de búsqueda nos quedamos con todo el dataset a excepción de
    # los bloques de la semana entre el 8 y el 14 de Junio.
    df_train = df_bloques[(df_bloques["inicio"] < fecha_inicio_test) |
                          (df_bloques["inicio"] >= fecha_fin_test)].copy()
    # Como conjunto de validación sobre el que haremos las predicciones aislamos
    # los bloques de la semana entre el 8 y el 14 de Junio.
    df_test = df_bloques[(df_bloques["inicio"] >= fecha_inicio_test) &
                         (df_bloques["inicio"] < fecha_fin_test)].copy()
    
    historico = df_train.copy()
    resultados = []

    # Iteramos sobre todos los bloques del conjunto de validación
    for _, fila_test in df_test.iterrows():
        # Sacamos las horas del bloque de conjunto de test que vamos a predecir,
        # la hora del bloque anterior a este que necesitamos para realizar la 
        # predicción y el perfil  real del valor a predecir para comprobar los
        # errores cometidos.
        hora_consulta = fila_test["inicio"]
        hora_bloque = df_train["inicio"].iloc[-1]
        real = fila_test["vector"]
        # Predicción del bloque.
        prediccion = predecir_bloque_siguiente_kWNN(historico, hora_bloque, k, metrica)
        # Calculo de erores cometidos en la predicción.
        metricas_error = calcular_metricas(real, prediccion)
        
        # Guardamos los resultados obtenidos.
        resultados.append({
            "k": k,
            "metrica": metrica,
            "inicio": hora_consulta,
            "fecha": hora_consulta.date(),
            "hora": hora_consulta.time(),
            "prediccion": prediccion,
            "mae": metricas_error["mae"],
            "mse": metricas_error["mse"],
            "mre": metricas_error["mre"]
        })
        
        print(f"Estimado bloque: {hora_consulta}", end="\r", flush=True)
        # Añadimos al espacio de búsqueda de vecinos el perfil real del bloque 
        # que acabamos de predecir para no arrastrar el error de predicción.
        historico = pd.concat([historico, fila_test.to_frame().T], ignore_index=True)

    df_resultados = pd.DataFrame(resultados)
    # Añadimos el dia de la semana al que se corresponde cada bloque
    df_resultados["dia_semana"] = pd.to_datetime(df_resultados["inicio"]).dt.dayofweek
    dias = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo"
    }
    df_resultados["nombre_dia"] = df_resultados["dia_semana"].map(dias)
    
    print("Completado!                             ")
    return df_resultados

def graficar_metricas_comparadas(resumen_parametros):
    df_euclidean = resumen_parametros[
        resumen_parametros["metrica"] == "euclidean"
    ].copy()

    df_cityblock = resumen_parametros[
        resumen_parametros["metrica"] == "cityblock"
    ].copy()

    df_euclidean = df_euclidean.sort_values("k")
    df_cityblock = df_cityblock.sort_values("k")
    
    scaler = MinMaxScaler()
    df_cityblock[["mae_norm", "mse_norm"]] = scaler.fit_transform(
        df_cityblock[["mae", "mse"]]
    )
    
    df_euclidean[["mae_norm", "mse_norm"]] = scaler.fit_transform(
        df_euclidean[["mae", "mse"]]
    )

    fig, ax1 = plt.subplots(figsize=(12, 7))

    color_mae = "tab:blue"
    color_mse = "tab:orange"
    color_mre = "tab:green"

    lineas = []

    lineas += ax1.plot(
        df_euclidean["k"],
        df_euclidean["mae_norm"],
        color=color_mae,
        linestyle="-",
        marker="x",
        label="MAE Euclídea"
    )

    lineas += ax1.plot(
        df_cityblock["k"],
        df_cityblock["mae_norm"],
        color=color_mae,
        linestyle="--",
        marker="o",
        label="MAE Manhattan"
    )

    lineas += ax1.plot(
        df_euclidean["k"],
        df_euclidean["mse_norm"],
        color=color_mse,
        linestyle="-",
        marker="x",
        label="MSE Euclídea"
    )

    lineas += ax1.plot(
        df_cityblock["k"],
        df_cityblock["mse_norm"],
        color=color_mse,
        linestyle="--",
        marker="o",
        label="MSE Manhattan"
    )

    ax1.set_xlabel("k")
    ax1.set_ylabel("MAE / MSE normalizados")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    lineas += ax2.plot(
        df_euclidean["k"],
        df_euclidean["mre"],
        color=color_mre,
        linestyle="-",
        marker="x",
        label="MRE Euclídea"
    )

    lineas += ax2.plot(
        df_cityblock["k"],
        df_cityblock["mre"],
        color=color_mre,
        linestyle="--",
        marker="o",
        label="MRE Manhattan"
    )

    ax2.set_ylabel("MRE")

    etiquetas = [linea.get_label() for linea in lineas]
    ax1.legend(lineas, etiquetas, loc="best")

    plt.title("Comparación de métricas para las distancias euclídea y Manhattan")
    plt.tight_layout()
    plt.show()
    
def graficar_histograma_errores(df, columna_x, titulo, etiqueta_x):

    df = df.copy()

    scaler = MinMaxScaler()
    df[["mae_norm", "mse_norm"]] = scaler.fit_transform(
        df[["mae", "mse"]]
    )

    x = np.arange(len(df))
    width = 0.25
    
    color_mae = "tab:blue"
    color_mse = "tab:orange"
    color_mre = "tab:green"

    fig, ax1 = plt.subplots(figsize=(12, 7))

    barras_mae = ax1.bar(
        x - width,
        df["mae_norm"],
        width,
        color=color_mae,
        label="MAE (normalizado)"
    )

    barras_mse = ax1.bar(
        x,
        df["mse_norm"],
        width,
        color=color_mse,
        label="MSE (normalizado)"
    )

    ax1.set_xlabel(etiqueta_x)
    ax1.set_ylabel("MAE / MSE normalizados")
    ax1.set_ylim(0, 1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df[columna_x])

    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()

    barras_mre = ax2.bar(
        x + width,
        df["mre"],
        width,
        color=color_mre,
        label="MRE"
    )

    ax2.set_ylabel("MRE")

    handles = [barras_mae, barras_mse, barras_mre]
    labels = [
        "MAE (normalizado)",
        "MSE (normalizado)",
        "MRE"
    ]

    ax1.legend(handles, labels)

    plt.title(titulo)
    plt.tight_layout()
    plt.show()

def visualizar_resultados(resultados, metrica):
    
    # Construimos el df de errores medios por hora
    errores_hora = (
     resultados
     .groupby("hora")[["mae","mse","mre"]]
     .mean()
     .reset_index()
     )

    graficar_histograma_errores(
     errores_hora,
     "hora",
     f"Errores medios por bloque horario ({metrica})",
     "Hora de inicio del bloque"
     )
    
    # Construimos el df de errores medios por día de la semana
    errores_dia = (
     resultados
     .groupby(["dia_semana","nombre_dia"])[["mae","mse","mre"]]
     .mean()
     .reset_index()
     .sort_values("dia_semana")
     )
    
    graficar_histograma_errores(
     errores_dia,
     "nombre_dia",
     f"Errores medios por día de la semana ({metrica})",
     "Día de la semana"
     )

def main():
   #Cargamos el dataset identificando las columnas  fecha hora y consumo
   df = pd.read_csv(
       "Demanda_2015.txt",
       sep="\t",
       header=None,
       names=["fecha", "hora", "consumo"]
   )
    
   #Creamos la columna timestamp para que cada consumo quede identificado
   #por el instante en el que se registró
   df["timestamp"] = pd.to_datetime(
       df["fecha"] + " " + df["hora"],
       format="%d/%m/%y %H:%M"
   )

   df.set_index("timestamp", inplace=True)
    
   #Ordenamos el datset por este timestamp para asegurarnos que la serie temporal
   #está correctamente construida.
   df.sort_index(inplace=True)
   
   # Construimos los vectores de 24 elementos para tener bloques de 4 horas.
   df_bloques = bloques_4h(df)
   
   ########### Calibración de parámetros
   
   # Definimos el listado de valores de cada parámetro a probar. El número de
   # vecinos va de 2 a 20 y las distancias a utilizar serán la euclidea y 
   # manhattan
   k_values = np.arange(2,21)
   metricas = ("euclidean", "cityblock")
   
   # Indicamos el timestamp del bloque en el que comenzará nuestro conjunto de
   # test
   fecha_inicio_test = pd.to_datetime("2015-12-1 00:00:00")

   # Ejecutamos la calibración de parámetros y obtenemos los errores medios de 
   # cada combinación de valores al predecir cada bloque
   resultados = calibrar_parametros(df_bloques, fecha_inicio_test, k_values, metricas)
   
   # Calculamos los erores medios cometidos en las predicciones de todo el conjunto
   # de test distinguiendo por pares de valores de parámetros
   resumen_parametros = (
        resultados
        .groupby(["k", "metrica"])[["mae", "mse", "mre"]]
        .mean()
        .reset_index()
        )
   
   # Mostramos la evolución de cada medida de error obtenida con cada distancia
   # al aumentar el valor de k.
   graficar_metricas_comparadas(resumen_parametros)
      
   # Extraemos los valores de k que menores errores han cometido por cada distancia
   # y por cada una de las métricas empleadas.
   mejores_resultados = mejores_calibraciones(resumen_parametros)
   
   print("\n------ Mejores valores de cada métrica por distancia ------\n")
   print(mejores_resultados)
   
   # Nos quedamos con los mejores resultados de cada distancia.
   errores_hist = mejores_resultados.loc[
       ["mejor_mae_cityblock", "mejor_mae_euclidean"]
    ].copy()
   
   graficar_histograma_errores(
    errores_hist,
    "metrica",
    "Errores medios por métrica",
    "Métrica"
    )
   
   ######### Predicción de la semana del 8 al 14 de junio
   
   # Realizamos las predicciones utilizando la mejor combinación de parámetros
   # que hemos obtenido en la calibración.
   resultados_euclidea = realizar_predicciones(df_bloques,3,"euclidean")
   
   # Visualizamos los errores medios por hora y por día de la semana de la medida
   # de las predicciones usando la distancia Manhattan que es la que mejores resultados
   # ha obtenido
   visualizar_resultados(resultados_euclidea, "Euclidea")
    
if __name__ == '__main__':
    main()
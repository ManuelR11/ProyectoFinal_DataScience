# ProyectoFinal_DataScience

# Proyecto de Análisis de Retrasos en Vuelos

## Descripción

Este proyecto realiza un análisis detallado de los retrasos en vuelos a través de un conjunto de datos de vuelos retrasados. Incluye la carga, limpieza y transformación de datos, seguido de un análisis exploratorio (EDA) y visualizaciones para identificar patrones de retraso. Finalmente, se implementa un modelo de clasificación y un modelo de regresión utilizando árboles de decisión para predecir los retrasos en la llegada.

### Integrantes del Equipo

- Juan M. González-Campo
- Lourdes Saavedra
- Manuel Rodas
- Wilfredo Gallegos
- Dolan Cuellar

## Contenido del Proyecto

1. **Carga de Datos**: Importación del conjunto de datos `DelayedFlights.csv` y presentación de los datos iniciales.
2. **Limpieza de Datos**: Proceso de limpieza que incluye:
   - Conversión de tipos de datos.
   - Eliminación de valores nulos y duplicados.
   - Imputación de valores en columnas de retraso.
3. **Transformación de Variables**: Ajuste de variables y creación de nuevas características, como `flightDate`.
4. **Exploración de Datos (EDA)**: Incluye visualizaciones y estadísticas descriptivas:
   - Histogramas para columnas numéricas.
   - Matriz de correlación entre variables numéricas.
   - Análisis por aerolínea, día de la semana, mes y aeropuertos de origen y destino.
5. **Modelos de Árboles de Decisión**:
   - **Clasificación**: Predice si un vuelo se retrasará más de 15 minutos.
   - **Regresión**: Predice el tiempo de retraso en minutos.
   - Se evalúan diferentes métricas y se utilizan gráficos para visualizar las predicciones y la matriz de confusión.
6. **Optimización de Hiperparámetros**: Uso de `GridSearchCV` para encontrar los mejores hiperparámetros para ambos modelos.

## Requisitos

- Python 3.7 o superior.
- Las siguientes librerías de Python:
  - `pandas`
  - `matplotlib`
  - `numpy`
  - `seaborn`
  - `scikit-learn`

Puedes instalar todas las dependencias usando el siguiente comando:

```bash
pip install pandas matplotlib numpy seaborn scikit-learn
```

## Ejecución

Para ejecutar el análisis y visualizar los gráficos, simplemente corre el script en tu entorno de Python.

```bash
python nombre_del_archivo.py
```


## Visualización en Web usando Streamlit

Este proyecto incluye un segundo archivo compatible con `Streamlit` para visualizar el análisis en una página web interactiva. `Streamlit` permite visualizar gráficos, tablas y métricas directamente en el navegador.

![Proyecto Web](videoProyecto Final Data science.mp4)


### Configuración para Streamlit

1. **Instalar Streamlit**:
   
   Si aún no tienes `Streamlit` instalado, puedes instalarlo ejecutando el siguiente comando en tu terminal:

   ```bash
   pip install streamlit
   ```

2. **Ejecución del archivo Streamlit**:
   
   Para ejecutar el archivo adaptado a `Streamlit`, usa el siguiente comando:

   ```bash
   streamlit run nombre_del_archivo_streamlit.py
   ```

3. **Visualización en el navegador**:

   Una vez que ejecutes el archivo con `Streamlit`, este abrirá una página web donde podrás interactuar con el análisis de datos, visualizar gráficos y explorar los resultados en tiempo real.


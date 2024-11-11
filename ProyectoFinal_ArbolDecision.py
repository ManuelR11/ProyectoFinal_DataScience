import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

# Configuración de la página en Streamlit
st.set_page_config(page_title="Avances Proyecto Final", layout="wide")

st.title("Asistencia - Avances Proyecto Final")
st.subheader("Equipo")
st.write("""
- Juan M. González-Campo
- Lourdes Saavedra
- Manuel Rodas
- Wilfredo Gallegos
- Dolan Cuellar
""")

st.header("Carga de los datos")
df = pd.read_csv('DelayedFlights.csv')
st.write("Datos cargados:", df.head())

# Limpieza de Datos
st.header("Limpieza de Datos")
df = df.iloc[:, 1:]  # Eliminar la primera columna sin nombre
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['DayofMonth'] = df['DayofMonth'].astype(int)
df['DayOfWeek'] = df['DayOfWeek'].astype(int)
df['Cancelled'] = df['Cancelled'].astype(bool)
df['Diverted'] = df['Diverted'].astype(bool)
df.drop_duplicates(inplace=True)
df.dropna(thresh=df.shape[0]/2, axis=1, inplace=True)
df['LateAircraftDelay'].fillna(0, inplace=True)
df['SecurityDelay'].fillna(0, inplace=True)
df['NASDelay'].fillna(0, inplace=True)
df['WeatherDelay'].fillna(0, inplace=True)
df['CarrierDelay'].fillna(0, inplace=True)
df.dropna(subset=['ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'TaxiIn', 'TaxiOut'], inplace=True)

st.write("Datos después de la limpieza:", df.head())

# Exploración de Datos (EDA)
st.header("Exploración de Datos (EDA)")
st.subheader("Transformación de Variables")
df['Cancelled'] = df['Cancelled'].astype('category')
df['FlightNum'] = df['FlightNum'].astype('object')
df['TailNum'] = df['TailNum'].astype('object')
df['flightDate'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['DayofMonth'].astype(str))
df["Month"] = df["Month"].astype('category')
df["DayofMonth"] = df["DayofMonth"].astype('category')
df["DayOfWeek"] = df["DayOfWeek"].astype('category')
df["Year"] = df["Year"].astype('category')
df["UniqueCarrier"] = df["UniqueCarrier"].astype('category')
df["Diverted"] = df["Diverted"].astype('category')

# Creación del modelo de Clasificación
st.header("Árbol de Decisión - Clasificación")
X = df[['Month', 'DayofMonth', 'DayOfWeek', 'DepDelay', 'Distance']]
y_classification = (df['ArrDelay'] > 15).astype(int)
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train_class)

# Guardar el modelo en un archivo
with open("clasificador_vuelo.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)
st.write("Modelo de clasificación exportado como `clasificador_vuelo.pkl`.")

# Solicitar parámetros de entrada al usuario
st.header("Predicción de Retraso de Vuelo")
st.subheader("Ingrese los detalles del vuelo")

month = st.selectbox("Mes", sorted(df['Month'].unique()))
day_of_month = st.selectbox("Día del mes", sorted(df['DayofMonth'].unique()))
day_of_week = st.selectbox("Día de la semana", sorted(df['DayOfWeek'].unique()))
dep_delay = st.number_input("Retraso en salida (en minutos)", min_value=0)
distance = st.number_input("Distancia del vuelo (en millas)", min_value=0)

# Cargar el modelo y realizar predicción
if st.button("Predecir Retraso"):
    with open("clasificador_vuelo.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    
    # Crear un DataFrame con los valores ingresados
    input_data = pd.DataFrame({
        "Month": [month],
        "DayofMonth": [day_of_month],
        "DayOfWeek": [day_of_week],
        "DepDelay": [dep_delay],
        "Distance": [distance]
    })
    
    # Realizar predicción
    prediction = loaded_model.predict(input_data)
    resultado = "El vuelo se retrasará más de 15 minutos." if prediction[0] == 1 else "El vuelo no se retrasará más de 15 minutos."
    st.subheader("Resultado de la Predicción")
    st.write(resultado)

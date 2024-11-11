import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

st.write("Información del DataFrame:", df.info())

# Histogramas
st.subheader("Histogramas de Columnas Numéricas")
fig, ax = plt.subplots(figsize=(20, 15))
df.select_dtypes(include=["float64", "int64"]).hist(ax=ax)
st.pyplot(fig)

# Correlación
st.subheader("Correlación de Variables Numéricas")
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Porcentaje de vuelos atrasados por aerolínea
st.subheader("% de vuelos atrasados por aerolínea")
aerolineas = pd.DataFrame({i: [df[df['UniqueCarrier'] == i].shape[0], df[(df['UniqueCarrier'] == i) & (df['ArrDelay'] > 0)].shape[0]] for i in df['UniqueCarrier'].unique()}, index=['Total', 'Delayed']).T
aerolineas['% de vuelos atrasados'] = aerolineas['Delayed'] / aerolineas['Total'] * 100
aerolineas = aerolineas.sort_values('% de vuelos atrasados', ascending=False)
fig, ax = plt.subplots(figsize=(15, 10))
aerolineas.plot(kind='bar', y='% de vuelos atrasados', ax=ax)
st.pyplot(fig)

# Atrasos promedio por aerolínea
st.subheader("Minutos de atraso promedio por aerolínea")
aerolinea_atrasos_promedio = pd.DataFrame({i: [df[df['UniqueCarrier'] == i]['ArrDelay'].mean(), df[df['UniqueCarrier'] == i]['DepDelay'].mean()] for i in df['UniqueCarrier'].unique()}, index=['ArrDelay', 'DepDelay']).T.sort_values('ArrDelay', ascending=False)
fig, ax = plt.subplots(figsize=(15, 10))
aerolinea_atrasos_promedio.plot(kind='bar', y='ArrDelay', ax=ax, title='Atraso Promedio por Aerolínea (ArrDelay)')
st.pyplot(fig)

# Árbol de decisión: Clasificación
st.header("Árbol de Decisión - Clasificación")
X = df[['Month', 'DayofMonth', 'DayOfWeek', 'DepDelay', 'Distance']]
y_classification = (df['ArrDelay'] > 15).astype(int)
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train_class)
y_pred_class = cross_val_predict(clf, X_test, y_test_class, cv=5)

# Evaluación Clasificación
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)
st.write("Métricas de rendimiento (Clasificación):", f"Accuracy: {accuracy}", f"Precision: {precision}", f"Recall: {recall}", f"F1-Score: {f1}")

# Visualización de la Matriz de Confusión
fig, ax = plt.subplots(figsize=(10, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_class, y_pred_class))
disp.plot(ax=ax, cmap='Blues')
st.pyplot(fig)

# Árbol de decisión: Regresión
st.header("Árbol de Decisión - Regresión")
y_regression = df['ArrDelay']
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.3, random_state=42)
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train_reg)
y_pred_reg = cross_val_predict(reg, X_test, y_test_reg, cv=5)

# Evaluación Regresión
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
mae = mean_absolute_error(y_test_reg, y_pred_reg)
st.write("Métricas de rendimiento (Regresión):", f"RMSE: {rmse}", f"MAE: {mae}")

# Gráfico Predicciones de Regresión vs Valores Reales
fig, ax = plt.subplots(figsize=(10, 5))
sample_size = min(len(y_test_reg), 5000)
indices = np.random.choice(range(len(y_test_reg)), sample_size, replace=False)
ax.scatter(y_test_reg.iloc[indices], y_pred_reg[indices], alpha=0.3)
ax.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', lw=2, color='red')
st.pyplot(fig)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# 1. Cargar datos
print("Cargando datos...")
df = pd.read_csv('dataset_market_final.csv')

# 2. Modelo Lineal (Ventas)
print("Entrenando Lineal...")
X_lin = df[['Precio_Unitario']]
y_lin = df['Cantidad']
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_lin, y_lin)

# 3. Modelo Logístico (Retrasos)
print("Entrenando Logístico...")
le_trafico = LabelEncoder()
df['Trafico_Cod'] = le_trafico.fit_transform(df['Nivel_Trafico'])
X_log = df[['Distancia_KM', 'Trafico_Cod']]
y_log = df['Llega_Tarde']
modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_log, y_log)

# 4. K-Means (Clientes)
print("Entrenando K-Means...")
X_cluster = df[['Edad_Cliente', 'Gasto_Hist_Cliente']]
scaler_kmeans = StandardScaler()
X_scaled = scaler_kmeans.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# 5. Guardar
pack = {
    'modelo_lineal': modelo_lineal,
    'modelo_logistico': modelo_logistico,
    'le_trafico': le_trafico,
    'modelo_kmeans': kmeans,
    'scaler_kmeans': scaler_kmeans
}
joblib.dump(pack, 'modelos_finales.pkl')
print("¡CEREBRO NUEVO CREADO EXITOSAMENTE!")
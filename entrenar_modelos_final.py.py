import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# 1. Cargar la Nueva Base de Datos
print("⏳ Cargando base de datos...")
try:
    df = pd.read_csv('dataset_market_final.csv')
    print("✅ Base de datos cargada correctamente.")
except FileNotFoundError:
    print("❌ ERROR: No se encuentra 'dataset_market_final.csv'.")
    exit()

# --- MODELO 1: REGRESIÓN LINEAL (Predicción de Demanda) ---
print("⚙️ Entrenando Modelo 1: Regresión Lineal...")
X_lin = df[['Precio_Unitario']]
y_lin = df['Cantidad']
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_lin, y_lin)

# --- MODELO 2: REGRESIÓN LOGÍSTICA (Probabilidad de Retraso) ---
print("⚙️ Entrenando Modelo 2: Regresión Logística...")
# Convertir 'Nivel_Trafico' a números
le_trafico = LabelEncoder()
df['Trafico_Cod'] = le_trafico.fit_transform(df['Nivel_Trafico'])

X_log = df[['Distancia_KM', 'Trafico_Cod']]
y_log = df['Llega_Tarde']
modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_log, y_log)

# --- MODELO 3: K-MEANS (Segmentación de Clientes) ---
print("⚙️ Entrenando Modelo 3: K-Means...")
X_cluster = df[['Edad_Cliente', 'Gasto_Hist_Cliente']]

# Escalar datos
scaler_kmeans = StandardScaler()
X_cluster_scaled = scaler_kmeans.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_cluster_scaled)

# --- GUARDAR TODO ---
print("💾 Guardando el archivo inteligente 'modelos_finales.pkl'...")

pack_modelos = {
    'modelo_lineal': modelo_lineal,
    'modelo_logistico': modelo_logistico,
    'le_trafico': le_trafico,      
    'modelo_kmeans': kmeans,
    'scaler_kmeans': scaler_kmeans 
}

joblib.dump(pack_modelos, 'modelos_finales.pkl')

print("🎉 ¡LISTO! Ya tienes el cerebro de tu IA actualizado.")
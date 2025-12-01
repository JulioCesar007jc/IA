import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

print("🔧 Iniciando reparación de modelos...")

# 1. Cargar la data
try:
    df = pd.read_csv('dataset_market_final.csv')
    print("✅ Datos leídos correctamente.")
except:
    print("❌ ERROR: No se encuentra 'dataset_market_final.csv'.")
    exit()

# --- MODELO 1: LINEAL (EL QUE FALLABA) ---
print("⚙️ Entrenando Regresión Lineal (Ventas)...")
X_lin = df[['Precio_Unitario']]
y_lin = df['Cantidad']
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_lin, y_lin) # <--- Aquí es donde aprende

# --- MODELO 2: LOGÍSTICA ---
print("⚙️ Entrenando Logística (Riesgos)...")
le_trafico = LabelEncoder()
df['Trafico_Cod'] = le_trafico.fit_transform(df['Nivel_Trafico'])
X_log = df[['Distancia_KM', 'Trafico_Cod']]
y_log = df['Llega_Tarde']
modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_log, y_log)

# --- MODELO 3: K-MEANS ---
print("⚙️ Entrenando K-Means (Clientes)...")
X_cluster = df[['Edad_Cliente', 'Gasto_Hist_Cliente']]
scaler_kmeans = StandardScaler()
X_scaled = scaler_kmeans.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# --- GUARDAR TODO ---
print("💾 Guardando mochila nueva...")
pack = {
    'modelo_lineal': modelo_lineal, # Guardamos el modelo ya entrenado
    'modelo_logistico': modelo_logistico,
    'le_trafico': le_trafico,
    'modelo_kmeans': kmeans,
    'scaler_kmeans': scaler_kmeans
}

joblib.dump(pack, 'modelos_finales.pkl')
print("🎉 ¡REPARACIÓN COMPLETADA! Ahora intenta usar la App.")
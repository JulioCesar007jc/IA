import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configuración inicial
np.random.seed(42)
num_registros = 1500  # Más datos para que los modelos aprendan mejor

# 1. Generar Fechas (Último año)
fecha_inicio = datetime(2024, 1, 1)
fechas = [fecha_inicio + timedelta(days=np.random.randint(0, 365)) for _ in range(num_registros)]
fechas.sort()

# 2. Datos de Productos (Categorías y Precios base)
productos_data = {
    'Abarrotes': ['Arroz', 'Azucar', 'Aceite', 'Fideos', 'Conservas'],
    'Verduras': ['Lechuga', 'Tomate', 'Zanahoria', 'Papa', 'Cebolla'],
    'Carnes': ['Pollo', 'Res', 'Cerdo', 'Pescado'],
    'Lacteos': ['Leche', 'Yogurt', 'Queso', 'Mantequilla']
}

precios_base = {
    'Arroz': 4.5, 'Azucar': 3.8, 'Aceite': 12.0, 'Fideos': 2.5, 'Conservas': 6.0,
    'Lechuga': 2.0, 'Tomate': 3.5, 'Zanahoria': 2.8, 'Papa': 3.0, 'Cebolla': 2.5,
    'Pollo': 18.0, 'Res': 28.0, 'Cerdo': 22.0, 'Pescado': 25.0,
    'Leche': 5.5, 'Yogurt': 7.0, 'Queso': 15.0, 'Mantequilla': 8.5
}

# 3. Generar Datos Transaccionales
data = []

for fecha in fechas:
    # Selección aleatoria de producto
    categoria = random.choice(list(productos_data.keys()))
    producto = random.choice(productos_data[categoria])
    
    # Cantidad y Precio (con pequeña variación aleatoria)
    cantidad = np.random.randint(1, 10)
    precio_unit = round(precios_base[producto] * np.random.uniform(0.9, 1.1), 2)
    total_venta = round(cantidad * precio_unit, 2)
    
    # --- DATOS PARA LOGÍSTICA (Envío) ---
    distancia_km = round(np.random.uniform(0.5, 15.0), 1)
    trafico = np.random.choice(['Bajo', 'Medio', 'Alto'], p=[0.4, 0.4, 0.2])
    
    # Lógica para definir si llega tarde (Target Binario)
    # Si hay mucho tráfico y distancia larga, prob de retraso es alta
    prob_retraso = 0.1
    if trafico == 'Alto': prob_retraso += 0.4
    if distancia_km > 10: prob_retraso += 0.3
    
    llega_tarde = 1 if np.random.random() < prob_retraso else 0 # 1=Sí, 0=No
    
    # --- DATOS PARA CLUSTERING (Cliente) ---
    # Simulamos 100 clientes recurrentes
    id_cliente = np.random.randint(1001, 1101) 
    edad_cliente = np.random.randint(18, 65)
    
    # Lógica de comportamiento del cliente (para que K-Means encuentre patrones)
    # Clientes mayores compran más abarrotes/carnes, jóvenes más snacks/lacteos (simulado)
    if edad_cliente > 50:
        gasto_promedio_hist = np.random.uniform(50, 150)
    else:
        gasto_promedio_hist = np.random.uniform(20, 80)

    data.append([
        fecha, 
        categoria, 
        producto, 
        cantidad, 
        precio_unit, 
        total_venta,
        distancia_km,
        trafico,
        llega_tarde, # Target Logística
        id_cliente,
        edad_cliente,
        gasto_promedio_hist # Feature Clustering
    ])

# 4. Crear DataFrame
df = pd.DataFrame(data, columns=[
    'Fecha', 'Categoria', 'Producto', 'Cantidad', 'Precio_Unitario', 'Total_Venta',
    'Distancia_KM', 'Nivel_Trafico', 'Llega_Tarde', 
    'ID_Cliente', 'Edad_Cliente', 'Gasto_Hist_Cliente'
])

# 5. Guardar CSV
df.to_csv('dataset_market_final.csv', index=False)
print("¡Archivo 'dataset_market_final.csv' generado exitosamente con datos para los 4 modelos!")
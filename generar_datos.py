import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

print("Iniciando la generación de datos... esto puede tardar un momento.")

# --- Configuración de Productos ---
# (Nombre, Categoría, Precio Base, Cantidad Base promedio)
products_list = [
    ("Manzana Fuji", "Frutas", 1.50, 50),
    ("Plátano de Seda", "Frutas", 1.20, 60),
    ("Naranja (kg)", "Frutas", 0.90, 40),
    ("Lechuga Americana", "Verduras", 0.80, 30),
    ("Tomate Italiano (kg)", "Verduras", 1.30, 70),
    ("Papa Amarilla (kg)", "Verduras", 1.00, 65),
    ("Zanahoria (kg)", "Verduras", 0.70, 45),
    ("Arroz (Bolsa 1kg)", "Abarrotes", 2.10, 80),
    ("Aceite Vegetal (1L)", "Abarrotes", 3.50, 90),
    ("Huevos (docena)", "Abarrotes", 2.50, 75),
    ("Pan de Molde", "Abarrotes", 3.00, 35),
    ("Pollo (kg)", "Carnes", 5.50, 55),
    ("Carne Molida (kg)", "Carnes", 6.20, 40)
]

# --- Configuración de Fechas ---
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 11, 12) # Hasta la fecha actual
n_rows = 500
date_range_days = (end_date - start_date).days

# --- Generación de Datos ---
data = []
for i in range(n_rows):
    
    # 1. Seleccionar Producto al azar
    prod_name, prod_cat, prod_price, base_qty = random.choice(products_list)
    
    # 2. Generar Fecha al azar
    random_days = random.randint(0, date_range_days)
    fecha = start_date + timedelta(days=random_days)
    weekday = fecha.weekday() # Lunes=0, Domingo=6
    
    # 3. Calcular Cantidad (con lógica profesional)
    cantidad = base_qty
    
    # Efecto Fin de Semana (Ventas suben 20-80%)
    if weekday >= 5: # Sábado o Domingo
        cantidad *= np.random.uniform(1.2, 1.8)
    
    # Efecto Promoción (20% de prob, ventas suben 50-120%)
    if random.random() < 0.2: 
        promocion = "Si"
        cantidad *= np.random.uniform(1.5, 2.2)
    else:
        promocion = "No"
        
    # Ruido aleatorio (variación diaria natural)
    cantidad *= np.random.normal(1.0, 0.15)
    
    # Asegurar que la cantidad sea un entero positivo
    cantidad_vendida = max(5, int(cantidad))
    
    # 4. Generar ID de producto (ej. F-001)
    id_prod = f"{prod_cat[0]}-{i:03d}"
    
    # 5. Guardar fila
    data.append([
        fecha, id_prod, prod_name, prod_cat, 
        cantidad_vendida, prod_price, promocion
    ])

# --- Creación del DataFrame ---
df = pd.DataFrame(data, columns=[
    'Fecha', 'ID_Producto', 'Nombre_Producto', 'Categoria', 
    'Cantidad_Vendida', 'Precio_Unitario', 'Promocion'
])

# Ordenar por fecha (importante para series temporales)
df = df.sort_values(by='Fecha')
# Formatear la fecha como texto (mejor para CSV)
df['Fecha'] = df['Fecha'].dt.strftime('%Y-%m-%d')

# --- Guardar Archivo ---
nuevo_nombre_archivo = 'Ventas_market_delivery_500.csv'
df.to_csv(nuevo_nombre_archivo, index=False)

print("¡Éxito!")
print(f"Se ha creado un nuevo archivo '{nuevo_nombre_archivo}' con {len(df)} filas.")
print("Por favor, renombra este archivo al nombre que usa tu script o actualiza tu script.")
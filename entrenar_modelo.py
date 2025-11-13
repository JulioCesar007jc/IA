import os
import pandas as pd
# ...el resto de tus imports...
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

# Ignorar advertencias futuras para mantener la salida limpia
warnings.filterwarnings('ignore', category=FutureWarning)

print("--- Iniciando el Proceso de Entrenamiento ---")

# --- Paso 2.1: Cargar y Preparar los Datos ---
print("\n[Paso 2.1] Cargando y preparando datos...")

# Obtener la ruta absoluta de la carpeta donde está este script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Unir esa ruta con el nombre del archivo CSV
csv_path = os.path.join(script_dir, 'ventas_market_delivery.csv')

print(f"Buscando el archivo en: {csv_path}") # Línea de depuración

try:
    # Leemos el archivo CSV usando la ruta absoluta
    df = pd.read_csv(csv_path)
    print("Datos cargados exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta: {csv_path}")
    print("Asegúrate de que 'ventas_market_delivery.csv' esté en la misma carpeta que 'entrenar_modelo.py'.")
    exit()

# --- Paso 2.2: Ingeniería de Características (Feature Engineering) ---
print("[Paso 2.2] Realizando Ingeniería de Características...")

# Convertir la columna 'Fecha' a un objeto datetime real
df['Fecha'] = pd.to_datetime(df['Fecha'])

# 1. Extraer características de la fecha
# El día de la semana (Lunes=0, Domingo=6)
df['dia_semana'] = df['Fecha'].dt.weekday 
# El mes
df['mes'] = df['Fecha'].dt.month
# El día del mes
df['dia_mes'] = df['Fecha'].dt.day
# Característica binaria: ¿Es fin de semana? (1 si es Sábado/Domingo, 0 si no)
df['es_fin_de_semana'] = (df['Fecha'].dt.weekday >= 5).astype(int)

# 2. Convertir 'Promocion' a números (Binario: 1 = Si, 0 = No)
df['Promocion'] = df['Promocion'].apply(lambda x: 1 if x == 'Si' else 0)

# 3. Convertir variables categóricas (Producto y Categoría)
# Usamos 'get_dummies' (One-Hot Encoding). Esto crea nuevas columnas
# para cada producto/categoría, p.ej., 'Nombre_Producto_Manzana Fuji'
# con un 1 o 0. Es crucial para el modelo.
df_procesado = pd.get_dummies(df, columns=['Nombre_Producto', 'Categoria'], drop_first=False)

# 4. Limpiar columnas que ya no necesitamos para el modelo
# 'Fecha' original, 'ID_Producto' y 'Precio_Unitario' no se usarán como características
df_procesado = df_procesado.drop(['Fecha', 'ID_Producto', 'Precio_Unitario'], axis=1)

print("Datos procesados y listos para el modelo.")

# --- Paso 2.3: Definir 'X' (Características) y 'y' (Objetivo) ---
print("[Paso 2.3] Definiendo Características (X) y Objetivo (y)...")

# 'y' es lo que queremos predecir: la Cantidad Vendida
y = df_procesado['Cantidad_Vendida']

# 'X' es todo lo demás que usaremos para predecir
X = df_procesado.drop('Cantidad_Vendida', axis=1)

# --- Paso 2.4: Dividir los Datos (Entrenamiento y Prueba) ---
print("[Paso 2.4] Dividiendo datos en Entrenamiento y Prueba (80/20)...")
# Separamos el 20% de los datos para evaluar el modelo.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Paso 2.5: Entrenar el Modelo (Random Forest) ---
print("[Paso 2.5] Entrenando el modelo (RandomForestRegressor)...")

# Usamos un "Random Forest" (Bosque Aleatorio)
modelo = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

# Aquí el modelo aprende de los datos de entrenamiento
modelo.fit(X_train, y_train)
print("¡Modelo entrenado exitosamente!")

# --- Paso 2.6: Evaluar el Modelo ---
print("\n[Paso 2.6] Evaluando el rendimiento del modelo...")
# Usamos el modelo para predecir sobre los datos de prueba
predicciones = modelo.predict(X_test)

# Calculamos métricas clave
mae = mean_absolute_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print("--- Resultados de la Evaluación ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f} unidades")
print(f"-> (En promedio, el pronóstico se desvía en {mae:.2f} unidades)")
print(f"Coeficiente de Determinación (R²): {r2:.2%}")
print("-----------------------------------")
print("(Nota: Con los datos de muestra, la evaluación puede ser limitada,")
print(" con más datos reales, estas métricas mejorarán)")


# --- Paso 2.7: Guardar el Modelo y las Columnas ---
print("\n[Paso 2.7] Guardando el modelo entrenado y las columnas...")

# 1. Guardar el modelo entrenado
joblib.dump(modelo, 'modelo_pronostico.pkl')

# 2. Guardar la lista de columnas (MUY IMPORTANTE)
# La aplicación web necesita saber exactamente qué columnas
# y en qué orden las espera el modelo.
columnas_del_modelo = X.columns.tolist()
joblib.dump(columnas_del_modelo, 'columnas_modelo.pkl')

print("\n--- ¡Proceso Completado! ---")
print("Se han generado dos archivos:")
print("1. modelo_pronostico.pkl (Tu modelo de IA listo para usarse)")
print("2. columnas_modelo.pkl (La estructura de datos que espera el modelo)")
print("\nAhora puedes crear el archivo 'app.py' (Punto 3).")
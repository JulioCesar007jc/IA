import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard de Pronóstico",
    page_icon="📈",
    layout="wide"  # Usamos un layout ancho para el dashboard
)

# --- Cargar el Modelo y las Columnas ---
script_dir = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(script_dir, 'modelo_pronostico.pkl')
columnas_path = os.path.join(script_dir, 'columnas_modelo.pkl')
csv_path = os.path.join(script_dir, 'Ventas_market_delivery.csv') # Ruta al CSV

try:
    modelo = joblib.load(modelo_path)
    columnas_modelo = joblib.load(columnas_path)
except Exception as e:
    st.error(f"Error fatal al cargar los archivos del modelo (.pkl): {e}")
    st.error("Asegúrate de haber ejecutado 'entrenar_modelo.py' primero.")
    st.stop()

# --- Cargar DATOS HISTÓRICOS (para gráficos y métricas) ---
# Se cargan una sola vez al inicio
@st.cache_data
def cargar_datos_historicos():
    try:
        df_hist = pd.read_csv(csv_path)
        df_hist['Fecha'] = pd.to_datetime(df_hist['Fecha'])
        return df_hist
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{os.path.basename(csv_path)}'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        return pd.DataFrame()

df_historico = cargar_datos_historicos()

# --- TÍTULO PRINCIPAL ---
st.title('📈 Dashboard de Pronóstico de Demanda')
st.header('Market Delivery')

# --- BARRA LATERAL (CONTROLES) ---
# IMPORTANTE: Cambia "logo.png" por el nombre real de tu imagen de logo
try:
    st.sidebar.image("logo.png", width=150)
except FileNotFoundError:
    st.sidebar.error("No se encontró el logo. Asegúrate de que 'logo.png' esté en la carpeta.")

st.sidebar.header("Datos de Entrada")

# --- Formularios de Entrada (AHORA EN LA BARRA LATERAL) ---
with st.sidebar.form(key="pronostico_form"):
    
    fecha_pronostico = st.date_input("Selecciona la fecha")
    
    # Listas de productos y categorías (idealmente desde el CSV)
    if not df_historico.empty:
        lista_productos = sorted(df_historico['Nombre_Producto'].unique())
        lista_categorias = sorted(df_historico['Categoria'].unique())
    else:
        # Fallback si el CSV no se pudo cargar
        lista_productos = [
            'Manzana Fuji', 'Lechuga Americana', 'Arroz (Bolsa 1kg)', 
            'Plátano de Seda', 'Tomate Italiano (kg)', 'Aceite Vegetal (1L)'
        ]
        lista_categorias = ['Frutas', 'Verduras', 'Abarrotes']

    producto_seleccionado = st.selectbox("Selecciona el producto", lista_productos)
    categoria_seleccionada = st.selectbox("Selecciona la categoría", lista_categorias)
    promocion = st.radio("¿Estará en promoción?", ("No", "Si"))

    submit_button = st.form_submit_button(label="Realizar Pronóstico y Análisis")


# --- ZONA DE RESULTADOS (Página Principal) ---

if not submit_button:
    st.info("Por favor, ingresa los datos en la barra lateral y presiona 'Realizar Pronóstico y Análisis'.")

# --- Lógica de Predicción y Análisis (cuando se presiona el botón) ---
if submit_button:
    
    # --- 1. LÓGICA DE PREDICCIÓN (Modelo) ---
    try:
        # 1.1. Preparar datos para el modelo
        dia_semana = fecha_pronostico.weekday()
        mes = fecha_pronostico.month
        dia_mes = fecha_pronostico.day
        es_fin_de_semana = 1 if dia_semana >= 5 else 0

        datos_entrada = pd.DataFrame(columns=columnas_modelo)
        datos_entrada.loc[0] = 0 

        datos_entrada['dia_semana'] = dia_semana
        datos_entrada['mes'] = mes
        datos_entrada['dia_mes'] = dia_mes
        datos_entrada['es_fin_de_semana'] = es_fin_de_semana
        datos_entrada['Promocion'] = 1 if promocion == "Si" else 0
        
        col_producto = f"Nombre_Producto_{producto_seleccionado}"
        if col_producto in datos_entrada.columns:
            datos_entrada[col_producto] = 1
        
        col_categoria = f"Categoria_{categoria_seleccionada}"
        if col_categoria in datos_entrada.columns:
            datos_entrada[col_categoria] = 1

        # 1.2. Realizar la predicción
        prediccion = modelo.predict(datos_entrada[columnas_modelo])
        unidades_pronosticadas = round(prediccion[0])
        if unidades_pronosticadas < 0:
            unidades_pronosticadas = 0

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la predicción: {e}")
        st.stop()


    # --- 2. LÓGICA DE ANÁLISIS (Datos Históricos) ---
    if not df_historico.empty:
        hist_producto = df_historico[df_historico['Nombre_Producto'] == producto_seleccionado]
        
        if not hist_producto.empty:
            venta_promedio = round(hist_producto['Cantidad_Vendida'].mean(), 1)
            venta_maxima = hist_producto['Cantidad_Vendida'].max()
            # Preparamos los datos para el gráfico (Fecha como índice)
            hist_producto_chart = hist_producto.set_index('Fecha')['Cantidad_Vendida']
        else:
            venta_promedio = "N/A"
            venta_maxima = "N/A"
            hist_producto_chart = pd.DataFrame() # Gráfico vacío
    else:
        st.warning("No se pudieron cargar los datos históricos para el análisis.")
        venta_promedio = "N/A"
        venta_maxima = "N/A"
        hist_producto_chart = pd.DataFrame()


    # --- 3. MOSTRAR RESULTADOS (En Pestañas) ---
    
    st.subheader(f"Resultados para: {producto_seleccionado}")
    
    tab1, tab2 = st.tabs(["📊 Pronóstico", "📈 Análisis Histórico"])

    # --- Pestaña 1: Pronóstico ---
    with tab1:
        st.header(f"Pronóstico para el {fecha_pronostico.strftime('%d/%m/%Y')}")
        
        # Métrica principal del pronóstico
        st.metric(
            label="Demanda Pronosticada",
            value=f"{unidades_pronosticadas} unidades"
        )
        
        st.info(f"""
        **Detalles de la Predicción:**
        * **Producto:** {producto_seleccionado}
        * **Fecha:** {fecha_pronostico.strftime('%d/%m/%Y')}
        * **En Promoción:** {promocion}
        """)

    # --- Pestaña 2: Análisis Histórico ---
    with tab2:
        st.header("Análisis de Ventas Históricas")
        
        if venta_promedio != "N/A":
            col_metrica1, col_metrica2 = st.columns(2)
            col_metrica1.metric("Venta Promedio Histórica", f"{venta_promedio} unidades")
            col_metrica2.metric("Venta Máxima Histórica", f"{venta_maxima} unidades")
            
            st.divider() # Una línea divisoria
            
            st.subheader("Tendencia de Ventas Históricas")
            st.line_chart(hist_producto_chart)
            st.caption("Gráfico de ventas históricas del producto seleccionado.")
        else:
            st.warning(f"No se encontraron datos históricos para '{producto_seleccionado}' en el CSV.")
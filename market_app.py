import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN PRO DE PÁGINA ---
st.set_page_config(
    page_title="Market Delivery AI", 
    layout="wide", 
    page_icon="🚚",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS (Para que se vea elegante) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGAR MODELOS ---
@st.cache_resource
def cargar_modelos():
    try:
        return joblib.load('modelos_finales.pkl')
    except:
        return None

pack = cargar_modelos()

# --- CARGAR DATOS ---
@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv('dataset_market_final.csv')
    except:
        return None

df = cargar_datos()

# --- BARRA LATERAL MEJORADA ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=80)
    st.title("Market AI 🚀")
    st.markdown("---")
    st.write("**Menú de Navegación**")
    opcion = st.radio("", 
        ["🏠 Inicio / Dashboard",
         "📈 Predicción Demanda", 
         "🚚 Riesgo de Envíos", 
         "👥 Segmentación Clientes",
         "🧬 Análisis Estructural"])
    
    st.markdown("---")
    st.caption("Developed by: Julio Aliaga")
    st.caption("© 2025 Market Delivery Corp")

# --- LÓGICA PRINCIPAL ---

if pack and df is not None:
    
    # === PÁGINA DE INICIO (NUEVA) ===
    if opcion == "🏠 Inicio / Dashboard":
        st.title("🚚 Sistema de Inteligencia Artificial")
        st.markdown("### Bienvenido al Panel de Control Estratégico")
        st.info("Este sistema utiliza 4 algoritmos de Machine Learning para optimizar la logística y ventas de Market Delivery.")
        
        # Métricas Generales (Simuladas para efecto visual)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ventas Totales", "S/. 124,500", "+15%")
        c2.metric("Clientes Activos", "1,240", "+8%")
        c3.metric("Precisión IA", "94.2%", "Modelos Activos")
        c4.metric("Pedidos a Tiempo", "89%", "-2%")
        
        st.markdown("---")
        st.image("https://images.unsplash.com/photo-1586880244406-556ebe35f282?q=80&w=2000&auto=format&fit=crop", caption="Logística Inteligente")

    # === VISTA 1: REGRESIÓN LINEAL ===
    elif opcion == "📈 Predicción Demanda":
        st.title("📈 Pronóstico de Ventas")
        st.markdown("**Objetivo:** Estimar la demanda futura para evitar stock insuficiente.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### ⚙️ Parámetros")
            precio = st.number_input("Precio del Producto (S/.)", 1.0, 100.0, 5.0)
            
            if st.button("Calcular Proyección"):
                modelo = pack['modelo_lineal']
                pred = modelo.predict([[precio]])[0]
                
                st.markdown("---")
                st.success(f"📦 Demanda Estimada: **{int(pred)} Unidades**")
                
                # Interpretación de Negocio
                ingreso = precio * int(pred)
                st.info(f"💰 Ingreso Proyectado: **S/. {ingreso:.2f}**")
        
        with col2:
            st.markdown("### 📊 Tendencia Histórica")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.regplot(x=df['Precio_Unitario'], y=df['Cantidad'], data=df, scatter_kws={'alpha':0.3, 'color':'blue'}, line_kws={'color':'red'}, ax=ax)
            plt.title("Elasticidad Precio-Demanda")
            st.pyplot(fig)

    # === VISTA 2: REGRESIÓN LOGÍSTICA ===
    elif opcion == "🚚 Riesgo de Envíos":
        st.title("🚚 Monitor de Riesgos Logísticos")
        st.markdown("**Objetivo:** Predecir retrasos para tomar acciones preventivas.")

        col1, col2 = st.columns(2)
        with col1:
            distancia = st.slider("Distancia de Ruta (Km)", 0.5, 20.0, 5.0)
            trafico = st.selectbox("Nivel de Tráfico", ["Bajo", "Medio", "Alto"])
            
            if st.button("Analizar Envío"):
                le = pack['le_trafico']
                modelo = pack['modelo_logistico']
                trafico_num = le.transform([trafico])[0]
                prob = modelo.predict_proba([[distancia, trafico_num]])[0][1]
                
                st.markdown("---")
                st.metric("Probabilidad de Retraso", f"{round(prob*100, 1)}%")
                
                # Semáforo de Riesgo (Interpretación)
                if prob > 0.6:
                    st.error("🚨 **ALERTA ROJA:** Retraso inminente. Se sugiere cambiar ruta.")
                elif prob > 0.3:
                    st.warning("⚠️ **ALERTA AMARILLA:** Riesgo moderado. Monitorear conductor.")
                else:
                    st.success("✅ **VERDE:** Envío seguro y a tiempo.")

        with col2:
            st.write("### 📉 Distribución de Incidencias")
            conteo = df['Llega_Tarde'].value_counts()
            fig, ax = plt.subplots(figsize=(5,3))
            ax.pie(conteo, labels=['A Tiempo', 'Retrasado'], autopct='%1.1f%%', colors=['#4CAF50','#FF5252'])
            st.pyplot(fig)

    # === VISTA 3: K-MEANS ===
    elif opcion == "👥 Segmentación Clientes":
        st.title("👥 Perfilamiento de Clientes (K-Means)")
        st.markdown("**Objetivo:** Agrupar clientes para campañas de marketing personalizadas.")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            edad = st.number_input("Edad del Cliente", 18, 90, 30)
            gasto = st.number_input("Gasto Mensual (S/.)", 0.0, 500.0, 50.0)
            
            if st.button("Identificar Segmento"):
                scaler = pack['scaler_kmeans']
                kmeans = pack['modelo_kmeans']
                datos = scaler.transform([[edad, gasto]])
                grupo = kmeans.predict(datos)[0]
                
                st.markdown("---")
                st.metric("Grupo Asignado", f"Cluster {grupo}")
                
                # Interpretación de Marketing (El toque PRO)
                if grupo == 0:
                    st.info("💡 **Estrategia:** Cliente Joven/Ahorrador. -> *Enviar Cupones 2x1*")
                elif grupo == 1:
                    st.info("💡 **Estrategia:** Cliente Promedio. -> *Ofrecer Puntos Bonus*")
                else:
                    st.success("💎 **Estrategia:** Cliente VIP. -> *Ofrecer Delivery Gratis*")

        with c2:
            st.write("### 🗺️ Mapa de Clientes")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='Edad_Cliente', y='Gasto_Hist_Cliente', hue='ID_Cliente', palette='viridis', legend=False, ax=ax)
            # Dibujar el punto nuevo
            if 'grupo' in locals():
                plt.scatter(edad, gasto, c='red', s=200, marker='*', label='Nuevo Cliente')
                plt.legend()
            plt.xlabel("Edad")
            plt.ylabel("Gasto Histórico")
            st.pyplot(fig)

    # === VISTA 4: JERÁRQUICO ===
    elif opcion == "🧬 Análisis Estructural":
        st.title("🧬 Dendrograma de Datos")
        st.markdown("**Objetivo:** Visualizar cómo se relacionan los datos de forma natural.")
        
        if st.button("Generar Árbol Jerárquico"):
            with st.spinner('Procesando estructura de datos...'):
                muestra = df[['Edad_Cliente', 'Gasto_Hist_Cliente']].sample(50, random_state=42)
                Z = linkage(muestra, 'ward')
                
                fig, ax = plt.subplots(figsize=(10, 5))
                dendrogram(Z, ax=ax)
                plt.title("Agrupación Jerárquica de Clientes")
                plt.ylabel("Distancia (Similitud)")
                st.pyplot(fig)
                st.success("✅ Estructura generada con éxito.")

else:
    st.error("⚠️ Error: Ejecuta 'entrenar_modelos_final.py' primero.")
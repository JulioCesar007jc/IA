import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURACIÓN PROFESIONAL DE PÁGINA ---
st.set_page_config(
    page_title="Market Delivery AI", 
    layout="wide", 
    page_icon="🚚",
    initial_sidebar_state="expanded"
)

# --- 2. ESTILOS CSS (MAQUILLAJE VISUAL) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CARGAR MODELOS Y DATOS ---
@st.cache_resource
def cargar_modelos():
    try:
        return joblib.load('modelos_finales.pkl')
    except:
        return None

pack = cargar_modelos()

@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv('dataset_market_final.csv')
    except:
        return None

df = cargar_datos()

# --- 4. MENÚ LATERAL MEJORADO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=80)
    st.title("Market AI 🚀")
    st.markdown("---")
    st.write("**Panel de Control**")
    opcion = st.radio("Selecciona una herramienta:", 
        ["🏠 Inicio / Dashboard",
         "📈 Predicción de Ventas", 
         "🚚 Riesgo de Logística", 
         "👥 Segmentación de Clientes",
         "🧬 Análisis Estructural"])
    
    st.markdown("---")
    st.caption("© 2025 Market Delivery Corp")
    st.caption("Desarrollado por: Julio Aliaga")

# --- 5. LÓGICA DE LA APLICACIÓN ---

if pack and df is not None:
    
    # === PÁGINA DE INICIO (DASHBOARD) ===
    if opcion == "🏠 Inicio / Dashboard":
        st.title("🚚 Centro de Comando - Inteligencia Artificial")
        st.markdown("### Bienvenido al sistema de optimización logística")
        st.info("Este software integra 4 modelos de Machine Learning para la toma de decisiones estratégicas.")
        
        # Métricas simuladas para que se vea como un sistema real en producción
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ventas del Mes", "S/. 45,200", "+5%")
        col2.metric("Clientes Nuevos", "124", "+12%")
        col3.metric("Precisión de IA", "94%", "Estable")
        col4.metric("Envíos a Tiempo", "98%", "+2%")
        
        st.markdown("---")
        st.image("https://images.unsplash.com/photo-1586880244406-556ebe35f282?q=80&w=2000&auto=format&fit=crop", caption="Logística Inteligente en Tiempo Real")

    # === VISTA 1: REGRESIÓN LINEAL (CON INTERPRETACIÓN) ===
    elif opcion == "📈 Predicción de Ventas":
        st.title("📈 Pronóstico Inteligente de Demanda")
        st.markdown("Estima cuánto venderás para optimizar tu inventario.")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### Parámetros")
            precio = st.number_input("Precio del Producto (S/.)", 1.0, 100.0, 5.0)
            
            if st.button("Calcular Proyección"):
                modelo = pack['modelo_lineal']
                pred = modelo.predict([[precio]])[0]
                
                st.markdown("---")
                st.metric("Demanda Estimada", f"{int(pred)} Unidades")
                
                # Interpretación de Negocio
                ingreso_proyectado = precio * int(pred)
                st.success(f"💰 **Impacto:** Se proyectan ingresos por **S/. {ingreso_proyectado:.2f}**")
        
        with c2:
            st.markdown("### Tendencia de Precios")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.regplot(x=df['Precio_Unitario'], y=df['Cantidad'], data=df, 
                       scatter_kws={'alpha':0.5, 'color':'#3b8ed0'}, line_kws={'color':'red'}, ax=ax)
            plt.title("Elasticidad Precio-Demanda")
            st.pyplot(fig)

    # === VISTA 2: REGRESIÓN LOGÍSTICA (CON SEMÁFORO) ===
    elif opcion == "🚚 Riesgo de Logística":
        st.title("🚚 Monitor de Riesgos de Envío")
        st.markdown("Sistema de alerta temprana para prevenir retrasos.")

        c1, c2 = st.columns(2)
        with c1:
            distancia = st.slider("Distancia de Ruta (Km)", 0.5, 20.0, 5.0)
            trafico = st.selectbox("Nivel de Tráfico", ["Bajo", "Medio", "Alto"])
            
            if st.button("Analizar Riesgo"):
                le = pack['le_trafico']
                modelo = pack['modelo_logistico']
                trafico_num = le.transform([trafico])[0]
                prob = modelo.predict_proba([[distancia, trafico_num]])[0][1]
                
                st.markdown("---")
                st.metric("Probabilidad de Retraso", f"{round(prob*100, 1)}%")
                
                # Semáforo de Riesgo (Lógica de Negocio)
                if prob > 0.6:
                    st.error("🚨 **ALERTA CRÍTICA:** Retraso inminente. Se sugiere cambiar de ruta o conductor.")
                elif prob > 0.3:
                    st.warning("⚠️ **ALERTA MEDIA:** Riesgo moderado. Monitorear envío.")
                else:
                    st.success("✅ **ENVÍO SEGURO:** Alta probabilidad de llegar a tiempo.")

        with c2:
            st.write("### Historial de Incidencias")
            conteo = df['Llega_Tarde'].value_counts()
            fig, ax = plt.subplots(figsize=(5,3))
            ax.pie(conteo, labels=['A Tiempo', 'Retrasado'], autopct='%1.1f%%', colors=['#4CAF50','#FF5252'])
            st.pyplot(fig)

    # === VISTA 3: K-MEANS (CON ESTRATEGIAS) ===
    elif opcion == "👥 Segmentación de Clientes":
        st.title("👥 Perfilamiento de Clientes")
        st.markdown("Identifica el tipo de cliente para aplicar marketing dirigido.")
        
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
                
                # Estrategias de Negocio Automáticas
                if grupo == 0:
                    st.info("💡 **Estrategia:** Cliente Joven/Ahorrador -> Enviar cupones de descuento 2x1.")
                elif grupo == 1:
                    st.info("💡 **Estrategia:** Cliente Estándar -> Fidelizar con acumulación de puntos.")
                else:
                    st.success("💎 **Estrategia:** Cliente VIP -> Ofrecer Delivery Gratis y atención preferencial.")

        with c2:
            st.write("### Mapa de Segmentos")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='Edad_Cliente', y='Gasto_Hist_Cliente', hue='ID_Cliente', palette='viridis', legend=False, ax=ax)
            
            # Dibujar el cliente actual como una estrella roja
            if 'grupo' in locals():
                plt.scatter(edad, gasto, c='red', s=200, marker='*', label='Nuevo Cliente')
                plt.legend()
                
            plt.xlabel("Edad")
            plt.ylabel("Gasto Histórico")
            st.pyplot(fig)

    # === VISTA 4: JERÁRQUICO ===
    elif opcion == "🧬 Análisis Estructural":
        st.title("🧬 Dendrograma de Datos")
        st.markdown("Visualización de las conexiones ocultas entre perfiles de clientes.")
        
        if st.button("Generar Árbol Jerárquico"):
            with st.spinner('Procesando estructura de datos...'):
                muestra = df[['Edad_Cliente', 'Gasto_Hist_Cliente']].sample(50, random_state=42)
                Z = linkage(muestra, 'ward')
                
                fig, ax = plt.subplots(figsize=(10, 5))
                dendrogram(Z, ax=ax)
                plt.title("Conexiones Jerárquicas")
                plt.ylabel("Distancia (Similitud)")
                st.pyplot(fig)
                st.success("✅ Gráfico generado correctamente.")

else:
    st.error("⚠️ Error: No se encontraron los modelos. Ejecuta el entrenamiento primero.")
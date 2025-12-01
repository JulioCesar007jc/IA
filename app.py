import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURACIÓN VISUAL (LAYOUT WIDE) ---
st.set_page_config(
    page_title="Market Delivery AI",
    layout="wide",
    page_icon="🚚",
    initial_sidebar_state="expanded"
)

# --- 2. ESTILOS CSS PERSONALIZADOS (MODO DARK/PRO) ---
st.markdown("""
    <style>
    /* Fondo principal y fuentes */
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1f2c56;
        font-family: 'Helvetica', sans-serif;
    }
    h3 {
        color: #FF4B4B;
    }
    /* Tarjetas de métricas */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    /* Botones personalizados */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 20px;
        height: 50px;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #d43535;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGAR MODELOS Y DATOS ---
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

# --- BARRA LATERAL ELEGANTE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=80)
    st.markdown("## **Market AI**")
    st.markdown("Sistema de Inteligencia Logística")
    st.write("---")
    
    opcion = st.radio("📍 **NAVEGACIÓN**", 
        ["🏠 Dashboard Ejecutivo",
         "📈 Predicción de Ventas", 
         "🚚 Monitor de Riesgos", 
         "👥 Segmentación Clientes",
         "🧬 Análisis Estructural"])
    
    st.write("---")
    st.info("💡 **Tip:** Interactúa con los gráficos haciendo zoom.")
    st.caption("© 2025 Julio Aliaga | v2.0 Pro")

# --- LÓGICA PRINCIPAL ---

if pack and df is not None:
    
    # === PÁGINA DE INICIO: DASHBOARD EJECUTIVO ===
    if opcion == "🏠 Dashboard Ejecutivo":
        st.title("📊 Tablero de Control Estratégico")
        st.markdown("Visión general del rendimiento operativo y predicciones de IA.")
        
        # Fila de métricas clave (KPIs)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Ingresos Proyectados", "S/. 45,200", "▲ 5.2%")
        kpi2.metric("Pedidos Procesados", "1,245", "▲ 12%")
        kpi3.metric("Tasa de Puntualidad", "94.8%", "▼ 0.5%")
        kpi4.metric("Precisión Modelos", "92%", "Estable")
        
        st.markdown("---")
        
        # Gráficos interactivos de resumen
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📈 Tendencia de Ventas (Histórico)")
            # Agrupar ventas por mes (simulado para el gráfico)
            df['Mes'] = pd.to_datetime(df['Fecha']).dt.month_name()
            ventas_mes = df.groupby('Mes')['Total_Venta'].sum().reset_index()
            fig_ventas = px.bar(ventas_mes, x='Mes', y='Total_Venta', color='Total_Venta', 
                                template='plotly_white', color_continuous_scale='Reds')
            st.plotly_chart(fig_ventas, use_container_width=True)
            
        with c2:
            st.subheader("🚚 Distribución de Tráfico")
            fig_pie = px.pie(df, names='Nivel_Trafico', title='Condiciones de Ruta', 
                             color_discrete_sequence=px.colors.sequential.RdBu, hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    # === VISTA 1: REGRESIÓN LINEAL (PLOTLY) ===
    elif opcion == "📈 Predicción de Ventas":
        st.title("📈 Pronóstico de Demanda (IA)")
        st.markdown("Modelo de **Regresión Lineal** para optimización de precios.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### ⚙️ Simulador")
            st.write("Ajusta el precio para ver la proyección.")
            precio = st.number_input("Precio Unitario (S/.)", 1.0, 100.0, 5.0)
            
            if st.button("Calcular Proyección"):
                modelo = pack['modelo_lineal']
                pred = modelo.predict([[precio]])[0]
                
                st.success(f"📦 Demanda: **{int(pred)} Unidades**")
                st.info(f"💰 Ingreso: **S/. {precio * int(pred):.2f}**")
        
        with col2:
            st.markdown("### 🔍 Análisis de Elasticidad")
            # Gráfico interactivo con línea de tendencia
            fig = px.scatter(df, x="Precio_Unitario", y="Cantidad", trendline="ols",
                             title="Relación Precio vs Cantidad (Interactivo)",
                             labels={"Precio_Unitario": "Precio (S/.)", "Cantidad": "Unidades Vendidas"},
                             template="plotly_white", opacity=0.6)
            fig.update_traces(marker=dict(size=8, color='#FF4B4B'))
            st.plotly_chart(fig, use_container_width=True)

    # === VISTA 2: REGRESIÓN LOGÍSTICA (GAUGE CHART) ===
    elif opcion == "🚚 Monitor de Riesgos":
        st.title("🚚 Predicción de Retrasos")
        st.markdown("Modelo de **Clasificación** para alertas logísticas.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📡 Datos del Envío")
            distancia = st.slider("Distancia (Km)", 0.5, 20.0, 5.0)
            trafico = st.select_slider("Nivel de Tráfico", options=["Bajo", "Medio", "Alto"])
            
            if st.button("Analizar Probabilidad"):
                le = pack['le_trafico']
                modelo = pack['modelo_logistico']
                trafico_num = le.transform([trafico])[0]
                prob = modelo.predict_proba([[distancia, trafico_num]])[0][1]
                
                # Gráfico de Velocímetro (Gauge)
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    title = {'text': "Probabilidad de Retraso"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prob > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "salmon"}],
                    }))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                if prob > 0.5:
                    st.error("🚨 **ALERTA:** Alta probabilidad de retraso.")
                else:
                    st.success("✅ **OK:** Envío seguro.")

        with c2:
            st.markdown("### 📊 Historial de Eficiencia")
            fig_hist = px.histogram(df, x="Distancia_KM", color="Llega_Tarde", 
                                    barmode="group", title="Retrasos por Distancia",
                                    color_discrete_map={0: "green", 1: "red"},
                                    labels={"Llega_Tarde": "Retraso (1=Sí)"})
            st.plotly_chart(fig_hist, use_container_width=True)

    # === VISTA 3: K-MEANS (SCATTER 3D O COLOR) ===
    elif opcion == "👥 Segmentación Clientes":
        st.title("👥 Clustering de Clientes")
        st.markdown("Segmentación automática basada en comportamiento.")
        
        tab1, tab2 = st.tabs(["🧩 Simulador de Perfil", "🗺️ Mapa de Clusters"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                edad = st.number_input("Edad", 18, 90, 30)
                gasto = st.number_input("Gasto (S/.)", 0.0, 500.0, 50.0)
                
                if st.button("Clasificar Cliente"):
                    scaler = pack['scaler_kmeans']
                    kmeans = pack['modelo_kmeans']
                    datos = scaler.transform([[edad, gasto]])
                    grupo = kmeans.predict(datos)[0]
                    
                    st.balloons() # Efecto visual divertido
                    st.metric("Segmento Asignado", f"Grupo {grupo}")
                    
                    if grupo == 0: st.info("🎯 **Estrategia:** Descuentos masivos.")
                    elif grupo == 1: st.warning("🎯 **Estrategia:** Fidelización.")
                    else: st.success("💎 **Estrategia:** Atención VIP.")

        with tab2:
            # Gráfico Interactivo de Clusters
            df['Cluster'] = pack['modelo_kmeans'].fit_predict(pack['scaler_kmeans'].transform(df[['Edad_Cliente', 'Gasto_Hist_Cliente']]))
            df['Cluster'] = df['Cluster'].astype(str) # Para que Plotly lo tome como categoría
            
            fig_cluster = px.scatter(df, x="Edad_Cliente", y="Gasto_Hist_Cliente", color="Cluster",
                                     title="Mapa Interactivo de Clientes",
                                     symbol="Cluster", size_max=10,
                                     template="plotly_white")
            st.plotly_chart(fig_cluster, use_container_width=True)

    # === VISTA 4: JERÁRQUICO (ESTÁTICO PERO BONITO) ===
    elif opcion == "🧬 Análisis Estructural":
        st.title("🧬 Dendrograma Jerárquico")
        st.markdown("Visualización de la estructura de datos.")
        
        with st.expander("ℹ️ ¿Cómo leer este gráfico?", expanded=True):
            st.write("Este gráfico muestra cómo se agrupan los clientes paso a paso. Las líneas verticales indican la distancia (diferencia) entre grupos.")
        
        if st.button("Generar Árbol"):
            with st.spinner('Procesando...'):
                muestra = df[['Edad_Cliente', 'Gasto_Hist_Cliente']].sample(50, random_state=42)
                Z = linkage(muestra, 'ward')
                
                fig, ax = plt.subplots(figsize=(12, 6))
                dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8)
                plt.title("Dendrograma de Clientes", fontsize=15)
                plt.xlabel("Clientes (Muestra)")
                plt.ylabel("Distancia Euclidiana")
                # Quitar bordes feos del gráfico matplotlib
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)

else:
    st.error("⚠️ Error: Ejecuta 'entrenar_modelos_final.py' primero.")
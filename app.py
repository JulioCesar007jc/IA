import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Market Delivery AI", layout="wide", page_icon="🚚")

# --- CARGAR MODELOS ---
@st.cache_resource
def cargar_modelos():
    try:
        return joblib.load('modelos_finales.pkl')
    except:
        st.error("⚠️ No se encontró 'modelos_finales.pkl'.")
        return None

pack = cargar_modelos()

# --- CARGAR DATOS (Para gráficos) ---
@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv('dataset_market_final.csv')
    except:
        return None

df = cargar_datos()

# --- MENÚ LATERAL ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=100)
st.sidebar.title("Sistema Inteligente")
opcion = st.sidebar.radio("Selecciona un Módulo:", 
    ["1. Predicción Demanda (Lineal)", 
     "2. Alerta Retrasos (Logística)", 
     "3. Segmentación Clientes (K-Means)",
     "4. Análisis Jerárquico (Cluster)"])

st.sidebar.markdown("---")
st.sidebar.info("Proyecto Final - Julio Aliaga")

# --- VISTAS PRINCIPALES ---

if pack and df is not None:
    
    # === VISTA 1: REGRESIÓN LINEAL ===
    if opcion == "1. Predicción Demanda (Lineal)":
        st.title("📈 Predicción de Demanda")
        st.markdown("Estima cuántas unidades venderás según el precio que fijes.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Panel de Control")
            precio = st.number_input("Precio del Producto (S/.)", 1.0, 100.0, 5.0)
            if st.button("Calcular Predicción"):
                modelo = pack['modelo_lineal']
                pred = modelo.predict([[precio]])[0]
                st.success(f"Se esperan vender: **{int(pred)} unidades**")
        
        with col2:
            st.write("### Datos Históricos")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['Precio_Unitario'], y=df['Cantidad'], ax=ax, alpha=0.5)
            plt.xlabel("Precio")
            plt.ylabel("Cantidad Vendida")
            st.pyplot(fig)

    # === VISTA 2: REGRESIÓN LOGÍSTICA ===
    elif opcion == "2. Alerta Retrasos (Logística)":
        st.title("🚚 Detector de Retrasos")
        st.markdown("Predice si un envío llegará tarde según el tráfico y distancia.")
        
        distancia = st.slider("Distancia (Km)", 0.5, 20.0, 5.0)
        trafico = st.selectbox("Nivel de Tráfico", ["Bajo", "Medio", "Alto"])
        
        if st.button("Analizar Riesgo"):
            le = pack['le_trafico']
            modelo = pack['modelo_logistico']
            
            # Codificar tráfico y predecir
            trafico_num = le.transform([trafico])[0]
            probabilidad = modelo.predict_proba([[distancia, trafico_num]])[0][1]
            
            st.metric("Probabilidad de Retraso", f"{round(probabilidad*100, 1)}%")
            
            if probabilidad > 0.5:
                st.error("🚨 ALERTA: Probable Retraso")
            else:
                st.success("✅ ENVÍO SEGURO: A tiempo")

    # === VISTA 3: K-MEANS ===
    elif opcion == "3. Segmentación Clientes (K-Means)":
        st.title("👥 Agrupamiento de Clientes")
        st.markdown("Clasifica nuevos clientes en grupos según su perfil.")
        
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("Edad del Cliente", 18, 90, 30)
            gasto = st.number_input("Gasto Histórico (S/.)", 0.0, 500.0, 50.0)
            
            if st.button("Clasificar"):
                scaler = pack['scaler_kmeans']
                kmeans = pack['modelo_kmeans']
                
                datos = scaler.transform([[edad, gasto]])
                grupo = kmeans.predict(datos)[0]
                st.info(f"Cliente pertenece al **GRUPO {grupo}**")

        with c2:
            st.write("### Mapa de Clientes")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='Edad_Cliente', y='Gasto_Hist_Cliente', hue='ID_Cliente', legend=False, palette='viridis', ax=ax)
            plt.xlabel("Edad")
            plt.ylabel("Gasto")
            st.pyplot(fig)

    # === VISTA 4: JERÁRQUICO ===
    elif opcion == "4. Análisis Jerárquico (Cluster)":
        st.title("🌳 Dendrograma de Productos")
        st.markdown("Visualización de cómo se agrupan los datos estructuralmente.")
        
        if st.button("Generar Gráfico Jerárquico"):
            # Usamos una muestra pequeña para que el gráfico se entienda bien
            muestra = df[['Edad_Cliente', 'Gasto_Hist_Cliente']].sample(40, random_state=42)
            Z = linkage(muestra, 'ward')
            
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(Z, ax=ax)
            plt.title("Agrupación Jerárquica")
            st.pyplot(fig)

else:
    st.warning("⚠️ Ejecuta 'entrenar_modelos_final.py' para generar los modelos primero.")
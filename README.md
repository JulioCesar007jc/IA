# 📈 Market Delivery - Sistema de Pronóstico de Demanda

Este es un proyecto de aplicación web que utiliza Machine Learning para pronosticar la demanda de ventas de productos (abarrotes, frutas y verduras) para el emprendimiento "Market Delivery".

La aplicación permite a un usuario ingresar un producto, una fecha y si está en promoción, y devuelve una estimación de la cantidad de unidades que se venderán.

## 🚀 Características

* **Modelo de Pronóstico:** Entrenado con un `RandomForestRegressor` de Scikit-Learn.
* **Interfaz Interactiva:** Construida con Streamlit, con controles en una barra lateral.
* **Re-entrenamiento:** El modelo puede ser fácilmente re-entrenado con nuevos datos.

---

## 🛠️ Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto en tu máquina local.

### 1. Prerrequisitos

* [Python 3.8+](https://www.python.org/downloads/)

### 2. Configuración del Proyecto

1.  Clona o descarga este repositorio en tu computadora.
2.  Abre una terminal en la carpeta del proyecto.

3.  Crea un entorno virtual:
    ```bash
    python -m venv venv
    ```

4.  Activa el entorno virtual:
    * **En Windows:**
        ```bash
        .\venv\Scripts\Activate.ps1
        ```
    * **En macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

5.  Instala todas las dependencias necesarias (¡usando el archivo que creamos!):
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♂️ Cómo Ejecutar el Proyecto

El proyecto tiene dos partes: entrenar el modelo y ejecutar la aplicación.

### 1. Entrenar el Modelo

Si agregas nuevos datos al archivo `Ventas_market_delivery.csv`, debes re-entrenar el modelo.

```bash
python entrenar_modelo.py
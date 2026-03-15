import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# =========================================================
# Cargar el pipeline entrenado
# =========================================================
modelo = load("Modelopipeline.joblib")

# =========================================================
# Diccionarios de opciones
# =========================================================
DOMINIO_OPCIONES = {
    "1 - Costa Norte": 1,
    "2 - Costa Centro": 2,
    "3 - Costa Sur": 3,
    "4 - Sierra Norte": 4,
    "5 - Sierra Centro": 5,
    "6 - Sierra Sur": 6,
    "7 - Selva": 7,
    "8 - Lima Metropolitana": 8
}

ESTRATO_OPCIONES = {
    "1 - De 500 000 a más habitantes": 1,
    "2 - De 100 000 a 499 999 habitantes": 2,
    "3 - De 50 000 a 99 999 habitantes": 3,
    "4 - De 20 000 a 49 999 habitantes": 4,
    "5 - De 2 000 a 19 999 habitantes": 5,
    "6 - De 500 a 1 999 habitantes": 6,
    "7 - Área de Empadronamiento Rural (AER) Compuesto": 7,
    "8 - Área de Empadronamiento Rural (AER) Simple": 8
}

# =========================================================
# Estado inicial
# =========================================================
if "totmieho" not in st.session_state:
    st.session_state["totmieho"] = 1

if "percepho" not in st.session_state:
    st.session_state["percepho"] = 1

if "dominio_label" not in st.session_state:
    st.session_state["dominio_label"] = "1 - Costa Norte"

if "estrato_label" not in st.session_state:
    st.session_state["estrato_label"] = "1 - De 500 000 a más habitantes"

# =========================================================
# Función para resetear entradas
# =========================================================
def reset_inputs():
    st.session_state["totmieho"] = 1
    st.session_state["percepho"] = 1
    st.session_state["dominio_label"] = "1 - Costa Norte"
    st.session_state["estrato_label"] = "1 - De 500 000 a más habitantes"

# =========================================================
# Título de la app
# =========================================================
st.title("Modelo de Regresión para la Predicción del Ingreso Monetario del Hogar")
st.markdown("##### Ingrese las características del hogar para estimar su ingreso monetario.")

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Campos a evaluar")

totmieho = st.sidebar.number_input(
    "TOTMIEHO - Total de miembros del hogar",
    min_value=1,
    max_value=20,
    step=1,
    key="totmieho"
)

percepho = st.sidebar.number_input(
    "PERCEPHO - Número de perceptores de ingreso",
    min_value=1,
    max_value=20,
    step=1,
    key="percepho"
)

dominio_label = st.sidebar.selectbox(
    "DOMINIO - Dominio geográfico",
    options=list(DOMINIO_OPCIONES.keys()),
    key="dominio_label"
)
dominio = DOMINIO_OPCIONES[dominio_label]

estrato_label = st.sidebar.selectbox(
    "ESTRATO - Estrato geográfico",
    options=list(ESTRATO_OPCIONES.keys()),
    key="estrato_label"
)
estrato = ESTRATO_OPCIONES[estrato_label]

# =========================================================
# Botón predecir
# =========================================================
if st.sidebar.button("Predecir"):
    if percepho > totmieho:
        st.warning("PERCEPHO no puede ser mayor que TOTMIEHO.")
    else:
        obs = pd.DataFrame({
            "TOTMIEHO": [totmieho],
            "PERCEPHO": [percepho],
            "DOMINIO": [dominio],
            "ESTRATO": [estrato]
        })

        # Forzar mismos tipos usados en entrenamiento
        obs["DOMINIO"] = obs["DOMINIO"].astype("category")
        obs["ESTRATO"] = obs["ESTRATO"].astype("category")

        st.write("Datos ingresados:")
        st.write(obs)

        # El modelo predice log_ingreso
        pred_log = modelo.predict(obs)

        # Volver a escala original
        pred_ingreso = np.exp(pred_log)[0]

        st.markdown(
            f"""
            <div style="
                padding: 16px;
                border-radius: 10px;
                background-color: #e8f5e9;
                border: 1px solid #81c784;
                margin-top: 10px;
            ">
                <p style="font-size: 18px; margin: 0; color: #1b5e20;">
                    Ingreso monetario estimado del hogar
                </p>
                <p style="font-size: 34px; font-weight: bold; margin: 0; color: #2e7d32;">
                    S/ {pred_ingreso:,.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("Ver detalle de la predicción"):
            st.write(f"Predicción en log_ingreso: {pred_log[0]:.4f}")
            st.write(f"Ingreso estimado en escala original: S/ {pred_ingreso:,.2f}")

# =========================================================
# Botón resetear
# =========================================================
if st.sidebar.button("Resetear"):
    reset_inputs()
    st.rerun()

# =========================================================
# Ejecución
# streamlit run streamlit_ingreso_hogar.py
# pip freeze > requirements.txt
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Configuración de página principal
st.set_page_config(
    page_title="Análisis de Interrupciones Aéreas",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo profesional en la interfaz
st.markdown("""
<style>
    .main-header {
        font-size:36px;
        font-weight:700;
        color: #4F46E5;
        text-align: center;
        margin-bottom: 25px;
    }
    .custom-tab-font {
        font-size: 18px;
        font-weight: bold;
    }
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">✈️ Análisis de Interrupciones Aéreas</div>', unsafe_allow_html=True)

# 1. Función para Cargar los Datos (Extracción y Transformación)
@st.cache_data
def transform_data(df_raw):
    df_clean = df_raw.dropna().copy()
    
    # Verificamos si ya está limpio (en caso de que suban cleaned_dataset.csv)
    if "impact_level" not in df_clean.columns:
        df_clean["cancelled_flights"] = df_clean["cancelled_flights"].astype(int)
        df_clean["passengers_impacted"] = df_clean["passengers_impacted"].astype(int)
        
        median_cancelled = df_clean["cancelled_flights"].median()
        df_clean["impact_level"] = np.where(
            df_clean["cancelled_flights"] > median_cancelled,
            "High Impact",
            "Low Impact"
        )
    return df_clean

try:
    df_raw = pd.read_csv("airline_losses_estimate.csv")
    df = transform_data(df_raw)
except FileNotFoundError:
    st.warning("⚠️ No se encontró el archivo 'airline_losses_estimate.csv' en el directorio local.")
    uploaded_file = st.file_uploader("Por favor sube tu dataset ('airline_losses_estimate.csv' o 'cleaned_dataset.csv') para continuar:", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        df = transform_data(df_raw)
    else:
        st.info("La aplicación se encuentra en pausa hasta que se suban los datos.")
        st.stop()

# 2. Panel Lateral de Navegación / Filtros
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/325/325258.png", width=120)
st.sidebar.title("Filtros de Búsqueda")
st.sidebar.markdown("Personaliza los datos usando estos controles:")

# Filtros
countries = ["Todos"] + sorted(df["country"].unique().tolist())
selected_country = st.sidebar.selectbox("Selecciona un País", countries)

airlines = ["Todas"] + sorted(df["airline"].unique().tolist())
selected_airline = st.sidebar.selectbox("Selecciona una Aerolínea", airlines)

# Aplicación de los filtros al dataframe
df_filtered = df.copy()

if selected_country != "Todos":
    df_filtered = df_filtered[df_filtered["country"] == selected_country]
    
if selected_airline != "Todas":
    df_filtered = df_filtered[df_filtered["airline"] == selected_airline]

# Recordatorio en el sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Desarrollado por:** Mariana Patiño Múnera.")

# 3. Fichas (Tabs) Profesionales
tab1, tab2, tab3, tab4 = st.tabs([
    "🗂️ Datos y Frecuencias", 
    "📈 Análisis Exploratorio", 
    "🌍 Mapa de Impacto", 
    "🤖 Análisis Predictivo"
])

# ------------- TAB 1: DATOS Y FRECUENCIAS -------------
with tab1:
    st.markdown("### 🗂️ Datos Limpios (Transformados)")
    st.dataframe(df_filtered.style.format(precision=2), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Frecuencia por País**")
        freq_country = df_filtered["country"].value_counts().reset_index()
        freq_country.columns = ["País", "Cantidad"]
        st.dataframe(freq_country, use_container_width=True)
        
    with col2:
        st.markdown("**Frecuencia por Nivel de Impacto**")
        freq_impact = df_filtered["impact_level"].value_counts().reset_index()
        freq_impact.columns = ["Nivel de Impacto", "Cantidad"]
        st.dataframe(freq_impact, use_container_width=True)

# ------------- TAB 2: ANÁLISIS EXPLORATORIO -------------
with tab2:
    st.markdown("### 📈 Visualizaciones Estadísticas")
    
    if len(df_filtered) > 0:
        # Histograma replicado del notebook
        fig_hist = px.histogram(
            df_filtered,
            x="cancelled_flights",
            color="impact_level",
            nbins=20,
            title="Distribución de vuelos cancelados",
            marginal="box",
            color_discrete_sequence=["#EF553B", "#636efa"] # colores Plotly
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Boxplot
        fig_box = px.box(
            df_filtered,
            x="country",
            y="cancelled_flights",
            color="country",
            title="Distribución de vuelos cancelados por país"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Bar Chart
        airline_counts = df_filtered["airline"].value_counts().reset_index()
        airline_counts.columns = ["airline", "count"]
        fig_bar = px.bar(
            airline_counts,
            x="airline",
            y="count",
            color="airline",
            title="Eventos de interrupción por aerolínea"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Los filtros actuales no devuelven datos para visualizar.")

# ------------- TAB 3: MAPA DE IMPACTO -------------
with tab3:
    st.markdown("### 🌍 Mapa Global del Impacto")
    if len(df_filtered) > 0:
        fig_map = px.scatter_geo(
            df_filtered,
            locations="country",
            locationmode="country names",
            color="estimated_daily_loss_usd",
            size="passengers_impacted",
            hover_name="airline",
            hover_data=["cancelled_flights", "rerouted_flights"],
            projection="natural earth",
            color_continuous_scale="Turbo",
            template="plotly_dark",
            title="Mapa global del impacto de interrupciones en aerolíneas"
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Los filtros actuales no devuelven datos para mapear.")

# ------------- TAB 4: ANÁLISIS PREDICTIVO -------------
with tab4:
    st.markdown("### 🤖 Predicción de Estimación Financiera en Tiempo Real")
    st.markdown("Hemos entrenado un modelo predictivo ligero basado en Regresión Lineal para estimar la pérdida diaria en dólares (USD) a partir de los atributos de un incidente aéreo hipotético.")
    
    # Preparación mínima del modelo
    X = df[["cancelled_flights", "rerouted_flights", "additional_fuel_cost_usd", "passengers_impacted"]]
    y = df["estimated_daily_loss_usd"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Paneles de Input
    st.markdown("#### Configura tu incidente hipotético:")
    colA, colB = st.columns(2)
    
    with colA:
        in_cancelled = st.number_input("Número de vuelos cancelados", min_value=0, max_value=500, value=10, step=1)
        in_rerouted = st.number_input("Número de vuelos desviados", min_value=0, max_value=500, value=30, step=1)
        
    with colB:
        in_fuel = st.number_input("Costo extra de tickets/combustible (USD)", min_value=0, max_value=5000000, value=1500000, step=10000)
        in_passengers = st.number_input("Número de pasajeros afectados", min_value=0, max_value=50000, value=5000, step=100)
        
    if st.button("Estimar Pérdida Financiera", type="primary"):
        input_data = pd.DataFrame({
            "cancelled_flights": [in_cancelled],
            "rerouted_flights": [in_rerouted],
            "additional_fuel_cost_usd": [in_fuel],
            "passengers_impacted": [in_passengers]
        })
        
        prediction = model.predict(input_data)[0]
        # Asegurarse de que no sea negativo (aunque podría serlo por la regresión lineal)
        prediction_val = max(0, prediction)
        
        st.success(f"La pérdida diaria proyectada es de aproximadamente: **${prediction_val:,.2f} USD**")
        
        # Un pequeño gauge / barra de medida
        st.progress(min(int(prediction_val / 5000000 * 100), 100))
        st.caption("Barra ilustrativa calculada respecto a incidentes críticos de \$5 Millones USD.")

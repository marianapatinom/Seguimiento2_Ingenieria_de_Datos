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
    
    # Conversiones limpias para el Dataset airline_losses.csv
    df_clean["cancellations_count"] = df_clean["cancellations_count"].astype(int)
    df_clean["reroutes_count"] = df_clean["reroutes_count"].astype(int)
    
    # Variable Derivada (Impact Level) basada en la cantidad de cancelaciones
    median_cancelled = df_clean["cancellations_count"].median()
    df_clean["impact_level"] = np.where(
        df_clean["cancellations_count"] > median_cancelled,
        "High Impact",
        "Low Impact"
    )
    return df_clean

# Cargar directamente 'airline_losses.csv' que está en la carpeta actual
try:
    df_raw = pd.read_csv("airline_losses.csv")
    df = transform_data(df_raw)
except FileNotFoundError:
    st.warning("⚠️ No se encontró el archivo 'airline_losses.csv' en el directorio local.")
    uploaded_file = st.file_uploader("Por favor sube tu dataset ('airline_losses.csv') para continuar:", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        
        # Validación
        required_columns = ["cancellations_count", "reroutes_count", "revenue_loss_pct", "estimated_loss_usd"]
        missing_cols = [col for col in required_columns if col not in df_raw.columns]
        
        if missing_cols:
            st.error(f"❌ El archivo subido no es válido. Faltan: {missing_cols}")
            st.stop()
        else:
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
        # Histograma 
        fig_hist = px.histogram(
            df_filtered,
            x="cancellations_count",
            color="impact_level",
            nbins=20,
            title="Distribución de vuelos cancelados",
            marginal="box",
            color_discrete_sequence=["#EF553B", "#636efa"] 
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Boxplot
        fig_box = px.box(
            df_filtered,
            x="country",
            y="cancellations_count",
            color="country",
            title="Distribución de vuelos cancelados por país"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Bar Chart (ahora muestra las cancelaciones reales, no el conteo de filas, ordenado mayor a menor)
        df_bar = df_filtered.sort_values(by="cancellations_count", ascending=False)
        fig_bar = px.bar(
            df_bar,
            x="airline",
            y="cancellations_count",
            color="airline",
            title="Vuelos cancelados totales por aerolínea",
            labels={"cancellations_count": "Cancelaciones"}
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
            color="estimated_loss_usd",
            size="cancellations_count",  # Se ajustó según las columnas existentes
            hover_name="airline",
            hover_data=["cancellations_count", "reroutes_count"],
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
    st.markdown("Basándonos en la estructura de tu archivo `airline_losses.csv`, entrenamos aquí un modelo ligero para predecir las pérdidas estimadas en USD.")
    
    # Preparación del modelo con las variables de este nuevo dataset
    X = df[["cancellations_count", "reroutes_count", "revenue_loss_pct"]]
    y = df["estimated_loss_usd"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Paneles de Input
    st.markdown("#### Configura tu incidente hipotético:")
    colA, colB = st.columns(2)
    
    with colA:
        in_cancelled = st.number_input("Número de vuelos cancelados", min_value=0, max_value=500, value=30, step=1)
        in_rerouted = st.number_input("Número de vuelos desviados", min_value=0, max_value=500, value=15, step=1)
        
    with colB:
        in_revloss = st.number_input("Pérdida de ingresos estimada (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        
    if st.button("Estimar Pérdida Financiera", type="primary"):
        input_data = pd.DataFrame({
            "cancellations_count": [in_cancelled],
            "reroutes_count": [in_rerouted],
            "revenue_loss_pct": [in_revloss]
        })
        
        prediction = model.predict(input_data)[0]
        prediction_val = max(0, prediction)
        
        st.success(f"La pérdida proyectada es de aproximadamente: **${prediction_val:,.2f} USD**")
        
        st.progress(min(int(prediction_val / 800000000 * 100), 100))
        st.caption("Barra gráfica respecto a pérdidas críticas de gran impacto (ej. \$800M USD).")

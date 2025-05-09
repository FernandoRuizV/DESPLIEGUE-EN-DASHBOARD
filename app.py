import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from sklearn.cluster import KMeans
from PIL import Image
import os
import numpy as np

# Configura la página
st.set_page_config(page_title="Amsterdam Airbnb", layout="wide", initial_sidebar_state="expanded")

# Cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv("Datos_limpios_Amsterdam.csv")
    # Limpieza de datos
    df = df[df['review_scores_cleanliness'] != 'Nothing']
    df = df[df['host_is_superhost'] != 'Without information']
    df = df[df['host_acceptance_rate'] != 'Without information']
    df['host_acceptance_rate'] = df['host_acceptance_rate'].str.rstrip('%').astype(float) / 100
    df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').astype(float) / 100
    df = df[df['price'] != 'Does not say']
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    
    # Separar tipos de columnas
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    numeric_cols = numeric_df.columns.tolist()
    text_df = df.select_dtypes(include=['object', 'bool'])
    text_cols = text_df.columns.tolist()
    
    # Configurar categorías
    category_col = 'host_is_superhost' if 'host_is_superhost' in df.columns else text_cols[0]
    unique_categories = df[category_col].dropna().unique()
    
    # Crear columna de rango de precios para análisis
    df['price_range'] = pd.cut(df['price'], 
                              bins=[0, 50, 100, 200, 500, df['price'].max()],
                              labels=['Muy barato', 'Barato', 'Medio', 'Caro', 'Muy caro'])
    
    return df, numeric_cols, text_cols, unique_categories, numeric_df

# Cargar datos
df, numeric_cols, text_cols, unique_categories, numeric_df = load_data()

# Aplicar estilos personalizados
def local_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #4A90E2;
            text-align: center;
            margin-bottom: 1rem;
            font-family: 'DM Serif Text', serif;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #388E3C;
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-family: 'DM Serif Text', serif;
        }
        .card {
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #4A90E2;
        }
        .metric-label {
            font-size: 1rem;
            color: #666;
        }
        .btn-custom {
            background-color: #4A90E2;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            text-align: center;
            margin: 0.5rem 0;
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Sidebar
st.sidebar.title("Panel de Control")
st.sidebar.markdown("Selecciona una visualización para explorar los datos de Airbnb")

# Control de rango de precios
price_range = (50.0, 500.0)
primary_color = '#4A90E2'

# Opciones de visualización
view = st.sidebar.radio("Selecciona una vista:",
                        ["Amsterdam",
                         "Análisis General",
                         "Dispersión de variables", 
                         "Distribución categórica", 
                         "Comparativa por categoría", 
                         "Análisis estadístico"])

st.markdown(f"<h1 class='main-header'>Airbnb Dashboard Amsterdam</h1>", unsafe_allow_html=True)

# Aplicar filtro de precios
filtered_df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]

if view == "Amsterdam":
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Text:ital@0;1&display=swap" rel="stylesheet">
    <h1 style="font-family: 'DM Serif Text', serif; color:#4A90E2; text-align: center;">
        Amsterdam
    </h1>
    """, unsafe_allow_html=True)

    image_folder = "imagenes_amsterdam"
    if os.path.exists(image_folder):
        images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.jpg', '.png')) and not img.startswith('.')]
        images.sort()  

        if images:
            if 'image_index' not in st.session_state:
                st.session_state.image_index = 0

            col1, col2, col3 = st.columns([1, 6, 1])

            with col1:
                if st.button("⏮️"):
                    st.session_state.image_index = (st.session_state.image_index - 1) % len(images)

            with col3:
                if st.button("⏭️"):
                    st.session_state.image_index = (st.session_state.image_index + 1) % len(images)

            current_image_path = os.path.join(image_folder, images[st.session_state.image_index])
            img = Image.open(current_image_path).resize((800, 500))  
            with col2:
                st.image(img, use_container_width=True)

        else:
            st.warning("No hay imágenes disponibles en la carpeta.")
    else:
        st.warning("La carpeta 'imagenes_amsterdam' no fue encontrada.")

    # Cuadrículas de información
    col_sup1, col_sup2 = st.columns(2)

    with col_sup1:
        st.markdown("""
        <div class='card'>
            <h3 style='color:#4A90E2; font-family: "DM Serif Text", serif;'>Historia</h3>
            <p>Fundada en el siglo XII como un pueblo pesquero.</p>
            <p>El nombre proviene de una presa en el río Amstel.</p>
            <p>Durante la Edad de Oro holandesa (siglo XVII), Amsterdam se convirtió en uno de los puertos más importantes del mundo.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_sup2:
        st.markdown("""
        <div class='card'>
            <h3 style='color:#4A90E2; font-family: "DM Serif Text", serif;'>Curiosidades</h3>
            <p>Ámsterdam tiene más bicicletas que habitantes.</p>
            <p>Cuenta con más de 100 km de canales navegables.</p>
            <p>El Aeropuerto de Schiphol está construido por debajo del nivel del mar.</p>
        </div>
        """, unsafe_allow_html=True)

    col_img, col_txt = st.columns([1, 2])

    with col_img:
        if os.path.exists(image_folder) and images:
            extra_img_path = os.path.join(image_folder, images[0])  # Usa la primera imagen como referencia
            st.image(extra_img_path, caption="Vista típica de Ámsterdam", use_container_width=True)

    with col_txt:
        st.markdown("""
        <div class='card' style='margin-left:40px;'>
            <h3 style='color:#388E3C; font-family: "DM Serif Text", serif;'>Airbnb y Cultura</h3>
            <ul>
                <li>Airbnb tiene gran presencia en los barrios céntricos.</li>
                <li>La ciudad alberga museos como el Rijksmuseum y Van Gogh.</li>
                <li>El queso Gouda y Edam son símbolos de la región.</li>
                <li>Amsterdam es conocida por sus cafés y una cultura liberal única.</li>
                <li>Los turistas suelen buscar alojamientos cercanos a atracciones como el Barrio Rojo y el Vondelpark.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif view == "Análisis General":
    st.markdown("<h2 class='sub-header'>Análisis General del Mercado</h2>", unsafe_allow_html=True)
    
    # KPIs en la parte superior
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Precio Promedio</div>
            <div class='metric-value'>${filtered_df['price'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_kpi2:
        st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Propiedades</div>
            <div class='metric-value'>{len(filtered_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_kpi3:
        superhost_percentage = (filtered_df['host_is_superhost'] == 'Yes').mean() * 100
        st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Superhosts</div>
            <div class='metric-value'>{superhost_percentage:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_kpi4:
        avg_score = filtered_df['review_scores_value'].replace('Nothing', np.nan).astype(float).mean()
        st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Puntuación Media</div>
            <div class='metric-value'>{avg_score:.1f}/10</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Selección de análisis
    analysis_options = ["Distribución de precios", "Puntuaciones", "Superhosts", "Localización", "Tipo de propiedad"]
    selected_analyses = st.multiselect("Selecciona análisis a mostrar:", analysis_options, 
                                      default=["Distribución de precios", "Localización"])
    
    # Distribución de columnas según selección
    if len(selected_analyses) > 0:
        cols = st.columns(min(len(selected_analyses), 2))
        col_index = 0
        
        if "Distribución de precios" in selected_analyses:
            with cols[col_index % 2]:
                st.subheader("Distribución de Precios")
                fig = px.histogram(filtered_df, x="price", nbins=30, 
                                   color_discrete_sequence=[primary_color],
                                   opacity=0.7)
                fig.update_layout(xaxis_title="Precio", yaxis_title="Frecuencia")
                st.plotly_chart(fig, use_container_width=True)
            col_index += 1
                
        if "Puntuaciones" in selected_analyses:
            with cols[col_index % 2]:
                st.subheader("Análisis de Puntuaciones")
                
                score_cols = [col for col in filtered_df.columns if 'review_scores' in col]
                score_df = filtered_df[score_cols].replace('Nothing', np.nan).astype(float)
                
                # Radar chart con puntuaciones promedio
                categories = [col.replace('review_scores_', '').capitalize() for col in score_cols]
                values = score_df.mean().tolist()
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Promedio',
                    line_color=primary_color
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )
                    ),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            col_index += 1
            
        if "Superhosts" in selected_analyses:
            with cols[col_index % 2]:
                st.subheader("Superhosts vs No Superhosts")
                
                # Comparativa de precios entre superhosts y no superhosts
                superhost_data = filtered_df.groupby('host_is_superhost')['price'].mean().reset_index()
                
                fig = px.bar(superhost_data, x='host_is_superhost', y='price',
                            labels={'host_is_superhost': 'Es Superhost', 'price': 'Precio Promedio'},
                            color='host_is_superhost',
                            color_discrete_sequence=[primary_color, '#FF6B6B'])
                
                fig.update_layout(xaxis_title="Superhost", yaxis_title="Precio Promedio ($)")
                st.plotly_chart(fig, use_container_width=True)
            col_index += 1
                
        if "Localización" in selected_analyses:
            with cols[col_index % 2]:
                st.subheader("Distribución por Barrios")
                
                if 'neighbourhood' in filtered_df.columns:
                    top_neighborhoods = filtered_df['neighbourhood'].value_counts().nlargest(10)
                    
                    fig = px.pie(values=top_neighborhoods.values, 
                                names=top_neighborhoods.index,
                                title="Top 10 barrios",
                                hole=0.4,
                                color_discrete_sequence=px.colors.sequential.Blues_r)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No se encontró la columna de barrios en los datos.")
            col_index += 1
            
        if "Tipo de propiedad" in selected_analyses:
            with cols[col_index % 2]:
                st.subheader("Análisis por Tipo de Propiedad")
                
                if 'property_type' in filtered_df.columns:
                    property_price = filtered_df.groupby('property_type')['price'].mean().sort_values(ascending=False).head(8)
                    
                    fig = px.bar(x=property_price.index, y=property_price.values,
                                labels={'x': 'Tipo de Propiedad', 'y': 'Precio Promedio'},
                                color=property_price.values,
                                color_continuous_scale='Blues')
                    
                    fig.update_layout(xaxis_title="Tipo de Propiedad", 
                                     yaxis_title="Precio Promedio ($)",
                                     xaxis={'categoryorder':'total descending'})
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No se encontró la columna de tipo de propiedad en los datos.")
            col_index += 1

elif view == "Dispersión de variables":
    st.subheader("Relación entre variables")
    x_var = st.selectbox("Eje X", numeric_cols)
    y_var = st.selectbox("Eje Y", numeric_cols)
    color_cat = st.selectbox("Color por", text_cols)
    
    # Opciones adicionales
    show_trendline = st.checkbox("Mostrar línea de tendencia", value=True)
    
    fig = px.scatter(filtered_df, x=x_var, y=y_var, 
                    color=filtered_df[color_cat], 
                    trendline="ols" if show_trendline else None,
                    hover_data=['price'])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas de correlación
    if show_trendline:
        corr = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
        st.metric("Correlación", f"{corr:.3f}")

elif view == "Distribución categórica":
    st.subheader("Distribución por categoría")
    
    # Opciones de visualización
    chart_type = st.radio("Tipo de gráfico", ["Gráfico circular", "Gráfico de barras"], horizontal=True)
    
    cat_col = st.selectbox("Variable Categórica", text_cols)
    num_col = st.selectbox("Variable Numérica", numeric_cols)
    
    # Preparar datos
    grouped = filtered_df.groupby(cat_col)[num_col].agg(['mean', 'count']).reset_index()
    grouped.columns = [cat_col, f'Promedio de {num_col}', 'Cantidad']
    
    # Visualizar según el tipo seleccionado
    if chart_type == "Gráfico circular":
        fig = px.pie(grouped, names=cat_col, values=f'Promedio de {num_col}', 
                    title=f"Distribución de {cat_col} por {num_col}",
                    hover_data=['Cantidad'])
    else:
        fig = px.bar(grouped, x=cat_col, y=f'Promedio de {num_col}', color=cat_col,
                    title=f"Promedio de {num_col} por {cat_col}",
                    text='Cantidad')
        fig.update_layout(xaxis={'categoryorder':'total descending'})
    
    st.plotly_chart(fig, use_container_width=True)

elif view == "Comparativa por categoría":
    st.subheader("Comparativa entre categorías")
    
    # Selección de variables
    cat_col = st.selectbox("Variable Categórica", text_cols)
    num_col = st.selectbox("Variable Numérica", numeric_cols)
    
    # Opciones de agrupación
    agg_function = st.radio("Función de agregación:", 
                           ["mean", "median", "sum", "count"],
                           format_func=lambda x: {
                               "mean": "Promedio",
                               "median": "Mediana",
                               "sum": "Suma",
                               "count": "Conteo"
                           }.get(x),
                           horizontal=True)
    
    # Agrupar datos según la función seleccionada
    grouped = filtered_df.groupby(cat_col)[num_col].agg(agg_function).reset_index()
    grouped.columns = [cat_col, f"{agg_function} de {num_col}"]
    
    # Crear visualización
    fig = px.bar(grouped, x=cat_col, y=f"{agg_function} de {num_col}", color=cat_col,
               title=f"{agg_function.capitalize()} de {num_col} por {cat_col}")
    
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Añadir vista comparativa como tabla
    st.markdown("### Tabla comparativa")
    st.dataframe(grouped)

elif view == "Análisis estadístico":
    st.subheader("Modelos Predictivos")
    
    # Opciones de modelo
    model_type = st.radio("Selecciona el tipo de modelo:", 
                         ["Regresión Lineal", "Regresión Logística"],
                         horizontal=True)
    
    # Selección de variables
    target_col = st.selectbox("Variable Objetivo (y)", numeric_cols + text_cols)
    feature_cols = st.multiselect("Variables Predictoras (X)", numeric_cols, 
                                default=[col for col in numeric_cols if col != target_col][:3])
    
    # Verificar que hay suficientes variables
    if len(feature_cols) > 0:
        
        if model_type == "Regresión Logística":
            df_model = filtered_df.copy()
            
            # Preprocesamiento para clasificación
            if df_model[target_col].dtype == object:
                label_encoder = LabelEncoder()
                df_model[target_col] = label_encoder.fit_transform(df_model[target_col])
                class_names = label_encoder.classes_
            
            # Preparar conjuntos de datos
            X = df_model[feature_cols].dropna()
            y = df_model.loc[X.index, target_col]
            
            # División train/test con tamaño fijo
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenamiento del modelo
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # Evaluación
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"Precisión del modelo: {acc:.2f}")
                
                # Matriz de confusión simplificada
                cm = confusion_matrix(y_test, y_pred)
                
                # Graficar de forma simple
                fig = plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.xlabel("Predicción")
                plt.ylabel("Valor real")
                plt.title("Matriz de Confusión")
                st.pyplot(fig)
            
            with col2:
                # Coeficientes
                coef_df = pd.DataFrame({
                    'Variable': feature_cols,
                    'Coeficiente': model.coef_[0]
                }).sort_values('Coeficiente', ascending=False)
                
                fig = px.bar(coef_df, x='Variable', y='Coeficiente', 
                           color='Coeficiente', color_continuous_scale='RdBu',
                           title="Importancia de variables")
                st.plotly_chart(fig, use_container_width=True)

        elif model_type == "Regresión Lineal":
            # Preparar datos
            X = filtered_df[feature_cols].dropna()
            y = filtered_df.loc[X.index, target_col]
            
            # Estandarizar características
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
            
            # Entrenamiento del modelo con tamaño fijo
            model = LinearRegression()
            model.fit(X_scaled_df, y)
            y_pred = model.predict(X_scaled_df)
            r2 = r2_score(y, y_pred)
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"Coeficiente de determinación R²: {r2:.2f}")
                
                # Gráfico de predicciones vs valores reales (solo puntos, sin línea)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y, y_pred, alpha=0.5)
                ax.set_xlabel("Valores Reales")
                ax.set_ylabel("Predicciones")
                ax.set_title("Predicciones vs Valores Reales")
                st.pyplot(fig)
            
            with col2:
                # Coeficientes
                coef_df = pd.DataFrame({
                    'Variable': feature_cols,
                    'Coeficiente': model.coef_
                }).sort_values('Coeficiente', ascending=False)
                
                fig = px.bar(coef_df, x='Variable', y='Coeficiente', 
                           color='Coeficiente', color_continuous_scale='RdBu',
                           title="Importancia de variables")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Por favor, selecciona al menos una variable predictora.")
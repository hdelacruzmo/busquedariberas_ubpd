import Definitions
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from src.back.ModelController import ModelController
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
from PIL import Image
from io import BytesIO


st.set_page_config(layout="wide", page_title="Búsqueda Riberas UBPD", page_icon="Sitios Búsquedaﾠ")


##Logo UBPD
logo = Image.open("resources/img/ubpd_color_logo.png")
st.image(logo, width=500, use_container_width=False)

st.title(":violet[Delimitación de áreas de interés para la búsqueda usando Análisis Espacial y Aprendizaje Computacional - UBPD​]")

##Imagen satelital rio cauca
rio_cauca = Image.open("resources/img/rio_cauca.png")
st.image(rio_cauca, width=1500, use_container_width=False)
# Fuente
st.caption("Imagen satelital del Río Cauca - Caucasia (Antioquia). Lat : 7.9351, Long:-75.1816")


# Subtítulo contextual
st.subheader(":gray[Proyecto de Búsqueda en Riberas — _Subdirección de Análisis, Planeación y Localización y Subdirección de Gestión de Información para la búsqueda_]")

st.subheader(":violet[Introducción:]", divider=True)
# Texto descriptivo
st.markdown("""
Este aplicativo permite realizar un análisis de probabilidad espacial orientado a la identificación de áreas con alta probabilidad de interés para la búsqueda de personas dadas por desaparecidas, con enfoque en riberas del río Cauca y rio Nechí.

Utilizando modelos de **Aprendizaje de Máquina supervisado**, se genera una superficie de probabilidad con base en atributos del terreno, factores antrópicos y elementos geográficos relacionados con eventos del conflicto en el área el Plan Regional de Búsqueda de Bajo Cauca y Valdivia.

El sistema compara el comportamiento de tres enfoques diferentes:

- Modelo 1: Regresión Logística (MaxEnt)
- Modelo 2: Ensamble de Regresiones Lineales
- Modelo 3: Random Forest (Árboles de decisión)

A través de visualizaciones interactivas, tablas de resumen y exportación de resultados, esta herramienta facilita la toma de decisiones basada en evidencia geoespacial.
""")

# Nota aclaratoria
st.caption("Los resultados de este aplicativo son de carácter exploratorio y Aunque se implementan técnicas de aprendizaje automático —herramientas computacionales diseñadas para reconocer patrones y clasificar datos a partir de ejemplos—, se ha optado por no usar el término “predicción” de forma directa. Esto se debe a que esta propuesta no busca ni pretende reemplazar la profunda labor investigativa, testimonial y humanitaria que realizan los buscadores y buscadoras, sino que propone una aproximación técnica complementaria, desde el campo de los análisis geoespaciales, para aportar datos que acoten esa labor titánica.")

# Construccion Dataset

with st.expander(":violet[Clic acá para ver información sobre la generación previa del dataset de entrada]"):
    st.markdown("""
    El dataset de entrada utilizado por los modelos fue construido mediante una serie de procesos de geoprocesamiento en `QGIS`. A partir de una cuadrícula base de puntos generada cada 500 metros sobre el área de estudio, se calcularon las siguientes variables espaciales:

    - **Distancias** a eventos de minas, acciones orientadas a civiles o a combatientes, y a vías.
        ¿a qué distancia está cada evento?
        
    - **Densidades** en radios definidos en torno a eventos de minas, acciones orientadas a civiles o a combatientes
        ¿Cuántos eventos han sucedido en un radio de 250 m? 
    
    - **Características topográficas** como pendiente y orientación (aspecto), derivadas de modelos de elevación digital  
    - **Tipo de cobertura del suelo**, **tipo de relieve morfométrico** y **tipo de vía**, extraídos por intersección con cartografía temática
    - **Solicitudes de restitución de predios** a partir de información dela URT.

    Finalmente, estas variables fueron compiladas en una única capa geográfica de puntos mediante una unión espacial, que luego fue exportada como un archivo `.gpkg` para su análisis en la herramienta.
    """)


# ---------------carga de datos
st.subheader(":violet[Carga de datos:]", divider=True)
# -------------------------------
#  VISUALIZACIÓN DEL GPKG
# -------------------------------

with st.expander(":violet[Clic acá para subir el archivo de entrada]"):
    uploaded_gpkg = st.file_uploader(
        "Sube tu archivo GPKG", accept_multiple_files=False, type=["gpkg"], key="gpkg"
    )

    if uploaded_gpkg is not None:
        try:
            gdf = gpd.read_file(uploaded_gpkg)

            if gdf.crs and gdf.crs.to_epsg() != 4326:
                st.info(f" Reproyectando desde {gdf.crs} a EPSG:4326 para visualización.")
                gdf = gdf.to_crs(epsg=4326)

            st.write("Vista previa de los datos de tu archivo:")
            st.dataframe(gdf, height=500, use_container_width=True)
            
            st.write("Vista geográfica del área de estudio:")
            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
            bounds = gdf.total_bounds
            minx, miny, maxx, maxy = bounds

            rectangle = Polygon([
                (minx, miny), (minx, maxy),
                (maxx, maxy), (maxx, miny),
                (minx, miny)
            ])
            gdf_rect = gpd.GeoDataFrame(geometry=[rectangle], crs=gdf.crs)

            mapa = folium.Map(location=[(miny + maxy)/2, (minx + maxx)/2], zoom_start=8, tiles=None)

            folium.TileLayer("OpenStreetMap", name="Mapa Base").add_to(mapa)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Satélite (Esri)",
                overlay=False,
                control=True
            ).add_to(mapa)
            folium.TileLayer(
                tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Nombres del Territorio",
                overlay=True,
                control=True
            ).add_to(mapa)

            folium.GeoJson(gdf_rect, name="Área de Estudio", tooltip="Área cubierta").add_to(mapa)
            mapa.fit_bounds([[miny, minx], [maxy, maxx]])
            folium.LayerControl(collapsed=False).add_to(mapa)

            #st_folium(mapa, width=900, height=500)
            st_folium(mapa, width='100%', height=600)

        except Exception as e:
            st.error(f"❌ Error leyendo el archivo: {e}")


# -------------------------------
#  PREDICCIÓN DESDE GPKG - TRES MODELOS
# -------------------------------
st.markdown("---")
st.subheader(":violet[Estimación de probabilidad de sitio de interés para la búsqueda:]", divider=True)

ctrl = ModelController()

if uploaded_gpkg is not None:
    try:
        gdf_input = gpd.read_file(uploaded_gpkg)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Modelo 1 : Regresión Logística (MaxEnt)",
            "Modelo 2 : Ensamble de Regresiones",
            "Modelo 3 : Random Forest",
            "Análisis : Comparación por umbral"
        ])

        modelos = [
            ("Modelo 1 : Regresión Logística (MaxEnt)", ctrl.predict_from_gdf(gdf_input)),
            ("Modelo 2 : Ensamble de Regresiones", ctrl.predict_with_second_model(gdf_input)),
            ("Modelo 3 : Random Forest", ctrl.predict_with_third_model(gdf_input))
        ]

        for nombre_modelo, gdf_resultado in modelos:
            tab = tab1 if nombre_modelo == "Modelo 1 : Regresión Logística (MaxEnt)" else tab2 if nombre_modelo == "Modelo 2 : Ensamble de Regresiones" else tab3

            with tab:
                st.markdown(f"Resultados del modelo {nombre_modelo}")

                ### acá

                col1, col2 = st.columns([1, 1])
    
                with col1:
                    st.markdown("#### :violet[Mapa estático del modelo]")
                    x_coords = gdf_resultado.geometry.x
                    y_coords = gdf_resultado.geometry.y
                    probs = gdf_resultado["probabilidad"]
                
                    fig, ax = plt.subplots(figsize=(1.5, 1.5))
                    scatter = ax.scatter(
                        x_coords, y_coords, c=probs,
                        cmap="viridis", s=0.2, edgecolor="none",
                        vmin=0, vmax=1
                    )
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, pad=0.01)
                    cbar.set_label("Probabilidad", fontsize=6)
                    cbar.ax.tick_params(labelsize=4)
                    ax.set_title("Distribución espacial de probabilidad", fontsize=8)
                    ax.axis("off")

                    # Guardar imagen en un buffer de memoria
                    buffer = BytesIO()
                    fig.savefig(buffer, format='png', dpi=1500, bbox_inches='tight')
                    buffer.seek(0)
                    
                    # Mostrar imagen reducida con Streamlit
                    st.image(buffer, caption="Mapa estático del modelo", use_container_width =True)
                    
                    # Botón para descargar la imagen
                    st.download_button(
                        label="📥 Descargar imagen del mapa",
                        data=buffer,
                        file_name=f"mapa_{nombre_modelo.lower().replace(' ', '_')}.png",
                        mime="image/png"
                    )

                
                
                with col2:
                    # Espaciador vertical
                    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
                    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

                
                    st.markdown("#### :violet[Estadísticas Generales:]")
                    st.markdown(f"- **Número total de puntos**: {len(gdf_resultado)}")
                    st.markdown(f"- **Probabilidad promedio**: {gdf_resultado['probabilidad'].mean():.3f}")
                    st.markdown(f"- **Máxima**: {gdf_resultado['probabilidad'].max():.3f} | **Mínima**: {gdf_resultado['probabilidad'].min():.3f}")
                    st.markdown(f"- **Puntos con probabilidad ≥ 0.8**: {(gdf_resultado['probabilidad'] >= 0.8).sum()}")


                tabla_mostrar = (
                    gdf_resultado
                    .drop(columns=["geometry", "FID_Mina"], errors="ignore")
                    .rename(columns={
                        "Num_PrediosURT": "PrediosURT",
                        "Minas1000m": "Dens_Minas",
                        "Dist_EventoCombatiente": "Dist_Comb",
                        "Dens_EventoCombatiente": "Dens_Comb"
                    })
                )
                #color en la columna probabilidad
                st.dataframe(
                    tabla_mostrar,
                    height=400,
                    use_container_width=True,
                    column_config={
                        "probabilidad": st.column_config.NumberColumn(
                            "Probabilidad",
                            help="Probabilidad de presencia (0 = baja, 1 = alta)",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f"
                        )
                    }
                )
                #st.dataframe(tabla_mostrar, height=400, use_container_width=True)
                
                st.subheader(":violet[Estadísticas por rangos de probabilidad]")
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
                labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"]
                gdf_resultado["rango_probabilidad"] = pd.cut(
                    gdf_resultado["probabilidad"], bins=bins, labels=labels, include_lowest=True
                )
                conteo = gdf_resultado["rango_probabilidad"].value_counts().sort_index()

                conteo_df = conteo.reset_index()
                conteo_df.columns = ["Rango", "Cantidad"]

                # Definir colores fijos en orden de los labels
                viridis_5 = {
                    "0–0.2": "#440154",
                    "0.2–0.4": "#3b528b",
                    "0.4–0.6": "#21918c",
                    "0.6–0.8": "#5ec962",
                    "0.8–1": "#fde725"
                }
                
                fig = px.bar(
                    conteo_df,
                    x="Rango",
                    y="Cantidad",
                    color="Rango",
                    title="Cantidad de puntos por rango de probabilidad",
                    text="Cantidad",
                    color_discrete_map=viridis_5
                )
                fig.update_layout(xaxis_title="Rango de probabilidad", yaxis_title="Número de puntos")
                st.plotly_chart(fig, use_container_width=True)

                ### Exploración interactiva por variable
                st.markdown("### :violet[Exploración interactiva por variable]")

                # Variables predictoras (excepto probabilidad y tipo_punto)
                variables_numericas = [
                    col for col in tabla_mostrar.columns
                    if col not in ["probabilidad", "tipo_punto"] and pd.api.types.is_numeric_dtype(tabla_mostrar[col])
                ]
                
                var_x = st.selectbox("Selecciona una variable para el eje X", variables_numericas, key=f"x_{nombre_modelo}")
                
                fig = px.scatter(
                    tabla_mostrar,
                    x=var_x,
                    y="probabilidad",
                    color="probabilidad",
                    color_continuous_scale="Viridis",
                    title=f"Probabilidad vs {var_x}",
                    hover_data=tabla_mostrar.columns
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)  


                
                ### Descargar archivo con resultados
                st.markdown("### :violet[Descargar archivo con resultados]")
                output_path = f"/tmp/resultados_{nombre_modelo.lower().replace(' ', '_')}.gpkg"
                gdf_resultado.to_file(output_path, driver="GPKG")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label=f"Descargar GPKG - {nombre_modelo}",
                        data=f,
                        file_name=f"resultados_{nombre_modelo.lower().replace(' ', '_')}.gpkg",
                        mime="application/octet-stream"
                    )

        # 🟦 Cuarta pestaña — COMPARACIÓN POR UMBRAL
        with tab4:
            st.subheader("Coincidencias por umbral en los tres modelos")

            umbral = st.number_input("Selecciona el umbral mínimo", min_value=0.0, max_value=1.0, step=0.01, value=0.8)

            comparado = modelos[0][1][["probabilidad"]].rename(columns={"probabilidad": "prob_modelo_1"}).copy()
            comparado["prob_modelo_2"] = modelos[1][1]["probabilidad"].values
            comparado["prob_modelo_3"] = modelos[2][1]["probabilidad"].values
            comparado["geometry"] = modelos[0][1].geometry.values

            seleccionados = comparado[
                (comparado["prob_modelo_1"] >= umbral) &
                (comparado["prob_modelo_2"] >= umbral) &
                (comparado["prob_modelo_3"] >= umbral)
            ]

            st.markdown(f"🔎 Se encontraron **{len(seleccionados)} puntos** donde los tres modelos tienen probabilidad ≥ {umbral:.2f}")
            st.data_editor(seleccionados.drop(columns="geometry"), height=500, use_container_width=True, disabled=True)

            salida_path = "/tmp/seleccionados_tres_modelos.gpkg"
            seleccionados_gdf = gpd.GeoDataFrame(seleccionados, geometry="geometry", crs=modelos[0][1].crs)
            seleccionados_gdf.to_file(salida_path, driver="GPKG")
            with open(salida_path, "rb") as f:
                st.download_button(
                    label="📥 Descargar selección como GPKG",
                    data=f,
                    file_name="seleccionados_tres_modelos.gpkg",
                    mime="application/octet-stream"
                )

    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")

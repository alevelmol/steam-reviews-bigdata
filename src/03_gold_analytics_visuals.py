# %% [markdown]
# # Capa Gold: Analítica Avanzada y Visualización
# Este script procesa los datos masivos usando PySpark, aplicando transformaciones, 
# joins y agregaciones para reducir el volumen de datos (Capa Gold).
# Luego, convierte los resultados agregados a Pandas (`.toPandas()`) para
# renderizar gráficas visualmente atractivas sin saturar la memoria local.

# %%
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast, count, avg, when, round

# Workaround para Java 18+ (en tu caso Java 25)
os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-java-options "-Djava.security.manager=allow" pyspark-shell'

# %%
def create_spark_session() -> SparkSession:
    """
    Crea y configura la sesión de Spark optimizada para ejecución local.
    """
    return (
        SparkSession.builder
        .appName("SteamReviews_Gold_Visuals")
        .master("local[*]") 
        .config("spark.driver.memory", "4g") 
        .config("spark.sql.shuffle.partitions", "16") 
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )

spark = create_spark_session()

# Rutas a los datos procesados en formato Parquet
PROCESSED_DIR = "data/processed"

# Spark lee del tirón todos los archivos Parquet de una carpeta.
reviews_files = glob.glob(os.path.join(PROCESSED_DIR, "reviews_part*.parquet"))
reviews_files.sort()  # Ordenar para garantizar consistencia

df_reviews = spark.read.parquet(*reviews_files)  # Desempacar la lista de archivos
df_apps = spark.read.parquet(os.path.join(PROCESSED_DIR, "applications.parquet"))
df_app_genres = spark.read.parquet(os.path.join(PROCESSED_DIR, "application_genres.parquet"))
df_genres = spark.read.parquet(os.path.join(PROCESSED_DIR, "genres.parquet"))

print("Datos cargados correctamente en PySpark.")

# %% [markdown]
# ## 1. La Paradoja del Hater (Scatter Plot)
# **Objetivo:** ¿La gente que vota negativo juega más horas que la que vota positivo?
# **Estrategia Spark:** Agruparemos por `appid` para calcular el promedio de horas jugadas en reseñas positivas y negativas por cada juego.
# Luego, compararemos estos dos valores en un Scatter Plot.

# %%
# Convertimos playtime a horas para facilitar la lectura
df_reviews_hours = df_reviews.withColumn("playtime_hours", col("author_playtime_forever") / 60.0)

# Agrupamos por juego y calculamos el promedio de horas según el voto
df_hater_paradox = (
    df_reviews_hours
    .groupBy("appid")
    .agg(
        avg(when(col("voted_up") == True, col("playtime_hours"))).alias("avg_hours_positive"),
        avg(when(col("voted_up") == False, col("playtime_hours"))).alias("avg_hours_negative"),
        count("recommendationid").alias("total_reviews")
    )
    .filter(col("total_reviews") > 50) # Filtramos juegos con pocas reseñas para evitar ruido
    .dropna()
)

# Convertimos a Pandas para graficar
# EXPLICACIÓN: Usar .toPandas() aquí es correcto porque ya hemos reducido millones de reseñas 
# a unas pocas miles de filas (una por juego). Si hiciéramos .toPandas() sobre el df original, 
# la RAM colapsaría.
pdf_hater = df_hater_paradox.toPandas()

# Gráfica: Scatter Plot con Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=pdf_hater, 
    x="avg_hours_positive", 
    y="avg_hours_negative", 
    size="total_reviews", 
    sizes=(20, 400),
    alpha=0.6,
    color="#e74c3c"
)

# Línea de referencia 1:1 (x=y)
max_val = min(pdf_hater["avg_hours_positive"].max(), pdf_hater["avg_hours_negative"].max())
plt.plot([0, 500], [0, 500], 'k--', zorder=0, label="Mismas horas (1:1)")

plt.xlim(0, 500) # Limitamos a 500 horas para ver el grueso de los datos
plt.ylim(0, 500)
plt.title("La Paradoja del Hater: Horas Jugadas (Positivo vs Negativo por Juego)", fontsize=14, pad=15)
plt.xlabel("Media de Horas Jugadas (Voto Positivo)")
plt.ylabel("Media de Horas Jugadas (Voto Negativo)")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 2. Top Géneros y la Torre de Babel (Bar Chart Interactivo)
# **Objetivo:** Conteo masivo de reseñas por género, aceptando la alta cardinalidad.

# %%
# Hacemos un Broadcast Join con las tablas de géneros, ya que son pequeñas
df_genre_bridge = df_app_genres.join(broadcast(df_genres), df_app_genres.genre_id == df_genres.id, "inner")

# Cruzamos las reseñas con la tabla puente de géneros
df_reviews_genres = df_reviews.select("recommendationid", "appid").join(
    broadcast(df_genre_bridge), 
    on="appid", 
    how="inner"
)

# Agrupamos masivamente por nombre del género
df_top_genres = (
    df_reviews_genres
    .groupBy(col("name").alias("genre_name"))
    .agg(count("recommendationid").alias("review_count"))
    .orderBy(col("review_count").desc())
    .limit(20) # Nos quedamos con el Top 20
)

# Descargamos a Pandas para la gráfica interactiva
pdf_top_genres = df_top_genres.toPandas()

# Gráfica: Bar Chart Interactivo con Plotly
fig = px.bar(
    pdf_top_genres.sort_values("review_count", ascending=True), # Invertimos para que el mayor quede arriba
    x="review_count", 
    y="genre_name", 
    orientation='h',
    title="Top 20 Géneros con más Reseñas (La Torre de Babel)",
    labels={"review_count": "Número de Reseñas", "genre_name": "Género"},
    color="review_count",
    color_continuous_scale=px.colors.sequential.Plasma
)
fig.update_layout(template="plotly_dark", height=600)
fig.show()


# %% [markdown]
# ## 3. Toxicidad vs Precio (Boxplot)
# **Objetivo:** Distribución de la toxicidad (% de votos negativos) según el rango de precio.

# %%
# Clasificamos el precio usando when/otherwise de PySpark
# is_free suele ser boolean o string, mat_initial_price numérico. Nos aseguramos casteando.
df_apps_priced = df_apps.select(
    "appid",
    when(col("is_free") == True, "1. Free to Play")
    .when(col("mat_initial_price") < 15, "2. Indie Barato (<15€)")
    .when(col("mat_initial_price") >= 50, "4. AAA (>=50€)")
    .otherwise("3. Mid-range (15-50€)")
    .alias("price_tier")
)

# Calculamos el % de reseñas negativas (Toxicidad) por juego
df_toxicity = (
    df_reviews
    .groupBy("appid")
    .agg(
        count("recommendationid").alias("total_reviews"),
        count(when(col("voted_up") == False, True)).alias("negative_reviews")
    )
    .filter(col("total_reviews") > 30) # Para que el porcentaje sea representativo
    .withColumn("toxicity_percent", round((col("negative_reviews") / col("total_reviews")) * 100, 2))
)

# Unimos Toxicidad con la categoría de precio usando Broadcast (df_apps_priced es pequeño)
df_toxic_vs_price = df_toxicity.join(broadcast(df_apps_priced), on="appid", how="inner")

# Reducimos a Pandas
pdf_toxic_price = df_toxic_vs_price.select("price_tier", "toxicity_percent").toPandas()
pdf_toxic_price = pdf_toxic_price.sort_values("price_tier")

# Gráfica: Boxplot con Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=pdf_toxic_price, 
    x="price_tier", 
    y="toxicity_percent", 
    palette="viridis"
)
plt.title("Toxicidad (% Reseñas Negativas) vs Precio del Juego", fontsize=14, pad=15)
plt.xlabel("Categoría de Precio")
plt.ylabel("Porcentaje de Reseñas Negativas (%)")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. El efecto "Early Access" (Donut Chart)
# **Objetivo:** Porcentaje de reseñas hechas durante Early Access vs Lanzamiento Completo.

# %%
# Agrupamos por la columna written_during_early_access
df_early_access = (
    df_reviews
    .groupBy("written_during_early_access")
    .agg(count("recommendationid").alias("total_reviews"))
    .dropna()
)

# Traemos el resumen a Pandas (son literalmente 2 filas, ultra ligero)
pdf_early_access = df_early_access.toPandas()

# Mapeamos los booleanos a textos amigables
pdf_early_access['Status'] = pdf_early_access['written_during_early_access'].map({
    True: 'Durante Early Access', 
    False: 'Post Lanzamiento'
})

# Gráfica: Donut Chart con Matplotlib
plt.figure(figsize=(8, 8))
colors = ['#2ecc71', '#3498db']
explode = (0.05, 0) # Destacamos una porción

plt.pie(
    pdf_early_access['total_reviews'], 
    labels=pdf_early_access['Status'], 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=colors, 
    explode=explode,
    pctdistance=0.85,
    textprops={'fontsize': 12, 'color': 'white' if sns.get_style() == 'dark' else 'black'}
)

# Dibujamos un círculo blanco en el centro para crear el efecto "Donut"
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Distribución de Reseñas: Early Access vs Post Lanzamiento', fontsize=15)
plt.tight_layout()
plt.show()

# %%
# Apagamos la sesión de Spark al terminar
spark.stop()
print("Procesamiento analítico y visualización finalizados.")

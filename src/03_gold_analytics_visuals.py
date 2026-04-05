# %% [markdown]
# # Capa Gold: Analítica Avanzada y Visualización
# Este script procesa los datos masivos usando PySpark, aplicando transformaciones,
# joins y agregaciones para reducir el volumen de datos (Capa Gold).
# Luego, convierte los resultados agregados a Pandas (`.toPandas()`) para
# renderizar gráficas visualmente atractivas sin saturar la memoria local.

# %%
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, broadcast, count, avg, when, round, log1p,
    from_unixtime, to_date, sum as spark_sum
)

# Workaround para Java 18+ (en tu caso Java 25)
os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-java-options "-Djava.security.manager=allow" pyspark-shell'

# Directorio de salida para las gráficas
PLOTS_DIR = "data/output/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

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
plt.savefig(os.path.join(PLOTS_DIR, "01_hater_paradox.png"), dpi=150)
plt.show()


# %% [markdown]
# ## 2. Top Géneros y la Torre de Babel (Bar Chart)
# **Objetivo:** Conteo masivo de reseñas por género, aceptando la alta cardinalidad.

# %%
# Hacemos un Broadcast Join con las tablas de géneros, ya que son pequeñas
df_genre_bridge = df_app_genres.join(broadcast(df_genres), df_app_genres.genre_id == df_genres.id, "inner")

# Cruzamos las reseñas con la tabla puente de géneros
# Incluimos voted_up para poder calcular sentimiento en gráficas posteriores
df_reviews_genres = df_reviews.select("recommendationid", "appid", "voted_up").join(
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

# Descargamos a Pandas para la gráfica
pdf_top_genres = df_top_genres.toPandas()
pdf_top_genres = pdf_top_genres.sort_values("review_count", ascending=True)

# Gráfica: Bar Chart horizontal con Seaborn
fig, ax = plt.subplots(figsize=(10, 8))
palette = sns.color_palette("plasma", n_colors=len(pdf_top_genres))
sns.barplot(
    data=pdf_top_genres,
    x="review_count",
    y="genre_name",
    hue="genre_name",
    palette=palette,
    legend=False,
    ax=ax
)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1_000_000)}M" if x >= 1_000_000 else f"{int(x/1_000)}K"))
ax.set_title("Top 20 Géneros con más Reseñas (La Torre de Babel)", fontsize=14, pad=15)
ax.set_xlabel("Número de Reseñas")
ax.set_ylabel("Género")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_top_genres_babel.png"), dpi=150)
plt.show()


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
    hue="price_tier",
    palette="viridis",
    legend=False
)
plt.title("Toxicidad (% Reseñas Negativas) vs Precio del Juego", fontsize=14, pad=15)
plt.xlabel("Categoría de Precio")
plt.ylabel("Porcentaje de Reseñas Negativas (%)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_toxicity_vs_price.png"), dpi=150)
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
    textprops={'fontsize': 12, 'color': 'black'}
)

# Dibujamos un círculo blanco en el centro para crear el efecto "Donut"
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Distribución de Reseñas: Early Access vs Post Lanzamiento', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_early_access_donut.png"), dpi=150)
plt.show()


# %% [markdown]
# ## 5. Ratio de Positividad por Género (Stacked Bar Chart)
# **Objetivo:** ¿Qué géneros generan comunidades más positivas o más críticas?
# **Estrategia Spark:** Reutilizamos el join reseña-género ya realizado y añadimos
# una dimensión de sentimiento con `when/otherwise`. La agregación es la única shuffle necesaria.

# %%
# Contamos reseñas positivas y negativas por género, filtrando géneros con poco volumen
df_sentiment_genre = (
    df_reviews_genres
    .groupBy(col("name").alias("genre_name"))
    .agg(
        count(when(col("voted_up") == True, True)).alias("positive"),
        count(when(col("voted_up") == False, True)).alias("negative"),
        count("recommendationid").alias("total")
    )
    .filter(col("total") > 10_000)  # Solo géneros con masa crítica suficiente
    .withColumn("pct_positive", round((col("positive") / col("total")) * 100, 1))
    .withColumn("pct_negative", round((col("negative") / col("total")) * 100, 1))
    .orderBy(col("pct_positive").desc())
    .limit(15)
)

pdf_sentiment = df_sentiment_genre.select("genre_name", "pct_positive", "pct_negative").toPandas()

# Construimos el stacked bar manualmente con dos series
fig, ax = plt.subplots(figsize=(10, 7))
genres = pdf_sentiment["genre_name"]
x = range(len(genres))

bars_pos = ax.barh(x, pdf_sentiment["pct_positive"], color="#2ecc71", label="Positivas (%)")
bars_neg = ax.barh(x, pdf_sentiment["pct_negative"], left=pdf_sentiment["pct_positive"], color="#e74c3c", label="Negativas (%)")

ax.set_yticks(list(x))
ax.set_yticklabels(genres)
ax.axvline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="50%")
ax.set_xlabel("Porcentaje de Reseñas (%)")
ax.set_title("Ratio de Positividad por Género (Top 15 por % positivo)", fontsize=14, pad=15)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_sentiment_by_genre.png"), dpi=150)
plt.show()


# %% [markdown]
# ## 6. Violin Plot: Distribución de Horas Jugadas por Sentimiento (escala log)
# **Objetivo:** Visualizar la distribución completa (no solo la media) de horas jugadas
# para votantes positivos vs negativos. La escala logarítmica es necesaria porque
# la distribución de playtime en Steam tiene una cola derecha extrema (jugadores con miles de horas).
# **Decisión técnica:** Aplicamos `log1p` en Spark antes de bajar a Pandas para evitar
# que outliers extremos colapsen la visualización.

# %%
# Transformación logarítmica en Spark: log1p(x) = ln(1+x), evita log(0) para playtime=0
df_playtime_log = (
    df_reviews_hours
    .select(
        log1p(col("playtime_hours")).alias("log_playtime"),
        when(col("voted_up") == True, "Positivo").otherwise("Negativo").alias("sentimiento")
    )
    .filter(col("playtime_hours") > 0)
    .sample(False, 0.05)  # Muestra del 5%: suficiente para la distribución, no colapsa Pandas
)

pdf_playtime = df_playtime_log.toPandas()

plt.figure(figsize=(9, 6))
sns.violinplot(
    data=pdf_playtime,
    x="sentimiento",
    y="log_playtime",
    palette={"Positivo": "#2ecc71", "Negativo": "#e74c3c"},
    inner="quartile",
    cut=0
)
# Etiquetas del eje Y en horas reales (transformación inversa de log1p)
tick_vals = [0, np.log1p(1), np.log1p(10), np.log1p(100), np.log1p(500), np.log1p(2000)]
tick_labels = ["0h", "1h", "10h", "100h", "500h", "2000h"]
plt.yticks(tick_vals, tick_labels)
plt.title("Distribución de Horas Jugadas por Sentimiento (escala log)", fontsize=14, pad=15)
plt.xlabel("Tipo de Voto")
plt.ylabel("Horas Jugadas (escala logarítmica)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_playtime_violin.png"), dpi=150)
plt.show()


# %% [markdown]
# ## 7. Heatmap de Toxicidad: Precio × Early Access
# **Objetivo:** ¿Cambia la actitud de la comunidad (toxicidad) según si el juego
# es Early Access Y según su rango de precio? Este cruce de dos dimensiones
# categóricas es ideal para un heatmap de pivote.
# **Estrategia Spark:** Reutilizamos `df_toxic_vs_price` (ya calculado) y le añadimos
# la dimensión de Early Access con otro join broadcast, minimizando shuffles.

# %%
# Calculamos la media de early_access por juego (True si la mayoría de reseñas son EA)
df_ea_flag = (
    df_reviews
    .groupBy("appid")
    .agg(
        avg(when(col("written_during_early_access") == True, 1).otherwise(0)).alias("ea_ratio")
    )
    .withColumn(
        "acceso_anticipado",
        when(col("ea_ratio") > 0.5, "Early Access").otherwise("Lanzamiento Completo")
    )
    .select("appid", "acceso_anticipado")
)

# Cruzamos con toxicidad y precio (ambos ya calculados, solo un join más)
df_heatmap = (
    df_toxic_vs_price
    .join(broadcast(df_ea_flag), on="appid", how="inner")
    .groupBy("price_tier", "acceso_anticipado")
    .agg(avg("toxicity_percent").alias("avg_toxicity"))
)

pdf_heatmap = df_heatmap.toPandas()
pivot = pdf_heatmap.pivot(index="price_tier", columns="acceso_anticipado", values="avg_toxicity")

plt.figure(figsize=(8, 5))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "% Reseñas Negativas (media)"}
)
plt.title("Heatmap de Toxicidad: Rango de Precio × Tipo de Acceso", fontsize=14, pad=15)
plt.xlabel("Tipo de Acceso")
plt.ylabel("Categoría de Precio")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_toxicity_heatmap.png"), dpi=150)
plt.show()


# %% [markdown]
# ## 8. Top 20 Juegos Mejor y Peor Valorados
# **Objetivo:** Identificar los juegos con mayor y menor porcentaje de reseñas positivas,
# filtrando juegos con pocas reseñas para que el ranking sea estadísticamente robusto.
# **Estrategia Spark:** Un único groupBy sobre df_reviews reduce todo a una fila por juego;
# el join posterior con df_apps (tabla pequeña → broadcast) añade el nombre sin shuffle extra.
# NOTA: Se asume que df_apps contiene una columna `name` con el nombre del juego.

# %%
df_game_ratings = (
    df_reviews
    .groupBy("appid")
    .agg(
        count("recommendationid").alias("total_reviews"),
        count(when(col("voted_up") == True, True)).alias("positive_reviews")
    )
    .filter(col("total_reviews") >= 200)  # Umbral para que el % sea representativo
    .withColumn("pct_positive", round((col("positive_reviews") / col("total_reviews")) * 100, 1))
    .join(broadcast(df_apps.select("appid", "name")), on="appid", how="inner")
)

pdf_best = (
    df_game_ratings.orderBy(col("pct_positive").desc()).limit(20)
    .select("name", "pct_positive", "total_reviews").toPandas()
    .sort_values("pct_positive")
)
pdf_worst = (
    df_game_ratings.orderBy(col("pct_positive").asc()).limit(20)
    .select("name", "pct_positive", "total_reviews").toPandas()
    .sort_values("pct_positive", ascending=False)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

sns.barplot(
    data=pdf_best, x="pct_positive", y="name",
    hue="name", palette="Greens_r", legend=False, ax=ax1
)
ax1.set_xlim(0, 100)
ax1.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax1.set_title("Top 20 Juegos Mejor Valorados", fontsize=13, pad=12)
ax1.set_xlabel("% Reseñas Positivas")
ax1.set_ylabel("")

sns.barplot(
    data=pdf_worst, x="pct_positive", y="name",
    hue="name", palette="Reds_r", legend=False, ax=ax2
)
ax2.set_xlim(0, 100)
ax2.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.set_title("Top 20 Juegos Peor Valorados", fontsize=13, pad=12)
ax2.set_xlabel("% Reseñas Positivas")
ax2.set_ylabel("")

plt.suptitle("Ranking de Juegos por Valoración de la Comunidad (mín. 200 reseñas)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_best_worst_games.png"), dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 9. Top 20 Usuarios con más Reseñas
# **Objetivo:** Detectar los usuarios más prolíficos del dataset y su tendencia de voto.
# El color de cada barra codifica el ratio de positividad, revelando si son críticos
# sistemáticos o fanáticos incondicionales.
# **Nota de privacidad:** Los Steam IDs se truncan a los últimos 8 dígitos para visualización.

# %%
df_top_users = (
    df_reviews
    .filter(col("author_steamid").isNotNull())
    .groupBy("author_steamid")
    .agg(
        count("recommendationid").alias("num_reviews"),
        avg(when(col("voted_up") == True, 1).otherwise(0)).alias("positivity_rate"),
        avg("author_num_games_owned").alias("avg_games_owned")
    )
    .orderBy(col("num_reviews").desc())
    .limit(20)
)

pdf_top_users = df_top_users.toPandas()
# Truncamos el Steam ID para no mostrar identificadores completos
pdf_top_users["user_label"] = pdf_top_users["author_steamid"].astype(str).str[-8:].apply(lambda x: f"...{x}")
pdf_top_users = pdf_top_users.sort_values("num_reviews")

# Mapeamos el ratio de positividad a colores de la paleta RdYlGn
cmap = plt.cm.get_cmap("RdYlGn")
colors = [cmap(r) for r in pdf_top_users["positivity_rate"]]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(pdf_top_users["user_label"], pdf_top_users["num_reviews"], color=colors)
ax.set_xlabel("Número de Reseñas en el Dataset")
ax.set_title("Top 20 Usuarios más Activos (color = ratio de positividad)", fontsize=13, pad=12)

# Añadir barra de color como referencia
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Ratio de Positividad (0=todo negativo, 1=todo positivo)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_top_users.png"), dpi=150)
plt.show()


# %% [markdown]
# ## 10. Perfil del Revisor: Biblioteca vs Actividad (Scatter)
# **Objetivo:** ¿Los usuarios con más juegos en su biblioteca son más o menos activos
# escribiendo reseñas? ¿Y más o menos críticos?
# **Decisión técnica:** Aplicamos `.sample(0.10)` en Spark antes del join para reducir
# la cardinalidad de usuarios únicos. La escala log-log es necesaria por la distribución
# de Pareto típica de Steam (pocos usuarios con bibliotecas y actividades extremas).

# %%
df_user_profile = (
    df_reviews
    .filter(
        col("author_steamid").isNotNull() &
        col("author_num_games_owned").isNotNull() &
        (col("author_num_games_owned") > 0)
    )
    .groupBy("author_steamid")
    .agg(
        avg("author_num_games_owned").alias("games_owned"),
        count("recommendationid").alias("reviews_written"),
        avg(when(col("voted_up") == True, 1).otherwise(0)).alias("positivity_rate"),
        avg(col("author_playtime_forever") / 60.0).alias("avg_playtime_hours")
    )
    .filter(col("reviews_written") >= 3)  # Al menos 3 reseñas para que el ratio sea significativo
    .sample(False, 0.15)  # 15% de usuarios — suficiente para patrones estadísticos
)

pdf_profile = df_user_profile.toPandas()

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    pdf_profile["games_owned"],
    pdf_profile["reviews_written"],
    c=pdf_profile["positivity_rate"],
    cmap="RdYlGn",
    alpha=0.35,
    s=15,
    vmin=0, vmax=1
)
ax.set_xscale("log")
ax.set_yscale("log")
plt.colorbar(scatter, ax=ax, label="Ratio de Positividad")
ax.set_xlabel("Juegos en Biblioteca (escala log)")
ax.set_ylabel("Reseñas Escritas en Dataset (escala log)")
ax.set_title("Perfil del Revisor: Biblioteca vs Actividad revisora", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "10_user_profile_scatter.png"), dpi=150)
plt.show()


# %% [markdown]
# ## 11. Análisis de Rate Bombing: Detección Temporal de Avalanchas Negativas
# **Objetivo:** Identificar días con picos anómalos de reseñas negativas, síntoma
# habitual de campañas coordinadas de rate bombing (review bombing).
# **Método estadístico:** Calculamos el ratio diario de negatividad y aplicamos
# una media móvil de 30 días con bandas de ±2 desviaciones típicas.
# Los días fuera de esa banda se marcan como anomalías potenciales.
# **Estrategia Spark:** `from_unixtime` convierte el Unix timestamp a fecha en el clúster,
# evitando bajar filas sin agregar. El resultado son ~N_días filas, no millones.

# %%
df_daily_reviews = (
    df_reviews
    .withColumn("review_date", to_date(from_unixtime(col("timestamp_created"))))
    .filter(col("review_date").isNotNull())
    .groupBy("review_date")
    .agg(
        count("recommendationid").alias("total"),
        count(when(col("voted_up") == False, True)).alias("negative"),
        count(when(col("voted_up") == True, True)).alias("positive")
    )
    .filter(col("total") >= 100)  # Días con actividad mínima significativa
    .orderBy("review_date")
)

pdf_daily = df_daily_reviews.toPandas()
pdf_daily["review_date"] = pd.to_datetime(pdf_daily["review_date"])
pdf_daily = pdf_daily.sort_values("review_date").reset_index(drop=True)
pdf_daily["neg_ratio"] = (pdf_daily["negative"] / pdf_daily["total"]) * 100

# Detección de anomalías: ventana móvil de 30 días centrada
WINDOW = 30
pdf_daily["rolling_mean"] = pdf_daily["neg_ratio"].rolling(WINDOW, center=True, min_periods=10).mean()
pdf_daily["rolling_std"]  = pdf_daily["neg_ratio"].rolling(WINDOW, center=True, min_periods=10).std()
pdf_daily["upper_band"]   = pdf_daily["rolling_mean"] + 2 * pdf_daily["rolling_std"]
pdf_daily["anomaly"]      = pdf_daily["neg_ratio"] > pdf_daily["upper_band"]

anomalies = pdf_daily[pdf_daily["anomaly"]]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(pdf_daily["review_date"], pdf_daily["neg_ratio"],
        color="#95a5a6", alpha=0.5, linewidth=0.7, label="% Negativas diario")
ax.plot(pdf_daily["review_date"], pdf_daily["rolling_mean"],
        color="#3498db", linewidth=1.8, label=f"Media móvil ({WINDOW}d)")
ax.fill_between(
    pdf_daily["review_date"],
    pdf_daily["rolling_mean"] - 2 * pdf_daily["rolling_std"],
    pdf_daily["upper_band"],
    alpha=0.15, color="#3498db", label="±2 desv. típica"
)
ax.scatter(
    anomalies["review_date"], anomalies["neg_ratio"],
    color="#e74c3c", s=35, zorder=5,
    label=f"Posible rate bombing ({len(anomalies)} días)"
)
ax.set_xlabel("Fecha")
ax.set_ylabel("% Reseñas Negativas ese día")
ax.set_title("Detección de Rate Bombing: Avalanchas Anómalas de Reseñas Negativas", fontsize=13, pad=12)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "11_rate_bombing.png"), dpi=150)
plt.show()


# %%
# Apagamos la sesión de Spark al terminar
spark.stop()
print(f"Procesamiento finalizado. Gráficas guardadas en '{PLOTS_DIR}/'.")

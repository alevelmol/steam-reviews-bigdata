"""
03_build_gold.py — Construcción de la Capa Gold
================================================
Responsabilidad única: leer Silver (Parquet), ejecutar toda la lógica Spark
y escribir las tablas Gold en data/gold/.

NO importa matplotlib, seaborn ni plotly.
El único uso de .toPandas() permitido es en prints de log para verificación de conteo.

Principio arquitectónico:
    La separación entre este script y 04_visualizations.py es el principio de
    *separación de responsabilidades* aplicado a pipelines de datos. La Capa Gold
    actúa como contrato de interfaz: el procesamiento distribuido (Spark) y el
    consumo analítico (Pandas/Matplotlib) pueden evolucionar de forma independiente.
"""

import glob
import os
import shutil

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    avg,
    broadcast,
    col,
    count,
    countDistinct,
    date_trunc,
    desc,
    from_unixtime,
    log1p,
    round,
    to_date,
    when,
)

# Workaround para Java 18+ (en tu caso Java 25)
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    '--driver-java-options "-Djava.security.manager=allow" pyspark-shell'
)

PROCESSED_DIR = "data/processed"
GOLD_DIR = "data/gold"


# ──────────────────────────────────────────────
# Sesión Spark
# ──────────────────────────────────────────────


def create_spark_session() -> SparkSession:
    """
    Crea y configura la sesión de Spark optimizada para ejecución local.

    Decisiones clave:
    - local[*]: usa todos los cores disponibles. En modo local no hay
      distinción driver/executor, por lo que toda la memoria va al driver.
    - spark.driver.memory=4g: necesario para operaciones de shuffle en el
      dataset de reseñas (>1M filas).
    - shuffle.partitions=16: reducimos las 200 particiones por defecto
      a 16, proporcional al número de cores típico de un PC local.
      Menos particiones = menos overhead de scheduling = más velocidad.
    """
    return (
        SparkSession.builder.appName("SteamReviews_BuildGold")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )


# ──────────────────────────────────────────────
# Carga Silver
# ──────────────────────────────────────────────


def load_silver(spark: SparkSession) -> dict:
    """
    Carga todos los DataFrames Silver necesarios.

    Justificación del orden de carga:
    - Las tablas dimensionales (apps, genres, developers) se cargan primero
      porque se pasarán como broadcast() en los joins, lo que requiere que
      Spark las tenga en el plan lógico antes de ejecutar las agregaciones
      sobre reviews.
    - Los archivos de reviews están particionados (part01, part02...) y se
      leen como un único DataFrame lógico mediante desempaquetado de lista.
    """
    reviews_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "reviews_part*.parquet")))
    if not reviews_files:
        raise FileNotFoundError(f"No se encontraron archivos reviews en {PROCESSED_DIR}")

    return {
        "reviews": spark.read.parquet(*reviews_files),
        "apps": spark.read.parquet(os.path.join(PROCESSED_DIR, "applications.parquet")),
        "app_genres": spark.read.parquet(os.path.join(PROCESSED_DIR, "application_genres.parquet")),
        "genres": spark.read.parquet(os.path.join(PROCESSED_DIR, "genres.parquet")),
        "app_developers": spark.read.parquet(os.path.join(PROCESSED_DIR, "application_developers.parquet")),
        "developers": spark.read.parquet(os.path.join(PROCESSED_DIR, "developers.parquet")),
    }


# ──────────────────────────────────────────────
# Funciones build_gm_*
# ──────────────────────────────────────────────


def build_gm_hater_paradox(df_reviews: DataFrame, df_apps: DataFrame) -> DataFrame:
    """
    La Paradoja del Hater: media de horas jugadas por sentimiento de voto, por juego.

    Columnas de salida:
        appid            — identificador del juego
        app_name         — nombre del juego (de Silver applications)
        avg_hours_positive — media de horas jugadas en reseñas con voto positivo
        avg_hours_negative — media de horas jugadas en reseñas con voto negativo
        total_reviews    — total de reseñas con playtime no nulo

    Propósito analítico: revelar si los jugadores que votan negativo tienen
    significativamente más horas de juego que los que votan positivo (el
    "hater paradox" en la cultura de Steam).

    Decisiones técnicas:
    - Filtrado temprano de nulos en author_playtime_forever antes del groupBy
      para evitar que rows nulas contaminen las medias.
    - Threshold total_reviews > 50: descarta juegos con pocas reseñas donde
      la media sería estadísticamente poco robusta.
    - dropna() sobre las columnas de medias: elimina juegos donde solo hay
      un tipo de voto (no hay base de comparación).
    - broadcast(df_apps): df_apps es una tabla pequeña (~10k juegos); enviarla
      a cada executor evita un shuffle costoso al hacer el join.
    """
    df_apps_small = broadcast(
        df_apps.withColumn("appid", col("appid").cast("long")).select("appid", col("name").alias("app_name"))
    )

    return (
        df_reviews
        .filter(col("author_playtime_forever").isNotNull())
        .withColumn("playtime_hours", col("author_playtime_forever") / 60.0)
        .withColumn("appid", col("appid").cast("long"))
        .groupBy("appid")
        .agg(
            avg(when(col("voted_up") == True, col("playtime_hours"))).alias("avg_hours_positive"),
            avg(when(col("voted_up") == False, col("playtime_hours"))).alias("avg_hours_negative"),
            count("recommendationid").alias("total_reviews"),
        )
        .filter(col("total_reviews") > 50)
        .dropna(subset=["avg_hours_positive", "avg_hours_negative"])
        .join(df_apps_small, on="appid", how="inner")
        .select("appid", "app_name", "avg_hours_positive", "avg_hours_negative", "total_reviews")
    )


def build_gm_top_genres(
    df_reviews: DataFrame, df_app_genres: DataFrame, df_genres: DataFrame
) -> DataFrame:
    """
    Conteo de reseñas por género — base del bar chart "Torre de Babel".

    Columnas de salida:
        genre_name    — nombre del género
        review_count  — número total de reseñas en ese género

    Propósito analítico: identificar qué géneros concentran más volumen de
    discusión en Steam. Sin límite de filas: la visualización aplica el Top 20.

    Decisiones técnicas:
    - Doble broadcast (df_genres y la tabla puente df_app_genres): ambas son
      tablas dimensionales pequeñas. El join reviews → genre es un M:M donde
      un appid puede tener múltiples géneros; broadcast evita el shuffle de
      la tabla grande de reseñas.
    - Se guarda el catálogo completo en Gold (sin LIMIT): permite que la
      visualización elija el corte dinámicamente.
    """
    df_genre_bridge = df_app_genres.join(
        broadcast(df_genres.select(col("id").alias("genre_id"), col("name").alias("genre_name"))),
        on="genre_id",
        how="inner",
    )

    return (
        df_reviews.select("recommendationid", "appid")
        .join(broadcast(df_genre_bridge), on="appid", how="inner")
        .groupBy("genre_name")
        .agg(count("recommendationid").alias("review_count"))
        .orderBy(desc("review_count"))
    )


def build_gm_toxicity_base(df_reviews: DataFrame, df_apps: DataFrame) -> DataFrame:
    """
    Tabla base de toxicidad por juego: % de reseñas negativas + tier de precio.

    Columnas de salida:
        appid             — identificador del juego
        total_reviews     — total de reseñas del juego
        negative_reviews  — reseñas con voto negativo
        toxicity_percent  — (negative_reviews / total_reviews) × 100, 2 decimales
        price_tier        — categoría de precio (4 niveles ordenados)

    Propósito analítico: tabla reutilizable por el boxplot, el heatmap y el
    análisis por desarrolladora. Construirla una sola vez evita recalcularla.

    Decisiones técnicas:
    - Filtrado total_reviews > 30: umbral mínimo para que el porcentaje sea
      estadísticamente representativo (evita que un juego con 1 reseña negativa
      tenga 100% de toxicidad).
    - when/otherwise para price_tier: expresión nativa Spark, se ejecuta en los
      executors sin bajar los datos al driver.
    - Las etiquetas de price_tier comienzan con número ("1. ", "2. "...) para que
      la ordenación alfabética coincida con el orden lógico de precio.
    """
    df_apps_priced = broadcast(
        df_apps.select(
            col("appid").cast("long").alias("appid"),
            "is_free",
            "mat_initial_price",
        )
    )

    return (
        df_reviews
        .withColumn("appid", col("appid").cast("long"))
        .groupBy("appid")
        .agg(
            count("recommendationid").alias("total_reviews"),
            count(when(col("voted_up") == False, True)).alias("negative_reviews"),
        )
        .filter(col("total_reviews") > 30)
        .withColumn(
            "toxicity_percent",
            round((col("negative_reviews") / col("total_reviews")) * 100, 2),
        )
        .join(df_apps_priced, on="appid", how="inner")
        .withColumn(
            "price_tier",
            when(col("is_free") == True, "1. Free to Play")
            .when(col("mat_initial_price") < 15, "2. Indie Barato (<15€)")
            .when(col("mat_initial_price") >= 50, "4. AAA (>=50€)")
            .otherwise("3. Mid-range (15-50€)"),
        )
        .select("appid", "total_reviews", "negative_reviews", "toxicity_percent", "price_tier")
    )


def build_gm_early_access_split(df_reviews: DataFrame) -> DataFrame:
    """
    Proporción de reseñas escritas durante Early Access vs post-lanzamiento.

    Columnas de salida:
        written_during_early_access — boolean
        total_reviews               — conteo de reseñas en esa categoría

    Propósito analítico: medir el peso de Early Access en el corpus total de
    reseñas de Steam. Son literalmente 2 filas; no requiere join.
    """
    return (
        df_reviews
        .groupBy("written_during_early_access")
        .agg(count("recommendationid").alias("total_reviews"))
        .dropna()
    )


def build_gm_sentiment_by_genre(
    df_reviews: DataFrame, df_app_genres: DataFrame, df_genres: DataFrame
) -> DataFrame:
    """
    Ratio positivo/negativo por género — base del stacked bar chart.

    Columnas de salida:
        genre_name   — nombre del género
        positive     — reseñas positivas
        negative     — reseñas negativas
        total        — total de reseñas
        pct_positive — porcentaje positivo (1 decimal)
        pct_negative — porcentaje negativo (1 decimal)

    Propósito analítico: identificar qué géneros generan comunidades más
    positivas o más críticas. Sin límite de filas (la visualización aplica Top 15).

    Filtro total > 10_000: garantiza masa crítica estadística para que los
    porcentajes sean representativos y no distorsionados por géneros nicho.
    """
    df_genre_bridge = df_app_genres.join(
        broadcast(df_genres.select(col("id").alias("genre_id"), col("name").alias("genre_name"))),
        on="genre_id",
        how="inner",
    )

    return (
        df_reviews.select("recommendationid", "appid", "voted_up")
        .join(broadcast(df_genre_bridge), on="appid", how="inner")
        .groupBy("genre_name")
        .agg(
            count(when(col("voted_up") == True, True)).alias("positive"),
            count(when(col("voted_up") == False, True)).alias("negative"),
            count("recommendationid").alias("total"),
        )
        .filter(col("total") > 10_000)
        .withColumn("pct_positive", round((col("positive") / col("total")) * 100, 1))
        .withColumn("pct_negative", round((col("negative") / col("total")) * 100, 1))
        .select("genre_name", "positive", "negative", "total", "pct_positive", "pct_negative")
    )


def build_gm_playtime_distribution(df_reviews: DataFrame) -> DataFrame:
    """
    Muestra estratificada de horas jugadas por sentimiento — base del violin plot.

    Columnas de salida:
        log_playtime — log1p(author_playtime_forever / 60.0)
        sentimiento  — "Positivo" o "Negativo"

    Propósito analítico: visualizar la distribución completa (no solo la media)
    de horas jugadas para votantes positivos vs negativos.

    Decisiones técnicas:
    - log1p en Spark antes de bajar a Pandas: transforma outliers extremos
      (jugadores con miles de horas) para que no colapsen la escala del violin.
    - .sample(False, 0.05): el 5% de muestra es estadísticamente representativo
      para distribuciones con >1M filas (teorema central del límite). Incluir
      todas las filas en Gold sería desperdicio de almacenamiento sin ganancia
      analítica para una visualización de distribución.
    """
    return (
        df_reviews
        .withColumn("playtime_hours", col("author_playtime_forever") / 60.0)
        .filter(col("playtime_hours") > 0)
        .withColumn("log_playtime", log1p(col("playtime_hours")))
        .withColumn(
            "sentimiento",
            when(col("voted_up") == True, "Positivo").otherwise("Negativo"),
        )
        .sample(False, 0.05)
        .select("log_playtime", "sentimiento")
    )


def build_gm_toxicity_heatmap(
    df_toxicity_base: DataFrame, df_reviews: DataFrame
) -> DataFrame:
    """
    Cruce precio × Early Access para el heatmap de toxicidad.

    Columnas de salida:
        price_tier        — categoría de precio (de gm_toxicity_base)
        acceso_anticipado — "Early Access" o "Lanzamiento Completo"
        avg_toxicity      — media de toxicity_percent por celda del heatmap

    Propósito analítico: responder si los juegos Early Access son más "tóxicos"
    que los de lanzamiento completo, y si esto varía según el precio.

    Reutiliza df_toxicity_base (ya computado en main()) para evitar recalcular
    el groupBy de toxicidad. Solo añade la dimensión de Early Access.
    """
    df_ea_flag = (
        df_reviews
        .withColumn("appid", col("appid").cast("long"))
        .groupBy("appid")
        .agg(
            avg(when(col("written_during_early_access") == True, 1).otherwise(0)).alias("ea_ratio")
        )
        .withColumn(
            "acceso_anticipado",
            when(col("ea_ratio") > 0.5, "Early Access").otherwise("Lanzamiento Completo"),
        )
        .select("appid", "acceso_anticipado")
    )

    return (
        df_toxicity_base
        .join(broadcast(df_ea_flag), on="appid", how="inner")
        .groupBy("price_tier", "acceso_anticipado")
        .agg(avg("toxicity_percent").alias("avg_toxicity"))
    )


def build_gm_game_ratings(
    df_reviews: DataFrame,
    df_apps: DataFrame,
    df_app_developers: DataFrame,
    df_developers: DataFrame,
) -> DataFrame:
    """
    Ranking de juegos por porcentaje de reseñas positivas.

    Columnas de salida:
        appid           — identificador del juego
        app_name        — nombre del juego
        developer_name  — nombre del desarrollador principal
        total_reviews   — total de reseñas (>= 50)
        positive_reviews — reseñas con voto positivo
        pct_positive    — (positive_reviews / total_reviews) × 100, 1 decimal

    Propósito analítico: base para los charts 08a (mejor valorados), 08b (peor
    valorados) y 13 (hater paradox por desarrolladora).

    developer_name: se añade para permitir la Gráfica 13 sin un join adicional
    en Pandas. Un juego puede tener múltiples developers; se toma el de menor
    id (developer_id mínimo) como "desarrollador principal".

    Cast appid a long: asegura compatibilidad de tipos en el join con df_apps,
    que puede tener appid como int32 o int64 según la inferencia de Parquet.
    """
    # Desarrollador principal por juego: el de menor developer_id
    df_primary_dev = (
        df_app_developers
        .withColumn("appid", col("appid").cast("long"))
        .groupBy("appid")
        .agg({"developer_id": "min"})
        .withColumnRenamed("min(developer_id)", "developer_id")
        .join(
            broadcast(df_developers.select(col("id").alias("developer_id"), col("name").alias("developer_name"))),
            on="developer_id",
            how="left",
        )
        .select("appid", "developer_name")
    )

    df_apps_small = broadcast(
        df_apps
        .withColumn("appid", col("appid").cast("long"))
        .select("appid", col("name").alias("app_name"))
    )

    return (
        df_reviews
        .withColumn("appid", col("appid").cast("long"))
        .groupBy("appid")
        .agg(
            count("recommendationid").alias("total_reviews"),
            count(when(col("voted_up") == True, True)).alias("positive_reviews"),
        )
        .filter(col("total_reviews") >= 50)
        .withColumn(
            "pct_positive",
            round((col("positive_reviews") / col("total_reviews")) * 100, 1),
        )
        .join(df_apps_small, on="appid", how="inner")
        .filter(col("app_name").isNotNull() & (col("app_name") != ""))
        .join(broadcast(df_primary_dev), on="appid", how="left")
        .select("appid", "app_name", "developer_name", "total_reviews", "positive_reviews", "pct_positive")
    )


def build_gm_top_users(df_reviews: DataFrame) -> DataFrame:
    """
    Top 20 usuarios más activos con ratio de positividad y biblioteca media.

    Columnas de salida:
        author_steamid   — Steam ID del usuario
        num_reviews      — número de reseñas escritas
        positivity_rate  — ratio de votos positivos (0.0 a 1.0)
        avg_games_owned  — media de juegos en biblioteca declarados en reseñas

    Propósito analítico: detectar los revisores más prolíficos y su tendencia
    de voto (crítico sistemático vs fanático incondicional).
    """
    return (
        df_reviews
        .filter(col("author_steamid").isNotNull())
        .groupBy("author_steamid")
        .agg(
            count("recommendationid").alias("num_reviews"),
            avg(when(col("voted_up") == True, 1).otherwise(0)).alias("positivity_rate"),
            avg("author_num_games_owned").alias("avg_games_owned"),
        )
        .orderBy(desc("num_reviews"))
        .limit(20)
    )


def build_gm_user_profiles(df_reviews: DataFrame) -> DataFrame:
    """
    Perfil de revisor: biblioteca vs actividad — base del scatter interactivo.

    Columnas de salida:
        author_steamid   — Steam ID del usuario
        games_owned      — media de juegos en biblioteca
        reviews_written  — número de reseñas escritas
        positivity_rate  — ratio de votos positivos (0.0 a 1.0)
        avg_playtime_hours — media de horas jugadas total (playtime_forever)

    Propósito analítico: explorar la correlación entre tamaño de biblioteca y
    actividad revisora, y si los usuarios con más juegos son más o menos críticos.

    Decisiones técnicas:
    - Filtro reviews_written >= 3: mínimo para que el ratio sea estadísticamente
      significativo (un usuario con 1 reseña negativa no es "crítico sistemático").
    - .sample(False, 0.15): el 15% es suficiente para patrones estadísticos en
      distribuciones de usuarios de Steam (distribución Pareto: pocos usuarios
      muy activos, muchos con pocas reseñas).
    """
    return (
        df_reviews
        .filter(
            col("author_steamid").isNotNull()
            & col("author_num_games_owned").isNotNull()
            & (col("author_num_games_owned") > 0)
        )
        .groupBy("author_steamid")
        .agg(
            avg("author_num_games_owned").alias("games_owned"),
            count("recommendationid").alias("reviews_written"),
            avg(when(col("voted_up") == True, 1).otherwise(0)).alias("positivity_rate"),
            avg(col("author_playtime_forever") / 60.0).alias("avg_playtime_hours"),
        )
        .filter(col("reviews_written") >= 3)
        .sample(False, 0.15)
        .select("author_steamid", "games_owned", "reviews_written", "positivity_rate", "avg_playtime_hours")
    )


def build_gm_daily_reviews(df_reviews: DataFrame) -> DataFrame:
    """
    Serie temporal diaria de reseñas con conteos positivo/negativo.

    Columnas de salida:
        review_date — fecha (tipo date)
        total       — reseñas totales ese día
        negative    — reseñas negativas ese día
        positive    — reseñas positivas ese día

    Propósito analítico: base para la detección de rate bombing. La media
    móvil y las bandas ±2σ se calculan en 04_visualizations.py con Pandas,
    ya que son operaciones de ventana sobre una serie temporal pequeña
    (~N_días filas) que no justifican el overhead de Spark.

    from_unixtime convierte el timestamp Unix a fecha directamente en los
    executors, evitando bajar filas sin agregar al driver.
    """
    return (
        df_reviews
        .withColumn("review_date", to_date(from_unixtime(col("timestamp_created"))))
        .filter(col("review_date").isNotNull())
        .groupBy("review_date")
        .agg(
            count("recommendationid").alias("total"),
            count(when(col("voted_up") == False, True)).alias("negative"),
            count(when(col("voted_up") == True, True)).alias("positive"),
        )
        .filter(col("total") >= 100)
        .orderBy("review_date")
    )


def build_gm_developer_performance(
    df_reviews: DataFrame,
    df_apps: DataFrame,
    df_app_developers: DataFrame,
    df_developers: DataFrame,
) -> DataFrame:
    """
    Rendimiento por desarrolladora: nº de juegos, positividad media, horas medias.

    Columnas de salida:
        developer_name   — nombre del desarrollador/publisher
        num_games        — número de juegos distintos en el dataset
        total_reviews    — total de reseñas en todos sus juegos
        positivity_rate  — ratio medio de votos positivos (0.0 a 1.0)
        avg_playtime_hours — media de horas jugadas (playtime_forever)

    Propósito analítico: Gráfica 12. Identificar qué studios tienen mejor
    recepción por la comunidad y mayor engagement (horas jugadas).

    Filtros:
    - num_games >= 3: excluye estudios con un solo juego donde el dato
      agregado no es representativo del studio sino del juego.
    - total_reviews >= 200: garantiza masa crítica para los promedios.

    Join strategy: reviews → apps (broadcast) → app_developers → developers
    (broadcast). Apps y developers son tablas pequeñas, el shuffle solo
    ocurre en reviews que es la tabla grande.
    """
    df_apps_small = broadcast(
        df_apps.withColumn("appid", col("appid").cast("long")).select("appid")
    )
    df_dev_bridge = (
        df_app_developers
        .withColumn("appid", col("appid").cast("long"))
        .join(
            broadcast(
                df_developers.select(col("id").alias("developer_id"), col("name").alias("developer_name"))
            ),
            on="developer_id",
            how="inner",
        )
        .select("appid", "developer_name")
    )

    return (
        df_reviews
        .withColumn("appid", col("appid").cast("long"))
        .join(df_apps_small, on="appid", how="inner")
        .join(broadcast(df_dev_bridge), on="appid", how="inner")
        .groupBy("developer_name")
        .agg(
            countDistinct("appid").alias("num_games"),
            count("recommendationid").alias("total_reviews"),
            avg(when(col("voted_up") == True, 1).otherwise(0)).alias("positivity_rate"),
            avg(col("author_playtime_forever") / 60.0).alias("avg_playtime_hours"),
        )
        .filter(col("num_games") >= 3)
        .filter(col("total_reviews") >= 200)
    )


def build_gm_genre_timeline(
    df_reviews: DataFrame,
    df_app_genres: DataFrame,
    df_genres: DataFrame,
    df_top_genres: DataFrame,
) -> DataFrame:
    """
    Evolución mensual del volumen de reseñas para los top 5 géneros.

    Columnas de salida:
        review_month — primer día del mes (date_trunc "month")
        genre_name   — nombre del género
        review_count — número de reseñas en ese mes y género

    Propósito analítico: Gráfica 14 (stacked area chart). Muestra cómo han
    crecido o declinado los géneros dominantes a lo largo del tiempo.

    Top 5 géneros: se filtran tomando los 5 primeros de gm_top_genres (ya
    computado), lo que evita recalcular el ranking con un Window adicional.

    date_trunc("month", ...): trunca la fecha al primer día del mes,
    produciendo una granularidad mensual sin perder el tipo date.
    """
    top5_genres = [row["genre_name"] for row in df_top_genres.limit(5).collect()]

    df_genre_bridge = df_app_genres.join(
        broadcast(df_genres.select(col("id").alias("genre_id"), col("name").alias("genre_name"))),
        on="genre_id",
        how="inner",
    ).filter(col("genre_name").isin(top5_genres))

    return (
        df_reviews
        .withColumn(
            "review_month",
            date_trunc("month", to_date(from_unixtime(col("timestamp_created")))),
        )
        .filter(col("review_month").isNotNull())
        .select("recommendationid", "appid", "review_month")
        .join(broadcast(df_genre_bridge), on="appid", how="inner")
        .groupBy("review_month", "genre_name")
        .agg(count("recommendationid").alias("review_count"))
        .orderBy("review_month", "genre_name")
    )


# ──────────────────────────────────────────────
# Escritura de Gold
# ──────────────────────────────────────────────


def write_gold(df: DataFrame, table_name: str) -> None:
    """
    Escribe un DataFrame como un único archivo Parquet en data/gold/.

    Spark siempre escribe en un directorio (part-*.parquet + _SUCCESS).
    Para obtener un archivo plano —igual que la capa Silver de ingesta—
    se usa el mismo patrón: escribir con coalesce(1) en un directorio
    temporal, mover el part file al destino final y borrar el directorio.
    """
    final_path = os.path.join(GOLD_DIR, f"{table_name}.parquet")
    temp_dir = final_path + "_tmp"

    df.coalesce(1).write.mode("overwrite").parquet(temp_dir)

    part_files = glob.glob(os.path.join(temp_dir, "part-*.parquet"))
    if not part_files:
        raise RuntimeError(f"No se encontró part file en {temp_dir}")

    if os.path.exists(final_path):
        os.remove(final_path)
    shutil.move(part_files[0], final_path)
    shutil.rmtree(temp_dir)

    print(f"  ✓ {table_name}.parquet escrito")


# ──────────────────────────────────────────────
# Orquestador principal
# ──────────────────────────────────────────────


def main() -> None:
    os.makedirs(GOLD_DIR, exist_ok=True)

    spark = create_spark_session()
    print("Sesión Spark iniciada.")

    silver = load_silver(spark)
    df_reviews = silver["reviews"]
    df_apps = silver["apps"]
    df_app_genres = silver["app_genres"]
    df_genres = silver["genres"]
    df_app_developers = silver["app_developers"]
    df_developers = silver["developers"]

    print("Silver cargado. Construyendo tablas Gold...")

    # ── gm_hater_paradox ──
    print("\n[1/13] gm_hater_paradox")
    df_hater = build_gm_hater_paradox(df_reviews, df_apps)
    write_gold(df_hater, "gm_hater_paradox")

    # ── gm_top_genres ──
    print("\n[2/13] gm_top_genres")
    df_top_genres = build_gm_top_genres(df_reviews, df_app_genres, df_genres)
    write_gold(df_top_genres, "gm_top_genres")

    # ── gm_toxicity_base ──
    print("\n[3/13] gm_toxicity_base")
    df_toxicity_base = build_gm_toxicity_base(df_reviews, df_apps)
    # Cache: es reutilizada por gm_toxicity_heatmap en la misma sesión
    df_toxicity_base.cache()
    write_gold(df_toxicity_base, "gm_toxicity_base")

    # ── gm_early_access_split ──
    print("\n[4/13] gm_early_access_split")
    df_ea_split = build_gm_early_access_split(df_reviews)
    write_gold(df_ea_split, "gm_early_access_split")

    # ── gm_sentiment_by_genre ──
    print("\n[5/13] gm_sentiment_by_genre")
    df_sentiment = build_gm_sentiment_by_genre(df_reviews, df_app_genres, df_genres)
    write_gold(df_sentiment, "gm_sentiment_by_genre")

    # ── gm_playtime_distribution ──
    print("\n[6/13] gm_playtime_distribution")
    df_playtime = build_gm_playtime_distribution(df_reviews)
    write_gold(df_playtime, "gm_playtime_distribution")

    # ── gm_toxicity_heatmap ──
    print("\n[7/13] gm_toxicity_heatmap")
    df_heatmap = build_gm_toxicity_heatmap(df_toxicity_base, df_reviews)
    write_gold(df_heatmap, "gm_toxicity_heatmap")

    # ── gm_game_ratings ──
    print("\n[8/13] gm_game_ratings")
    df_game_ratings = build_gm_game_ratings(df_reviews, df_apps, df_app_developers, df_developers)
    write_gold(df_game_ratings, "gm_game_ratings")

    # ── gm_top_users ──
    print("\n[9/13] gm_top_users")
    df_top_users = build_gm_top_users(df_reviews)
    write_gold(df_top_users, "gm_top_users")

    # ── gm_user_profiles ──
    print("\n[10/13] gm_user_profiles")
    df_user_profiles = build_gm_user_profiles(df_reviews)
    write_gold(df_user_profiles, "gm_user_profiles")

    # ── gm_daily_reviews ──
    print("\n[11/13] gm_daily_reviews")
    df_daily = build_gm_daily_reviews(df_reviews)
    write_gold(df_daily, "gm_daily_reviews")

    # ── gm_developer_performance ──
    print("\n[12/13] gm_developer_performance")
    df_dev_perf = build_gm_developer_performance(df_reviews, df_apps, df_app_developers, df_developers)
    write_gold(df_dev_perf, "gm_developer_performance")

    # ── gm_genre_timeline ──
    print("\n[13/13] gm_genre_timeline")
    df_timeline = build_gm_genre_timeline(df_reviews, df_app_genres, df_genres, df_top_genres)
    write_gold(df_timeline, "gm_genre_timeline")

    df_toxicity_base.unpersist()
    spark.stop()
    print(f"\nCapa Gold completa. Tablas escritas en '{GOLD_DIR}/'.")


if __name__ == "__main__":
    main()

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast, avg, round, to_date, from_unixtime

# Workaround para Java 18+ (en tu caso Java 25)
os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-java-options "-Djava.security.manager=allow" pyspark-shell'

def create_spark_session() -> SparkSession:
    """
    Crea y configura la sesión de Spark manteniendo las optimizaciones locales.
    """
    return (
        SparkSession.builder
        .appName("SteamReviews_Analytics_SilverToGold")
        .master("local[*]") 
        .config("spark.driver.memory", "4g") 
        .config("spark.sql.shuffle.partitions", "16") 
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )

def main():
    # Rutas a los datos procesados en formato Parquet
    PROCESSED_DIR = "data/processed"
    
    reviews_path = os.path.join(PROCESSED_DIR, "reviews.parquet")
    apps_path = os.path.join(PROCESSED_DIR, "applications.parquet")
    app_genres_path = os.path.join(PROCESSED_DIR, "application_genres.parquet")
    genres_path = os.path.join(PROCESSED_DIR, "genres.parquet")

    spark = create_spark_session()
    
    # 1. Lectura de las tablas (Capa Silver)
    print("--- Cargando datos en formato Parquet ---")
    df_reviews = spark.read.parquet(reviews_path)
    df_apps = spark.read.parquet(apps_path)
    
    # Para traer los géneros, necesitamos cruzar con las tablas puente y de catálogo de géneros
    df_app_genres = spark.read.parquet(app_genres_path)
    df_genres = spark.read.parquet(genres_path)

    # 2. Limpieza básica y Casteos
    print("--- Limpiando datos y ajustando tipos ---")
    
    df_reviews_clean = (
        df_reviews
        .dropna(subset=["appid"]) # Eliminamos filas sin appid
        # Convertimos la fecha (suele venir en unix timestamp en estos datasets) a DateType. 
        # Si venía como string, to_date(col("timestamp_created")) funcionaría. 
        # Si es un número entero/double (unix time), usamos from_unixtime.
        # Por seguridad y flexibilidad en este dataset, aplicaremos un casteo básico a Date (o Timestamp).
        .withColumn("review_date", to_date(col("timestamp_created").cast("string"))) 
        # Convertimos las horas jugadas a tipo numérico y las pasamos de minutos a horas
        .withColumn("playtime_hours", col("author_playtime_at_review").cast("double") / 60.0)
        # Limpiamos nulos en las métricas clave para el análisis
        .dropna(subset=["playtime_hours", "voted_up"])
    )

    # Preparamos las tablas dimensionales (Catálogos)
    df_apps_clean = df_apps.select("appid", col("name").alias("app_name")).dropna(subset=["appid"])
    df_genres_clean = df_genres.select(col("id").alias("genre_id"), col("name").alias("genre_name"))

    # 3. Construcción del Broadcast Join
    print("--- Realizando Broadcast Join y Cálculo de Métricas ---")
    
    # ¿Por qué Broadcast Join? (Explicación para la defensa del proyecto)
    # ----------------------------------------------------------------------
    # En Big Data, la tabla de reviews es la "Tabla de Hechos" (Fact Table) y puede tener millones de registros.
    # Las tablas de juegos (applications) y géneros son "Tablas Dimensionales" (Dimension Tables) y son mucho más pequeñas.
    # Un Join normal provocaría un "Shuffle" masivo: movería todas las reviews por la red/memoria para agruparlas
    # con los juegos correspondientes, saturando los recursos del entorno local.
    # Al usar broadcast(), le decimos a Spark que envíe una copia completa de la tabla de juegos/géneros 
    # a la memoria de cada nodo (o hilo en local) de trabajo. 
    # Esto evita totalmente el Shuffle de la tabla masiva de reviews, logrando un rendimiento extremadamente superior.
    # ----------------------------------------------------------------------
    
    # Primero unimos los juegos con sus géneros (ambas son tablas pequeñas, podemos usar broadcast)
    df_apps_enriched = (
        df_apps_clean
        .join(broadcast(df_app_genres), on="appid", how="inner")
        .join(broadcast(df_genres_clean), on="genre_id", how="inner")
    )
    
    # Ahora cruzamos las millones de reviews con el catálogo de juegos+géneros (Broadcast Join principal)
    # Nota: Si un juego tiene 3 géneros (ej. Acción, RPG, Aventura), una review generará 3 filas,
    # lo cual es perfecto para nuestro objetivo de agrupar el tiempo de juego por género.
    df_gold = df_reviews_clean.join(
        broadcast(df_apps_enriched),
        on="appid",
        how="inner"
    )

    # 4. Agregación Analítica (Métrica Interesante)
    # Calculamos el tiempo medio de juego en el momento de la reseña, agrupado por:
    # - Voto (voted_up = True/False)
    # - Género del juego
    df_analytics = (
        df_gold
        .groupBy("genre_name", "voted_up")
        .agg(
            round(avg("playtime_hours"), 2).alias("avg_playtime_hours"),
            # También podemos añadir un conteo para ver la popularidad y dar más contexto al profesor
            # count("appid").alias("total_reviews") 
        )
        # Ordenamos para ver los géneros con más tiempo jugado en reseñas positivas
        .orderBy(col("voted_up").desc(), col("avg_playtime_hours").desc())
    )

    # 5. Mostrar resultados
    print("\n--- TOP 20: Tiempo Medio de Juego por Género y Tipo de Reseña ---")
    df_analytics.show(20, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()

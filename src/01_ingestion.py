import os
from pyspark.sql import SparkSession, DataFrame

# Workaround para Java 18+ (en tu caso Java 25)
# Permite a Spark acceder al Security Manager antiguo para que no lance el error "getSubject is not supported"
os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-java-options "-Djava.security.manager=allow" pyspark-shell'

def create_spark_session() -> SparkSession:
    """
    Crea y configura una sesión de Spark optimizada para ejecución local.
    """
    return (
        SparkSession.builder
        .appName("SteamReviews_Ingestion_BronzeToSilver")
        # .master("local[*]"): Indica a Spark que se ejecute en modo local, utilizando 
        # tantos hilos (threads) de trabajo como núcleos lógicos tenga tu CPU. No dependemos de un clúster.
        .master("local[*]") 
        
        # Optimización local 1 - Memoria del Driver:
        # En modo local, el driver y los executors comparten la misma JVM (Máquina Virtual de Java). 
        # Aumentamos la memoria asignada a 4GB para asegurarnos de que Spark tenga suficiente RAM 
        # al inferir el esquema de archivos grandes (como el CSV de 1M de reseñas) sin lanzar un OutOfMemoryError.
        .config("spark.driver.memory", "4g") 
        
        # Optimización local 2 - Particiones de Shuffle:
        # Por defecto, cuando Spark hace operaciones que mueven datos entre nodos (shuffle) como joins o aggregations,
        # divide los datos en 200 particiones. En un entorno local, 200 particiones generan un exceso de tareas pequeñas
        # (overhead de gestión). Reducirlo a un número menor (ej. 8 o equivalente al doble de tus cores) acelera el proceso.
        .config("spark.sql.shuffle.partitions", "16") 
        
        # Workaround para Java 18+
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        
        .getOrCreate()
    )

def ingest_csv_to_parquet(spark: SparkSession, csv_path: str, parquet_path: str, entity_name: str) -> DataFrame:
    """
    Lee un archivo CSV (Capa Bronze), infiere su esquema, maneja caracteres especiales
    y lo guarda en formato columnar Parquet (Capa Silver).
    
    Retorna el DataFrame para mantener la modularidad.
    """
    print(f"--- Iniciando ingesta de {entity_name} ---")
    
    # Lectura del DataFrame (Lazy Evaluation):
    # En este punto, Spark aún no lee el archivo (solo cuando es estrictamente necesario).
    # Opciones clave:
    # - header=True: La primera fila contiene los nombres de las columnas.
    # - inferSchema=True: Spark hace una pasada previa para adivinar el tipo de dato (Int, String, etc.).
    # - escape='"': Las reviews de Steam están escritas por usuarios y a menudo contienen comillas 
    #   dentro del texto. Si no escapamos las comillas, Spark romperá las filas al leer.
    # - multiLine=True: Esencial para reseñas largas que contienen saltos de línea (Enters) en su texto.
    df = (
        spark.read
        .option("header", "True")
        .option("inferSchema", "True")
        .option("escape", '"')
        .option("multiLine", "True")
        .csv(csv_path)
    )
    
    # df.printSchema() lee el esquema inferido y lo muestra por consola.
    print(f"Esquema inferido para {entity_name}:")
    df.printSchema()
    
    # Acción de Spark: df.count() es una "Action".
    # Debido a la evaluación perezosa (Lazy Evaluation) de Spark, las transformaciones anteriores 
    # no se ejecutan realmente hasta que llamamos a una Action como count() o write().
    # Aquí es cuando Spark procesa los datos en memoria.
    row_count = df.count()
    print(f"Total de registros cargados en {entity_name}: {row_count}")
    
    # Escritura del DataFrame:
    # Transformamos a formato Parquet. Parquet es un formato de almacenamiento columnar.
    # Ventajas frente a CSV:
    # 1. Comprime los datos mucho mejor (ocupa menos espacio en disco y memoria).
    # 2. Guarda el esquema internamente (no hay que volver a inferirlo).
    # 3. Permite leer solo las columnas necesarias en el futuro sin cargar toda la fila.
    # mode("overwrite") garantiza que el pipeline sea idempotente (puedes ejecutarlo varias veces sin duplicar datos).
    (
        df.write
        .mode("overwrite") # hay que tener cuidado con la ruta porque si no se le pasa un archivo, borra la carpeta completa.
        .parquet(parquet_path)
    )
    print(f"Datos de {entity_name} guardados exitosamente en {parquet_path}\n")
    
    return df

def main():
    # Rutas base asumiendo ejecución desde la raíz del proyecto
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    
    # Creación del directorio de destino si no existe
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Inicializar sesión optimizada
    spark = create_spark_session()
    
    # Iterar sobre todos los archivos CSV en la carpeta raw
    if os.path.exists(RAW_DIR):
        for filename in os.listdir(RAW_DIR):
            if filename.endswith(".csv"):
                # Generar nombre amigable para los logs (ej: 'application_categories.csv' -> 'Application Categories')
                entity_name = filename.replace(".csv", "").replace("_", " ").title()
                
                csv_path = os.path.join(RAW_DIR, filename)
                parquet_filename = filename.replace(".csv", ".parquet")
                parquet_path = os.path.join(PROCESSED_DIR, parquet_filename)
                
                ingest_csv_to_parquet(spark, csv_path, parquet_path, entity_name)
    else:
        print(f"Error: El directorio {RAW_DIR} no existe.")

    # Buenas prácticas: Detener la sesión de Spark al terminar para liberar recursos de memoria del PC.
    spark.stop()

if __name__ == "__main__":
    main()

import os
import glob
import math
import shutil
from pyspark.sql import SparkSession, DataFrame

# Límite de GitHub para archivos individuales
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024   # 100 MB
# Tamaño objetivo por parte (margen de seguridad para no rozar el límite)
TARGET_FILE_SIZE_BYTES = 90 * 1024 * 1024  # 90 MB

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

def _write_parquet(df: DataFrame, parquet_path: str, entity_name: str) -> None:
    """
    Escribe el DataFrame como archivos Parquet limpios (sin archivos de control de Spark).

    Estrategia en dos fases:
    Fase 1 — Escritura de prueba con coalesce(1):
        Se escribe el dataset completo en un único archivo temporal para conocer su tamaño
        real en Parquet. No es posible calcularlo antes porque la tasa de compresión de
        Parquet (Snappy por defecto) varía según la naturaleza de los datos.

    Fase 2 — Decisión según tamaño:
        a) Si el archivo <= MAX_FILE_SIZE_BYTES: se mueve directamente a la ruta final.
        b) Si supera el límite (caso típico en el CSV de reseñas): se calcula el número de
           particiones necesarias como ceil(tamaño_real / TARGET_FILE_SIZE_BYTES), se
           re-escribe con repartition(n) y cada parte se guarda como
           {nombre}_part01.parquet, {nombre}_part02.parquet, etc.

    Justificación de repartition vs coalesce para la fase de división:
        coalesce(n) reduce particiones sin shuffle, pero produce archivos de tamaño muy
        desigual porque las particiones originales no están balanceadas. repartition(n)
        hace un shuffle completo que redistribuye las filas uniformemente, garantizando
        partes de tamaño similar y controlable.

    En ambas fases el directorio temporal se elimina al final, lo que borra
    automáticamente los ficheros de control (_SUCCESS, .crc) generados por Spark.
    """
    base_path = parquet_path.replace(".parquet", "")
    temp_dir = parquet_path + "_tmp"

    # --- Fase 1: escritura de prueba para medir tamaño real ---
    df.coalesce(1).write.mode("overwrite").parquet(temp_dir)

    part_files = glob.glob(os.path.join(temp_dir, "part-*.parquet"))
    if not part_files:
        raise RuntimeError(f"No se generó ningún archivo Parquet en {temp_dir}")

    file_size = os.path.getsize(part_files[0])
    size_mb = file_size / (1024 ** 2)

    # --- Fase 2a: archivo único (dentro del límite) ---
    if file_size <= MAX_FILE_SIZE_BYTES:
        if os.path.exists(parquet_path):
            os.remove(parquet_path)
        shutil.move(part_files[0], parquet_path)
        shutil.rmtree(temp_dir)
        print(f"  Tamaño: {size_mb:.1f} MB → guardado como archivo único.")
        print(f"  Archivo: {parquet_path}\n")
        return

    # --- Fase 2b: particionado (supera el límite de GitHub) ---
    num_partitions = math.ceil(file_size / TARGET_FILE_SIZE_BYTES)
    print(f"  Tamaño: {size_mb:.1f} MB → supera 100 MB, dividiendo en {num_partitions} partes.")

    shutil.rmtree(temp_dir)

    temp_dir_multi = parquet_path + "_tmp_multi"
    df.repartition(num_partitions).write.mode("overwrite").parquet(temp_dir_multi)

    part_files = sorted(glob.glob(os.path.join(temp_dir_multi, "part-*.parquet")))

    # Limpiar partes anteriores de ejecuciones previas
    for old_file in glob.glob(f"{base_path}_part*.parquet"):
        os.remove(old_file)

    for i, pf in enumerate(part_files, start=1):
        dest = f"{base_path}_part{i:02d}.parquet"
        shutil.move(pf, dest)
        part_mb = os.path.getsize(dest) / (1024 ** 2)
        print(f"  → {os.path.basename(dest)} ({part_mb:.1f} MB)")

    shutil.rmtree(temp_dir_multi)
    print(f"  Datos de {entity_name} guardados en {num_partitions} partes bajo {os.path.dirname(parquet_path)}/\n")


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
    
    _write_parquet(df, parquet_path, entity_name)
    
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

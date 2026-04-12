# steam-reviews-bigdata

Pipeline de Big Data en PySpark procesando +1M de reseñas de Steam con arquitectura Medallion. Incluye transformación CSV → Parquet, cómputo distribuido en local con Spark y analítica visual sobre la Capa Gold.

---

## Arquitectura Medallion

```
data/raw/          data/processed/         data/gold/            data/output/plots/
(CSV Bronze)  →    (Parquet Silver)   →    (Parquets Gold)   →   (PNG / HTML)
                   01_ingestion.py         03_build_gold.py       04_visualizations.py
```

| Capa | Formato | Descripción |
|------|---------|-------------|
| **Bronze** | CSV | Datos crudos del dataset de Steam 2025 |
| **Silver** | Parquet Snappy | Datos ingeridos, tipados y limpios |
| **Gold** | Parquet Snappy | Agregados analíticos pre-calculados |
| **Output** | PNG / HTML | Gráficas estáticas e interactivas |

**Principio de separación de responsabilidades:** `03_build_gold.py` (Spark) y `04_visualizations.py` (Pandas) son independientes. El procesamiento distribuido y el consumo analítico pueden evolucionar sin acoplamiento.

---

## Scripts del pipeline

| Script | Responsabilidad | Dependencias |
|--------|----------------|--------------|
| `src/01_ingestion.py` | Bronze → Silver: lee CSV, infiere esquema, escribe Parquet | PySpark |
| `src/03_build_gold.py` | Silver → Gold: ejecuta toda la lógica Spark y escribe agregados | PySpark |
| `src/04_visualizations.py` | Gold → Plots: lee Gold con Pandas y genera gráficas | Pandas, Matplotlib, Seaborn, Plotly, scikit-learn |

Los scripts en `src/deprecated/` conservan el código histórico exploratorio y no forman parte del pipeline activo.

---

## Orden de ejecución

```bash
# 1. Bronze → Silver (una sola vez, o cuando cambian los datos fuente)
python src/01_ingestion.py

# 2. Silver → Gold (recalcular si cambia la lógica analítica)
python src/03_build_gold.py

# 3. Gold → Plots (regenerar gráficas sin re-ejecutar Spark)
python src/04_visualizations.py
```

---

## Tablas Gold generadas (`data/gold/`)

| Archivo | Columnas de salida | Propósito |
|---------|-------------------|-----------|
| `gm_hater_paradox.parquet` | `appid, app_name, avg_hours_positive, avg_hours_negative, total_reviews` | ¿Los haters han jugado más horas? |
| `gm_top_genres.parquet` | `genre_name, review_count` | Ranking de géneros por volumen de reseñas |
| `gm_toxicity_base.parquet` | `appid, total_reviews, negative_reviews, toxicity_percent, price_tier` | % de reseñas negativas por juego y tier de precio |
| `gm_early_access_split.parquet` | `written_during_early_access, total_reviews` | Proporción Early Access vs post-lanzamiento |
| `gm_sentiment_by_genre.parquet` | `genre_name, positive, negative, total, pct_positive, pct_negative` | Ratio positivo/negativo por género |
| `gm_playtime_distribution.parquet` | `log_playtime, sentimiento` | Muestra 5% del log de horas jugadas por sentimiento |
| `gm_toxicity_heatmap.parquet` | `price_tier, acceso_anticipado, avg_toxicity` | Toxicidad media por precio × tipo de acceso |
| `gm_game_ratings.parquet` | `appid, app_name, developer_name, total_reviews, positive_reviews, pct_positive` | Ranking de juegos por % positivo |
| `gm_top_users.parquet` | `author_steamid, num_reviews, positivity_rate, avg_games_owned` | Top 20 usuarios más activos |
| `gm_user_profiles.parquet` | `author_steamid, games_owned, reviews_written, positivity_rate, avg_playtime_hours` | Perfil de revisor (muestra 15%) |
| `gm_daily_reviews.parquet` | `review_date, total, negative, positive` | Serie temporal diaria de reseñas |
| `gm_developer_performance.parquet` | `developer_name, num_games, total_reviews, positivity_rate, avg_playtime_hours` | Rendimiento por desarrolladora |
| `gm_genre_timeline.parquet` | `review_month, genre_name, review_count` | Evolución mensual Top 5 géneros |

---

## Visualizaciones generadas (`data/output/plots/`)

| Archivo | Tipo | Fuente Gold |
|---------|------|-------------|
| `01_hater_paradox.png` | Scatter | `gm_hater_paradox` |
| `02_top_genres_babel.png` | Bar horizontal | `gm_top_genres` |
| `03_toxicity_vs_price.png` | Boxplot | `gm_toxicity_base` |
| `04_early_access_donut.png` | Donut chart | `gm_early_access_split` |
| `05_sentiment_by_genre.png` | Stacked bar | `gm_sentiment_by_genre` |
| `06_playtime_violin.png` | Violin plot | `gm_playtime_distribution` |
| `07_toxicity_heatmap.png` | Heatmap | `gm_toxicity_heatmap` |
| `08a_best_games.png` | Bar horizontal | `gm_game_ratings` |
| `08b_worst_games.png` | Bar horizontal | `gm_game_ratings` |
| `09_top_users.png` | Bar horizontal | `gm_top_users` |
| `10_user_profile_scatter.html/.png` | Scatter interactivo | `gm_user_profiles` |
| `11_rate_bombing.html/.png` | Serie temporal interactiva | `gm_daily_reviews` |
| `12_developer_ranking.png` | Grouped bar | `gm_developer_performance` |
| `13_developer_hater_paradox.html/.png` | Scatter interactivo | `gm_developer_performance` + `gm_game_ratings` + `gm_hater_paradox` |
| `14_genre_timeline.html/.png` | Stacked area interactivo | `gm_genre_timeline` |
| `15_negative_cdf_by_price.png` | CDF | `gm_toxicity_base` + `gm_hater_paradox` |

---

## Dependencias

```bash
pip install -r requirements.txt
```

Paquetes principales:

| Paquete | Uso |
|---------|-----|
| `pyspark` | Procesamiento distribuido (Bronze→Silver, Silver→Gold) |
| `pandas` | Lectura de Gold y manipulación para visualización |
| `matplotlib` | Gráficas estáticas |
| `seaborn` | Gráficas estadísticas sobre Matplotlib |
| `plotly` | Gráficas interactivas (HTML) |
| `scikit-learn` | TF-IDF para análisis NLP del rate bombing |
| `numpy` | Operaciones numéricas (CDF, log) |
| `umap-learn` | (Opcional) Reducción de dimensionalidad para embeddings |

---

## Datos fuente (`data/raw/`)

Dataset relacional de Steam 2025. Archivos CSV ingeridos en Silver:

| CSV | Descripción |
|-----|-------------|
| `applications.csv` | Metadata de juegos (precio, plataformas, metacritic...) |
| `application_genres.csv` | Tabla puente appid ↔ genre_id |
| `genres.csv` | Catálogo de géneros |
| `application_developers.csv` | Tabla puente appid ↔ developer_id |
| `developers.csv` | Catálogo de desarrolladores |
| `reviews_*.csv` | Reseñas de usuarios (>1M filas, particionadas en Silver) |

> Las reseñas en Silver se particionan automáticamente en archivos `reviews_part01.parquet`, `reviews_part02.parquet`, etc. cuando el tamaño supera 100 MB (límite de GitHub).

"""
04_visualizations.py — Visualizaciones de la Capa Gold
=======================================================
Responsabilidad única: leer tablas Gold con pd.read_parquet() y generar gráficas.

NO importa PySpark. Todas las fuentes de datos se leen desde data/gold/.

Principio arquitectónico:
    Este script es el consumidor de la Capa Gold. Al leer agregados pre-calculados
    (filas en orden de miles, no millones), Pandas opera sobre volúmenes que caben
    cómodamente en RAM sin riesgo de OOM. El coste de computación Spark ya fue
    pagado en 03_build_gold.py.
"""

import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

GOLD_DIR = "data/gold"
PLOTS_DIR = "data/output/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _gold(table_name: str) -> str:
    """Devuelve la ruta al parquet Gold de una tabla."""
    return os.path.join(GOLD_DIR, f"{table_name}.parquet")


def _save(fig_or_path, filename: str, *, plotly_fig=False, dpi: int = 150) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    if plotly_fig:
        fig_or_path.write_html(path.replace(".png", ".html"))
        fig_or_path.write_image(path, scale=1.5)
    else:
        fig_or_path.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_or_path)
    print(f"  ✓ {filename}")
    return path


# ──────────────────────────────────────────────
# Gráficas 01–11 (migradas de 03_gold_analytics_visuals.py)
# ──────────────────────────────────────────────


def plot_01_hater_paradox() -> None:
    """
    Gráfica 01 — La Paradoja del Hater (Scatter Plot)
    Fuente Gold: gm_hater_paradox.parquet
    """
    pdf = pd.read_parquet(_gold("gm_hater_paradox"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        pdf["avg_hours_positive"],
        pdf["avg_hours_negative"],
        s=pdf["total_reviews"] / pdf["total_reviews"].max() * 400 + 20,
        alpha=0.6,
        color="#e74c3c",
    )
    ax.plot([0, 500], [0, 500], "k--", zorder=0, label="Mismas horas (1:1)")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_title("La Paradoja del Hater: Horas Jugadas (Positivo vs Negativo por Juego)", fontsize=14, pad=15)
    ax.set_xlabel("Media de Horas Jugadas (Voto Positivo)")
    ax.set_ylabel("Media de Horas Jugadas (Voto Negativo)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "01_hater_paradox.png")


def plot_02_top_genres_babel() -> None:
    """
    Gráfica 02 — Top Géneros: La Torre de Babel (Bar Chart horizontal)
    Fuente Gold: gm_top_genres.parquet
    """
    pdf = pd.read_parquet(_gold("gm_top_genres")).head(20)
    pdf = pdf.sort_values("review_count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("plasma", n_colors=len(pdf))
    sns.barplot(
        data=pdf, x="review_count", y="genre_name",
        hue="genre_name", palette=palette, legend=False, ax=ax,
    )
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1_000_000)}M" if x >= 1_000_000 else f"{int(x/1_000)}K")
    )
    ax.set_title("Top 20 Géneros con más Reseñas (La Torre de Babel)", fontsize=14, pad=15)
    ax.set_xlabel("Número de Reseñas")
    ax.set_ylabel("Género")
    fig.tight_layout()
    _save(fig, "02_top_genres_babel.png")


def plot_03_toxicity_vs_price() -> None:
    """
    Gráfica 03 — Toxicidad vs Precio (Boxplot)
    Fuente Gold: gm_toxicity_base.parquet
    """
    pdf = pd.read_parquet(_gold("gm_toxicity_base"), columns=["price_tier", "toxicity_percent"])
    pdf = pdf.sort_values("price_tier")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=pdf, x="price_tier", y="toxicity_percent",
        hue="price_tier", palette="viridis", legend=False, ax=ax,
    )
    ax.set_title("Toxicidad (% Reseñas Negativas) vs Precio del Juego", fontsize=14, pad=15)
    ax.set_xlabel("Categoría de Precio")
    ax.set_ylabel("Porcentaje de Reseñas Negativas (%)")
    fig.tight_layout()
    _save(fig, "03_toxicity_vs_price.png")


def plot_04_early_access_donut() -> None:
    """
    Gráfica 04 — Early Access vs Post Lanzamiento (Donut Chart)
    Fuente Gold: gm_early_access_split.parquet
    """
    pdf = pd.read_parquet(_gold("gm_early_access_split"))
    pdf["Status"] = pdf["written_during_early_access"].map({
        True: "Durante Early Access",
        False: "Post Lanzamiento",
    })

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#2ecc71", "#3498db"]
    explode = (0.05, 0)
    ax.pie(
        pdf["total_reviews"],
        labels=pdf["Status"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=explode,
        pctdistance=0.85,
        textprops={"fontsize": 12, "color": "black"},
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    ax.add_artist(centre_circle)
    ax.set_title("Distribución de Reseñas: Early Access vs Post Lanzamiento", fontsize=15)
    fig.tight_layout()
    _save(fig, "04_early_access_donut.png")


def plot_05_sentiment_by_genre() -> None:
    """
    Gráfica 05 — Ratio de Positividad por Género (Stacked Bar Chart)
    Fuente Gold: gm_sentiment_by_genre.parquet
    """
    pdf = (
        pd.read_parquet(_gold("gm_sentiment_by_genre"))
        .sort_values("pct_positive", ascending=False)
        .head(15)
    )
    genres = pdf["genre_name"]
    x = range(len(genres))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(x, pdf["pct_positive"], color="#2ecc71", label="Positivas (%)")
    ax.barh(x, pdf["pct_negative"], left=pdf["pct_positive"], color="#e74c3c", label="Negativas (%)")
    ax.set_yticks(list(x))
    ax.set_yticklabels(genres)
    ax.axvline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="50%")
    ax.set_xlabel("Porcentaje de Reseñas (%)")
    ax.set_title("Ratio de Positividad por Género (Top 15 por % positivo)", fontsize=14, pad=15)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "05_sentiment_by_genre.png")


def plot_06_playtime_violin() -> None:
    """
    Gráfica 06 — Distribución de Horas Jugadas por Sentimiento (Violin Plot, escala log)
    Fuente Gold: gm_playtime_distribution.parquet

    La escala logarítmica ya está aplicada en Gold (log_playtime = log1p(hours)).
    Las etiquetas del eje Y muestran horas reales mediante transformación inversa.
    """
    pdf = pd.read_parquet(_gold("gm_playtime_distribution"))

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(
        data=pdf, x="sentimiento", y="log_playtime",
        palette={"Positivo": "#2ecc71", "Negativo": "#e74c3c"},
        inner="quartile", cut=0, ax=ax,
    )
    tick_vals = [0, np.log1p(1), np.log1p(10), np.log1p(100), np.log1p(500), np.log1p(2000)]
    tick_labels = ["0h", "1h", "10h", "100h", "500h", "2000h"]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)
    ax.set_title("Distribución de Horas Jugadas por Sentimiento (escala log)", fontsize=14, pad=15)
    ax.set_xlabel("Tipo de Voto")
    ax.set_ylabel("Horas Jugadas (escala logarítmica)")
    fig.tight_layout()
    _save(fig, "06_playtime_violin.png")


def plot_07_toxicity_heatmap() -> None:
    """
    Gráfica 07 — Heatmap de Toxicidad: Precio × Early Access
    Fuente Gold: gm_toxicity_heatmap.parquet
    """
    pdf = pd.read_parquet(_gold("gm_toxicity_heatmap"))
    pivot = pdf.pivot(index="price_tier", columns="acceso_anticipado", values="avg_toxicity")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, cbar_kws={"label": "% Reseñas Negativas (media)"}, ax=ax,
    )
    ax.set_title("Heatmap de Toxicidad: Rango de Precio × Tipo de Acceso", fontsize=14, pad=15)
    ax.set_xlabel("Tipo de Acceso")
    ax.set_ylabel("Categoría de Precio")
    fig.tight_layout()
    _save(fig, "07_toxicity_heatmap.png")


def plot_08_game_ratings() -> None:
    """
    Gráficas 08a & 08b — Top 20 Juegos Mejor y Peor Valorados (Bar Chart horizontal)
    Fuente Gold: gm_game_ratings.parquet
    """
    pdf = pd.read_parquet(_gold("gm_game_ratings"))

    pdf_best = (
        pdf.nlargest(20, "pct_positive")
        .dropna(subset=["app_name", "pct_positive"])
        .sort_values("pct_positive", ascending=True)
        .reset_index(drop=True)
    )
    pdf_worst = (
        pdf.nsmallest(20, "pct_positive")
        .dropna(subset=["app_name", "pct_positive"])
        .sort_values("pct_positive", ascending=False)
        .reset_index(drop=True)
    )

    for suffix, data, palette_name, title in [
        ("08a_best_games.png", pdf_best, "Greens_r", "Top 20 Juegos Mejor Valorados por la Comunidad (mín. 50 reseñas)"),
        ("08b_worst_games.png", pdf_worst, "Reds_r", "Top 20 Juegos Peor Valorados por la Comunidad (mín. 50 reseñas)"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette(palette_name, n_colors=len(data))
        bars = ax.barh(data["app_name"], data["pct_positive"], color=colors)
        for bar, (_, row) in zip(bars, data.iterrows()):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{row["pct_positive"]:.1f}%  ({int(row["total_reviews"]):,} reseñas)',
                va="center", ha="left", fontsize=8.5,
            )
        ax.set_xlim(0, 115)
        ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="50%")
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel("% Reseñas Positivas")
        ax.legend()
        fig.tight_layout()
        _save(fig, suffix)


def plot_09_top_users() -> None:
    """
    Gráfica 09 — Top 20 Usuarios más Activos (Bar Chart coloreado por positividad)
    Fuente Gold: gm_top_users.parquet
    """
    pdf = pd.read_parquet(_gold("gm_top_users"))
    pdf["user_label"] = pdf["author_steamid"].astype(str)
    pdf = pdf.sort_values("num_reviews")

    cmap = plt.cm.get_cmap("RdYlGn")
    colors = [cmap(r) for r in pdf["positivity_rate"]]

    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(pdf["user_label"], pdf["num_reviews"], color=colors)
    for bar, count_val in zip(bars, pdf["num_reviews"]):
        ax.text(
            bar.get_width() * 0.97, bar.get_y() + bar.get_height() / 2,
            f"{int(count_val):,}", va="center", ha="right",
            fontsize=9, fontweight="bold", color="white",
        )
    ax.set_xlabel("Número de Reseñas en el Dataset")
    ax.set_title("Top 20 Usuarios más Activos (color = ratio de positividad)", fontsize=13, pad=12)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Ratio de Positividad (0=todo negativo, 1=todo positivo)")
    fig.tight_layout()
    _save(fig, "09_top_users.png")


def plot_10_user_profile_scatter() -> None:
    """
    Gráfica 10 — Perfil del Revisor: Biblioteca vs Actividad (Scatter interactivo)
    Fuente Gold: gm_user_profiles.parquet
    """
    pdf = pd.read_parquet(_gold("gm_user_profiles"))
    pdf["positivity_pct"] = (pdf["positivity_rate"] * 100).round(1)
    pdf["avg_playtime_hours"] = pdf["avg_playtime_hours"].round(1)
    pdf["steamid_str"] = pdf["author_steamid"].astype(str)

    fig = px.scatter(
        pdf,
        x="games_owned", y="reviews_written",
        color="positivity_rate",
        color_continuous_scale="RdYlGn", range_color=[0, 1],
        opacity=0.45, log_x=True, log_y=True,
        hover_name="steamid_str",
        hover_data={
            "games_owned": True, "reviews_written": True,
            "positivity_pct": True, "avg_playtime_hours": True,
            "positivity_rate": False, "steamid_str": False, "author_steamid": False,
        },
        labels={
            "games_owned": "Juegos en Biblioteca",
            "reviews_written": "Reseñas Escritas",
            "positivity_pct": "Positividad (%)",
            "avg_playtime_hours": "Media Horas Jugadas",
        },
        title="Perfil del Revisor: Biblioteca vs Actividad Revisora (interactivo)",
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        coloraxis_colorbar=dict(title="Positividad"),
        xaxis_title="Juegos en Biblioteca (escala log)",
        yaxis_title="Reseñas Escritas (escala log)",
        width=950, height=650,
    )
    _save(fig, "10_user_profile_scatter.png", plotly_fig=True)


def plot_11_rate_bombing(silver_reviews_dir: str = "data/processed") -> None:
    """
    Gráfica 11 — Detección de Rate Bombing (Serie temporal interactiva)
    Fuente Gold: gm_daily_reviews.parquet
    Fuente Silver (NLP): reviews_part*.parquet — se usa Pandas directamente
    porque el NLP (TF-IDF) opera sobre texto y no requiere Spark.

    Método estadístico:
    - Media móvil de 30 días con bandas ±2σ.
    - Días cuyo % negativo supera la banda superior = posibles rate bombing.
    - TF-IDF sobre reseñas en inglés de esos días para identificar palabras clave.
    """
    import glob as _glob

    pdf = pd.read_parquet(_gold("gm_daily_reviews"))
    pdf["review_date"] = pd.to_datetime(pdf["review_date"])
    pdf = pdf.sort_values("review_date").reset_index(drop=True)
    pdf["neg_ratio"] = (pdf["negative"] / pdf["total"]) * 100

    WINDOW = 30
    pdf["rolling_mean"] = pdf["neg_ratio"].rolling(WINDOW, center=True, min_periods=10).mean()
    pdf["rolling_std"] = pdf["neg_ratio"].rolling(WINDOW, center=True, min_periods=10).std()
    pdf["upper_band"] = pdf["rolling_mean"] + 2 * pdf["rolling_std"]
    pdf["anomaly"] = pdf["neg_ratio"] > pdf["upper_band"]

    anomalies = pdf[pdf["anomaly"]]
    anomaly_date_strings = [str(d.date()) for d in anomalies["review_date"].tolist()]

    top_games_per_day: dict = {}
    keywords_per_day: dict = {}

    if anomaly_date_strings:
        # Cargar Silver reviews para NLP (solo columnas necesarias)
        review_files = sorted(_glob.glob(os.path.join(silver_reviews_dir, "reviews_part*.parquet")))
        cols_needed = ["timestamp_created", "appid", "voted_up", "language", "review_text", "recommendationid"]
        pdf_reviews = pd.concat(
            [pd.read_parquet(f, columns=cols_needed) for f in review_files],
            ignore_index=True,
        )
        pdf_reviews["review_date"] = pd.to_datetime(
            pdf_reviews["timestamp_created"], unit="s", errors="coerce"
        ).dt.date.astype(str)
        pdf_reviews = pdf_reviews[pdf_reviews["review_date"].isin(anomaly_date_strings)]

        # Cargar aplicaciones para nombres
        pdf_apps = pd.read_parquet(
            os.path.join(silver_reviews_dir.replace("processed", "processed"), "applications.parquet"),
            columns=["appid", "name"],
        )

        # Top juegos bombeados por día
        pdf_neg = pdf_reviews[pdf_reviews["voted_up"] == False].copy()
        pdf_neg["appid"] = pd.to_numeric(pdf_neg["appid"], errors="coerce")
        pdf_neg = pdf_neg.merge(pdf_apps.rename(columns={"appid": "appid"}), on="appid", how="left")
        for date_val, group in pdf_neg.groupby("review_date"):
            top3 = (
                group.groupby("name")["recommendationid"]
                .count()
                .nlargest(3)
                .reset_index()
                .values.tolist()
            )
            top_games_per_day[date_val] = "<br>".join(
                [f"• {n}  ({int(c):,} neg)" for n, c in top3 if n and str(n) != "nan"]
            ) or "Sin datos"

        # TF-IDF por día anómalo (reseñas en inglés)
        pdf_en = pdf_reviews[
            (pdf_reviews["voted_up"] == False) &
            (pdf_reviews["language"] == "english") &
            pdf_reviews["review_text"].notna() &
            (pdf_reviews["review_text"] != "")
        ].copy()
        for date_val, group in pdf_en.groupby("review_date"):
            texts = (
                group["review_text"]
                .dropna()
                .str.lower()
                .str.replace(r"[^a-z\s]", " ", regex=True)
                .tolist()
            )
            texts = [t for t in texts if len(t.strip()) > 10]
            if len(texts) >= 3:
                try:
                    tfidf = TfidfVectorizer(
                        max_features=60, stop_words="english",
                        ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                    )
                    tfidf.fit_transform(texts)
                    keywords_per_day[date_val] = ", ".join(tfidf.get_feature_names_out()[:7])
                except Exception:
                    keywords_per_day[date_val] = "Error NLP"
            else:
                keywords_per_day[date_val] = "Pocas reseñas en inglés"

    pdf["date_str"] = pdf["review_date"].dt.date.astype(str)
    pdf["top_games"] = pdf["date_str"].map(top_games_per_day).fillna("")
    pdf["nlp_keywords"] = pdf["date_str"].map(keywords_per_day).fillna("")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pdf["review_date"], y=pdf["neg_ratio"],
        mode="lines", name="% Negativas diario",
        line=dict(color="#95a5a6", width=0.8), opacity=0.6,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>% Negativas: %{y:.2f}%<br>Total reseñas: %{customdata[0]:,}<extra></extra>",
        customdata=pdf[["total"]].values,
    ))
    fig.add_trace(go.Scatter(
        x=pdf["review_date"], y=pdf["rolling_mean"],
        mode="lines", name=f"Media móvil ({WINDOW}d)",
        line=dict(color="#3498db", width=2), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([pdf["review_date"], pdf["review_date"][::-1]]),
        y=pd.concat([pdf["upper_band"], (pdf["rolling_mean"] - 2 * pdf["rolling_std"])[::-1]]),
        fill="toself", fillcolor="rgba(52,152,219,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="±2 desv. típica", hoverinfo="skip",
    ))

    anomaly_rows = pdf[pdf["anomaly"]].copy()
    fig.add_trace(go.Scatter(
        x=anomaly_rows["review_date"], y=anomaly_rows["neg_ratio"],
        mode="markers", name=f"Posible rate bombing ({len(anomaly_rows)} días)",
        marker=dict(color="#e74c3c", size=9, symbol="circle"),
        customdata=anomaly_rows[["total", "negative", "top_games", "nlp_keywords"]].values,
        hovertemplate=(
            "<b>%{x|%d %b %Y} — ANOMALÍA</b><br>"
            "% Negativas: <b>%{y:.2f}%</b><br>"
            "Reseñas ese día: %{customdata[0]:,}  |  Negativas: %{customdata[1]:,}<br>"
            "<br><b>Juegos más bombeados:</b><br>%{customdata[2]}<br>"
            "<br><b>Palabras clave NLP (inglés):</b><br>%{customdata[3]}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Detección de Rate Bombing: Avalanchas Anómalas de Reseñas Negativas (interactivo)",
        xaxis_title="Fecha", yaxis_title="% Reseñas Negativas ese día",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        width=1200, height=560,
    )
    _save(fig, "11_rate_bombing.png", plotly_fig=True)


# ──────────────────────────────────────────────
# Gráficas 12–15 (nuevas)
# ──────────────────────────────────────────────


def plot_12_developer_ranking() -> None:
    """
    Gráfica 12 — Ranking de Desarrolladoras (Grouped horizontal bar chart)
    Fuente Gold: gm_developer_performance.parquet

    Dos subplots side-by-side:
    - Izquierda: positivity_rate (0–1) en verde.
    - Derecha: avg_playtime_hours en azul.
    Top 20 por total_reviews.
    """
    pdf = (
        pd.read_parquet(_gold("gm_developer_performance"))
        .nlargest(20, "total_reviews")
        .sort_values("total_reviews", ascending=True)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

    palette_green = sns.color_palette("Greens_d", n_colors=len(pdf))
    palette_blue = sns.color_palette("Blues_d", n_colors=len(pdf))

    ax1.barh(pdf["developer_name"], pdf["positivity_rate"], color=palette_green)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("Ratio de Positividad (0–1)")
    ax1.set_title("Positividad media", fontsize=12)
    ax1.axvline(0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for i, (val, n_rev) in enumerate(zip(pdf["positivity_rate"], pdf["total_reviews"])):
        ax1.text(val + 0.01, i, f"{val:.2f}  ({int(n_rev):,} reseñas)", va="center", fontsize=7.5)

    ax2.barh(pdf["developer_name"], pdf["avg_playtime_hours"], color=palette_blue)
    ax2.set_xlabel("Media de Horas Jugadas")
    ax2.set_title("Engagement (horas jugadas)", fontsize=12)
    for i, val in enumerate(pdf["avg_playtime_hours"]):
        ax2.text(val + 0.5, i, f"{val:.0f}h", va="center", fontsize=7.5)

    fig.suptitle("Ranking de Desarrolladoras: Positividad y Engagement (Top 20 por volumen)", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "12_developer_ranking.png")


def plot_13_developer_hater_paradox() -> None:
    """
    Gráfica 13 — Desarrolladoras con Mayor 'Hater Paradox' (Scatter Plot)
    Fuente Gold: gm_developer_performance.parquet + gm_game_ratings.parquet

    Cada punto = una desarrolladora. Posición X/Y = media de horas de sus juegos
    para votos positivos vs negativos. Tamaño = num_games. Color = positivity_rate.

    Lógica Pandas:
    1. Unir gm_game_ratings (tiene developer_name, avg_hours_*) con gm_developer_performance.
    2. Agregar por developer_name: media de avg_hours_positive y avg_hours_negative.
    3. Join con developer_performance para traer num_games y positivity_rate.
    """
    pdf_ratings = pd.read_parquet(_gold("gm_game_ratings"))
    pdf_hater = pd.read_parquet(_gold("gm_hater_paradox"))
    pdf_dev = pd.read_parquet(_gold("gm_developer_performance"))

    # Unir hater_paradox con developer_name a través de gm_game_ratings
    pdf_merged = (
        pdf_ratings[["appid", "developer_name"]]
        .merge(pdf_hater[["appid", "avg_hours_positive", "avg_hours_negative"]], on="appid", how="inner")
        .dropna(subset=["developer_name"])
    )
    pdf_dev_hater = (
        pdf_merged
        .groupby("developer_name", as_index=False)
        .agg(
            avg_hours_positive=("avg_hours_positive", "mean"),
            avg_hours_negative=("avg_hours_negative", "mean"),
        )
        .merge(pdf_dev[["developer_name", "num_games", "positivity_rate", "total_reviews"]], on="developer_name", how="inner")
    )

    fig = px.scatter(
        pdf_dev_hater,
        x="avg_hours_positive",
        y="avg_hours_negative",
        size="num_games",
        color="positivity_rate",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        hover_name="developer_name",
        hover_data={
            "avg_hours_positive": ":.1f",
            "avg_hours_negative": ":.1f",
            "num_games": True,
            "total_reviews": True,
            "positivity_rate": ":.2f",
        },
        labels={
            "avg_hours_positive": "Media Horas Votos Positivos",
            "avg_hours_negative": "Media Horas Votos Negativos",
            "num_games": "Nº Juegos",
            "positivity_rate": "Positividad",
        },
        title="Hater Paradox por Desarrolladora: ¿Sus haters juegan más?",
    )
    max_val = max(pdf_dev_hater["avg_hours_positive"].max(), pdf_dev_hater["avg_hours_negative"].max())
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="black", dash="dash", width=1),
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="Positividad"),
        width=950, height=700,
    )
    _save(fig, "13_developer_hater_paradox.png", plotly_fig=True)


def plot_14_genre_timeline() -> None:
    """
    Gráfica 14 — Evolución Mensual por Género (Stacked Area Chart interactivo)
    Fuente Gold: gm_genre_timeline.parquet

    Lógica Pandas:
    - pivot_table para reorganizar de formato largo (month, genre, count)
      a formato ancho (month como índice, géneros como columnas).
    - fill_value=0 para meses sin reseñas en algún género.
    """
    pdf = pd.read_parquet(_gold("gm_genre_timeline"))
    pdf["review_month"] = pd.to_datetime(pdf["review_month"])

    pivot = pdf.pivot_table(
        index="review_month", columns="genre_name", values="review_count", fill_value=0
    ).reset_index()

    genre_cols = [c for c in pivot.columns if c != "review_month"]
    pivot_long = pivot.melt(id_vars="review_month", value_vars=genre_cols, var_name="genre_name", value_name="review_count")

    fig = px.area(
        pivot_long,
        x="review_month", y="review_count", color="genre_name",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        labels={"review_month": "Mes", "review_count": "Reseñas", "genre_name": "Género"},
        title="Evolución Mensual de Reseñas por Género (Top 5)",
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        width=1100, height=550,
    )
    _save(fig, "14_genre_timeline.png", plotly_fig=True)


def plot_15_negative_cdf_by_price() -> None:
    """
    Gráfica 15 — Curva de Adopción Negativa: CDF por rango de precio
    Fuente Gold: gm_toxicity_base.parquet + gm_hater_paradox.parquet

    Interpretación: "a partir de X horas medias de los haters, el Y% de los juegos
    de este rango de precio ya presenta ese nivel de negatividad".

    Lógica Pandas:
    - Join gm_hater_paradox (avg_hours_negative por juego) con
      gm_toxicity_base (price_tier por juego) por appid.
    - Por cada price_tier: ordenar avg_hours_negative de menor a mayor y
      calcular la proporción acumulada (0 a 1).
    - El eje X en escala log revela la distribución de Pareto típica de Steam.
    """
    pdf_tox = pd.read_parquet(_gold("gm_toxicity_base"), columns=["appid", "price_tier"])
    pdf_hater = pd.read_parquet(_gold("gm_hater_paradox"), columns=["appid", "avg_hours_negative"])

    pdf = pdf_tox.merge(pdf_hater, on="appid", how="inner").dropna(subset=["avg_hours_negative"])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
    for (tier, group), color in zip(pdf.groupby("price_tier"), colors):
        sorted_hours = np.sort(group["avg_hours_negative"].values)
        cdf = np.arange(1, len(sorted_hours) + 1) / len(sorted_hours)
        ax.plot(sorted_hours, cdf, label=tier, color=color, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Horas medias jugadas por haters (escala log)")
    ax.set_ylabel("Proporción acumulada de juegos")
    ax.set_title("Curva de Adopción Negativa: CDF por Rango de Precio", fontsize=14, pad=15)
    ax.legend(title="Rango de precio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "15_negative_cdf_by_price.png")


# ──────────────────────────────────────────────
# Orquestador principal
# ──────────────────────────────────────────────


def main() -> None:
    print("Generando visualizaciones desde la Capa Gold...\n")

    print("[Gráficas existentes]")
    plot_01_hater_paradox()
    plot_02_top_genres_babel()
    plot_03_toxicity_vs_price()
    plot_04_early_access_donut()
    plot_05_sentiment_by_genre()
    plot_06_playtime_violin()
    plot_07_toxicity_heatmap()
    plot_08_game_ratings()
    plot_09_top_users()
    plot_10_user_profile_scatter()
    plot_11_rate_bombing()

    print("\n[Nuevas gráficas]")
    plot_12_developer_ranking()
    plot_13_developer_hater_paradox()
    plot_14_genre_timeline()
    plot_15_negative_cdf_by_price()

    print(f"\nTodas las gráficas guardadas en '{PLOTS_DIR}/'.")


if __name__ == "__main__":
    main()

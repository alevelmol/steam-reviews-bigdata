"""
05_dashboard.py — Dashboard Interactivo Web (Steam Analytics)
=============================================================
Lee las 13 tablas de la Capa Gold (Parquet) y genera un dashboard HTML
autocontenido con gráficas interactivas Plotly.js.

Arquitectura:
    Gold Parquet → Pandas → Plotly Python → JSON → HTML + Plotly.js (CDN)

El HTML resultante no necesita servidor web: se abre directamente en el
navegador. Los datos van embebidos como JSON y Plotly.js se carga desde CDN.
"""

import datetime
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Rutas ──────────────────────────────────────────────
GOLD_DIR = "data/gold"
OUTPUT_PATH = "data/output/dashboard.html"

# ── Paleta de colores ──────────────────────────────────
ORANGE = "#f97316"
ORANGE_LIGHT = "#fb923c"
ORANGE_DIM = "#c2410c"
BG_DARK = "#0a0a0a"
BG_CARD = "#141414"
BG_CHART = "#141414"
TEXT_PRIMARY = "#e5e5e5"
TEXT_SECONDARY = "#a3a3a3"
GRID_COLOR = "#252525"
GREEN = "#34d399"
RED = "#ef4444"
BLUE = "#38bdf8"
PURPLE = "#a78bfa"
YELLOW = "#fbbf24"
PINK = "#fb7185"

COLORWAY = [ORANGE, BLUE, GREEN, PURPLE, PINK, YELLOW, RED, ORANGE_LIGHT]


def _gold(name: str) -> str:
    return os.path.join(GOLD_DIR, f"{name}.parquet")


def _theme(fig: go.Figure, height: int = 460) -> go.Figure:
    """Aplica el tema oscuro/naranja a cualquier figura Plotly."""
    fig.update_layout(
        paper_bgcolor=BG_CHART,
        plot_bgcolor=BG_CHART,
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=12),
        title_font=dict(color=ORANGE, size=15, family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor="#333",
            title_font=dict(color=TEXT_SECONDARY),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor="#333",
            title_font=dict(color=TEXT_SECONDARY),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        hoverlabel=dict(
            bgcolor="#1e1e1e", font=dict(color="#fff", family="Inter"),
            bordercolor="#333",
        ),
        height=height,
        autosize=True,
        legend=dict(font=dict(color=TEXT_SECONDARY)),
        colorway=COLORWAY,
    )
    return fig


# ── Carga de datos ─────────────────────────────────────


def load_data() -> dict[str, pd.DataFrame]:
    """Lee todas las tablas Gold en un diccionario de DataFrames."""
    tables = [
        "gm_hater_paradox", "gm_top_genres", "gm_toxicity_base",
        "gm_early_access_split", "gm_sentiment_by_genre",
        "gm_playtime_distribution", "gm_toxicity_heatmap",
        "gm_game_ratings", "gm_top_users", "gm_user_profiles",
        "gm_daily_reviews", "gm_developer_performance", "gm_genre_timeline",
    ]
    data = {}
    for t in tables:
        path = _gold(t)
        if os.path.exists(path):
            data[t] = pd.read_parquet(path)
            print(f"  [OK] {t}  ({len(data[t]):,} filas)")
        else:
            print(f"  [!!] {t}  (no encontrado)")
    return data


# ── KPIs ───────────────────────────────────────────────


def compute_kpis(data: dict) -> list[dict]:
    """Calcula métricas resumen para las tarjetas KPI."""
    total_reviews = int(data["gm_early_access_split"]["total_reviews"].sum())
    total_games = len(data["gm_game_ratings"])
    avg_positivity = float(data["gm_game_ratings"]["pct_positive"].mean())
    total_devs = len(data["gm_developer_performance"])
    total_genres = len(data["gm_top_genres"])

    def _fmt(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    return [
        {"value": _fmt(total_reviews), "label": "Reseñas Analizadas", "detail": f"{total_reviews:,}"},
        {"value": _fmt(total_games), "label": "Juegos en el Dataset", "detail": f"{total_games:,}"},
        {"value": f"{avg_positivity:.1f}%", "label": "Positividad Media", "detail": "Por juego"},
        {"value": str(total_devs), "label": "Desarrolladoras", "detail": f"{total_devs:,} estudios"},
        {"value": str(total_genres), "label": "Géneros Únicos", "detail": f"{total_genres:,} categorías"},
    ]


# ── Constructores de figuras ──────────────────────────


def fig_early_access(data: dict) -> go.Figure:
    pdf = data["gm_early_access_split"].copy()
    pdf["label"] = pdf["written_during_early_access"].map(
        {True: "Early Access", False: "Post Lanzamiento"}
    )
    fig = go.Figure(go.Pie(
        labels=pdf["label"], values=pdf["total_reviews"],
        hole=0.65,
        marker=dict(colors=[ORANGE, BLUE], line=dict(color=BG_CARD, width=2)),
        textinfo="label+percent",
        textfont=dict(color="#fff", size=13),
        hovertemplate="<b>%{label}</b><br>Reseñas: %{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Early Access vs Post Lanzamiento",
        showlegend=False,
    )
    return _theme(fig, height=400)


def fig_top_genres(data: dict) -> go.Figure:
    pdf = data["gm_top_genres"].head(20).sort_values("review_count", ascending=True)
    n = len(pdf)
    colors = [f"rgba(249,115,22,{0.4 + 0.6 * i / n:.2f})" for i in range(n)]

    fig = go.Figure(go.Bar(
        x=pdf["review_count"], y=pdf["genre_name"],
        orientation="h", marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>Reseñas: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Top 20 Géneros por Volumen de Reseñas",
        xaxis_title="Número de Reseñas",
        yaxis_title="",
    )
    return _theme(fig, height=520)


def fig_hater_paradox(data: dict) -> go.Figure:
    pdf = data["gm_hater_paradox"].copy()
    pdf = pdf[(pdf["avg_hours_positive"] <= 500) & (pdf["avg_hours_negative"] <= 500)]
    sizes = pdf["total_reviews"] / pdf["total_reviews"].max() * 30 + 4

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pdf["avg_hours_positive"], y=pdf["avg_hours_negative"],
        mode="markers",
        marker=dict(size=sizes, color=RED, opacity=0.5, line=dict(width=0)),
        text=pdf.get("app_name", pd.Series([""] * len(pdf))),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Horas positivos: %{x:.1f}<br>"
            "Horas negativos: %{y:.1f}<extra></extra>"
        ),
    ))
    fig.add_shape(
        type="line", x0=0, y0=0, x1=500, y1=500,
        line=dict(color=TEXT_SECONDARY, dash="dash", width=1),
    )
    fig.add_annotation(
        x=400, y=420, text="Los haters juegan más →",
        showarrow=False, font=dict(color=TEXT_SECONDARY, size=11),
    )
    fig.update_layout(
        title="La Paradoja del Hater: Horas Jugadas (Positivo vs Negativo)",
        xaxis_title="Media Horas Jugadas (Voto Positivo)",
        yaxis_title="Media Horas Jugadas (Voto Negativo)",
        showlegend=False,
    )
    return _theme(fig, height=500)


def fig_best_games(data: dict) -> go.Figure:
    pdf = (
        data["gm_game_ratings"]
        .nlargest(20, "pct_positive")
        .dropna(subset=["app_name", "pct_positive"])
        .sort_values("pct_positive", ascending=True)
    )
    n = len(pdf)
    colors = [f"rgba(52,211,153,{0.4 + 0.6 * i / n:.2f})" for i in range(n)]

    fig = go.Figure(go.Bar(
        x=pdf["pct_positive"], y=pdf["app_name"],
        orientation="h", marker=dict(color=colors),
        text=[f'{v:.1f}% ({int(r):,})' for v, r in zip(pdf["pct_positive"], pdf["total_reviews"])],
        textposition="outside", textfont=dict(size=10, color=TEXT_SECONDARY),
        hovertemplate="<b>%{y}</b><br>Positividad: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Top 20 Juegos Mejor Valorados (mín. 50 reseñas)",
        xaxis_title="% Reseñas Positivas",
        xaxis=dict(range=[0, 130]),
        margin=dict(l=10, r=120, t=60, b=10),
        yaxis=dict(automargin=True),
    )
    fig.add_vline(x=50, line=dict(color=TEXT_SECONDARY, dash="dash", width=0.8))
    return _theme(fig, height=580)


def fig_worst_games(data: dict) -> go.Figure:
    pdf = (
        data["gm_game_ratings"]
        .nsmallest(20, "pct_positive")
        .dropna(subset=["app_name", "pct_positive"])
        .sort_values("pct_positive", ascending=False)
    )
    n = len(pdf)
    colors = [f"rgba(239,68,68,{0.4 + 0.6 * i / n:.2f})" for i in range(n)]

    fig = go.Figure(go.Bar(
        x=pdf["pct_positive"], y=pdf["app_name"],
        orientation="h", marker=dict(color=colors),
        text=[f'{v:.1f}% ({int(r):,})' for v, r in zip(pdf["pct_positive"], pdf["total_reviews"])],
        textposition="outside", textfont=dict(size=10, color=TEXT_SECONDARY),
        hovertemplate="<b>%{y}</b><br>Positividad: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Top 20 Juegos Peor Valorados (mín. 50 reseñas)",
        xaxis_title="% Reseñas Positivas",
        xaxis=dict(range=[0, 130]),
        margin=dict(l=10, r=120, t=60, b=10),
        yaxis=dict(automargin=True),
    )
    fig.add_vline(x=50, line=dict(color=TEXT_SECONDARY, dash="dash", width=0.8))
    return _theme(fig, height=580)


def fig_sentiment_genre(data: dict) -> go.Figure:
    pdf = (
        data["gm_sentiment_by_genre"]
        .sort_values("pct_positive", ascending=True)
        .tail(15)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pdf["pct_positive"], y=pdf["genre_name"],
        orientation="h", name="Positivas (%)",
        marker=dict(color=GREEN),
        hovertemplate="%{y}: %{x:.1f}% positivas<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=pdf["pct_negative"], y=pdf["genre_name"],
        orientation="h", name="Negativas (%)",
        marker=dict(color=RED),
        hovertemplate="%{y}: %{x:.1f}% negativas<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        title="Ratio de Positividad por Género (Top 15)",
        xaxis_title="Porcentaje de Reseñas (%)",
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=70),
    )
    fig.add_vline(x=50, line=dict(color=TEXT_SECONDARY, dash="dash", width=0.8))
    return _theme(fig, height=520)


def fig_genre_timeline(data: dict) -> go.Figure:
    pdf = data["gm_genre_timeline"].copy()
    pdf["review_month"] = pd.to_datetime(pdf["review_month"])
    pivot = pdf.pivot_table(
        index="review_month", columns="genre_name",
        values="review_count", fill_value=0,
    ).reset_index()

    genre_cols = [c for c in pivot.columns if c != "review_month"]
    palette = [ORANGE, BLUE, GREEN, PURPLE, PINK]

    # Convertir a string para evitar notación científica en el eje X
    x_labels = pivot["review_month"].dt.strftime("%Y-%m-%d").tolist()

    fig = go.Figure()
    for i, genre in enumerate(genre_cols):
        fig.add_trace(go.Scatter(
            x=x_labels, y=pivot[genre].tolist(),
            mode="lines", name=genre, stackgroup="one",
            line=dict(width=0.5, color=palette[i % len(palette)]),
            hovertemplate=f"<b>{genre}</b><br>Mes: %{{x}}<br>Reseñas: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Evolución Mensual de Reseñas por Género (Top 5)",
        xaxis_title="Mes",
        yaxis_title="Reseñas",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=70),
    )
    return _theme(fig, height=480)


def fig_toxicity_price(data: dict) -> go.Figure:
    pdf = data["gm_toxicity_base"][["price_tier", "toxicity_percent"]].copy()
    pdf = pdf.sort_values("price_tier")
    tiers = sorted(pdf["price_tier"].unique())
    palette = [BLUE, GREEN, ORANGE, RED]

    fig = go.Figure()
    for i, tier in enumerate(tiers):
        subset = pdf[pdf["price_tier"] == tier]["toxicity_percent"]
        fig.add_trace(go.Box(
            y=subset, name=str(tier),
            marker=dict(color=palette[i % len(palette)]),
            line=dict(color=palette[i % len(palette)]),
            boxmean="sd",
        ))
    fig.update_layout(
        title="Toxicidad (% Negativas) vs Precio del Juego",
        yaxis_title="% Reseñas Negativas",
        xaxis_title="Categoría de Precio",
        showlegend=False,
    )
    return _theme(fig, height=460)


def fig_toxicity_heatmap(data: dict) -> go.Figure:
    pdf = data["gm_toxicity_heatmap"].copy()
    pivot = pdf.pivot(index="price_tier", columns="acceso_anticipado", values="avg_toxicity")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#1a1a2e"], [0.5, ORANGE_DIM], [1, RED]],
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        textfont=dict(size=14, color="#fff"),
        hovertemplate=(
            "Precio: %{y}<br>Acceso: %{x}<br>"
            "Toxicidad media: %{z:.1f}%<extra></extra>"
        ),
        colorbar=dict(title="% Negativas"),
    ))
    fig.update_layout(
        title="Heatmap de Toxicidad: Precio × Tipo de Acceso",
        xaxis_title="Tipo de Acceso",
        yaxis_title="Categoría de Precio",
    )
    return _theme(fig, height=400)


def fig_rate_bombing(data: dict) -> go.Figure:
    pdf = data["gm_daily_reviews"].copy()
    pdf["review_date"] = pd.to_datetime(pdf["review_date"])
    pdf = pdf.sort_values("review_date").reset_index(drop=True)
    pdf["neg_ratio"] = (pdf["negative"] / pdf["total"]) * 100

    window = 30
    pdf["rolling_mean"] = pdf["neg_ratio"].rolling(window, center=True, min_periods=10).mean()
    pdf["rolling_std"] = pdf["neg_ratio"].rolling(window, center=True, min_periods=10).std()
    pdf["upper_band"] = pdf["rolling_mean"] + 2 * pdf["rolling_std"]
    pdf["lower_band"] = pdf["rolling_mean"] - 2 * pdf["rolling_std"]
    pdf["anomaly"] = pdf["neg_ratio"] > pdf["upper_band"]

    fig = go.Figure()
    # Serie diaria
    fig.add_trace(go.Scatter(
        x=pdf["review_date"], y=pdf["neg_ratio"],
        mode="lines", name="% Negativas diario",
        line=dict(color=TEXT_SECONDARY, width=0.8), opacity=0.5,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>% Negativas: %{y:.2f}%<extra></extra>",
    ))
    # Media móvil
    fig.add_trace(go.Scatter(
        x=pdf["review_date"], y=pdf["rolling_mean"],
        mode="lines", name=f"Media móvil ({window}d)",
        line=dict(color=BLUE, width=2), hoverinfo="skip",
    ))
    # Banda de confianza
    fig.add_trace(go.Scatter(
        x=pd.concat([pdf["review_date"], pdf["review_date"][::-1]]),
        y=pd.concat([pdf["upper_band"], pdf["lower_band"][::-1]]),
        fill="toself", fillcolor="rgba(56,189,248,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="±2σ", hoverinfo="skip",
    ))
    # Anomalías
    anom = pdf[pdf["anomaly"]]
    fig.add_trace(go.Scatter(
        x=anom["review_date"], y=anom["neg_ratio"],
        mode="markers", name=f"Anomalías ({len(anom)} días)",
        marker=dict(color=RED, size=8, symbol="circle",
                    line=dict(width=1, color="#fff")),
        hovertemplate=(
            "<b>%{x|%d %b %Y} — ANOMALÍA</b><br>"
            "% Negativas: <b>%{y:.2f}%</b><extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Detección de Rate Bombing: Anomalías de Negatividad",
        xaxis_title="Fecha", yaxis_title="% Reseñas Negativas",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=80),
    )
    return _theme(fig, height=500)


def fig_cdf_price(data: dict) -> go.Figure:
    pdf_tox = data["gm_toxicity_base"][["appid", "price_tier"]].copy()
    pdf_hater = data["gm_hater_paradox"][["appid", "avg_hours_negative"]].copy()
    pdf = pdf_tox.merge(pdf_hater, on="appid", how="inner").dropna(subset=["avg_hours_negative"])

    colors = [BLUE, GREEN, ORANGE, RED]
    fig = go.Figure()
    for i, (tier, group) in enumerate(sorted(pdf.groupby("price_tier"))):
        sorted_h = np.sort(group["avg_hours_negative"].values)
        cdf = np.arange(1, len(sorted_h) + 1) / len(sorted_h)
        fig.add_trace(go.Scatter(
            x=sorted_h, y=cdf, mode="lines", name=str(tier),
            line=dict(color=colors[i % len(colors)], width=2.5),
            hovertemplate=f"<b>{tier}</b><br>Horas: %{{x:.1f}}<br>CDF: %{{y:.2%}}<extra></extra>",
        ))
    fig.update_layout(
        title="CDF de Horas Jugadas por Haters según Precio",
        xaxis_title="Horas medias por haters (escala log)",
        yaxis_title="Proporción acumulada de juegos",
        xaxis_type="log",
        legend=dict(title="Rango de Precio"),
    )
    return _theme(fig, height=460)


def fig_top_users(data: dict) -> go.Figure:
    pdf = data["gm_top_users"].copy()
    pdf["user_label"] = pdf["author_steamid"].astype(str)
    pdf = pdf.sort_values("num_reviews")

    fig = go.Figure(go.Bar(
        x=pdf["num_reviews"], y=pdf["user_label"],
        orientation="h",
        marker=dict(
            color=pdf["positivity_rate"],
            colorscale="RdYlGn", cmin=0, cmax=1,
            colorbar=dict(title="Positividad"),
            line=dict(width=0),
        ),
        text=[f'{int(n):,}' for n in pdf["num_reviews"]],
        textposition="inside", textfont=dict(color="#fff", size=10),
        hovertemplate=(
            "<b>Steam ID: %{y}</b><br>"
            "Reseñas: %{x:,}<br>"
            "Positividad: %{marker.color:.1%}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Top 20 Usuarios más Activos (color = positividad)",
        xaxis_title="Número de Reseñas",
        yaxis_title="",
    )
    return _theme(fig, height=520)


def fig_user_profiles(data: dict) -> go.Figure:
    pdf = data["gm_user_profiles"].copy()
    pdf["positivity_pct"] = (pdf["positivity_rate"] * 100).round(1)
    pdf["avg_playtime_hours"] = pdf["avg_playtime_hours"].round(1)

    fig = go.Figure(go.Scatter(
        x=pdf["games_owned"], y=pdf["reviews_written"],
        mode="markers",
        marker=dict(
            size=5, opacity=0.45,
            color=pdf["positivity_rate"],
            colorscale="RdYlGn", cmin=0, cmax=1,
            colorbar=dict(title="Positividad"),
            line=dict(width=0),
        ),
        customdata=np.stack([pdf["positivity_pct"], pdf["avg_playtime_hours"]], axis=-1),
        hovertemplate=(
            "Juegos: %{x:,}<br>"
            "Reseñas: %{y:,}<br>"
            "Positividad: %{customdata[0]:.1f}%<br>"
            "Media horas: %{customdata[1]:.1f}h<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Perfil del Revisor: Biblioteca vs Actividad",
        xaxis_title="Juegos en Biblioteca (log)", xaxis_type="log",
        yaxis_title="Reseñas Escritas (log)", yaxis_type="log",
    )
    return _theme(fig, height=500)


def fig_playtime_violin(data: dict) -> go.Figure:
    pdf = data["gm_playtime_distribution"].copy()
    colors = {"Positivo": GREEN, "Negativo": RED}

    fill_colors = {"Positivo": "rgba(52,211,153,0.3)", "Negativo": "rgba(239,68,68,0.3)"}

    fig = go.Figure()
    for sentiment in ["Positivo", "Negativo"]:
        subset = pdf[pdf["sentimiento"] == sentiment]
        fig.add_trace(go.Violin(
            y=subset["log_playtime"], name=sentiment,
            box_visible=True, meanline_visible=True,
            line_color=colors[sentiment],
            fillcolor=fill_colors[sentiment],
            opacity=0.7,
        ))

    # Etiquetas de horas reales en eje Y
    tick_vals = [0, np.log1p(1), np.log1p(10), np.log1p(100), np.log1p(500), np.log1p(2000)]
    tick_labels = ["0h", "1h", "10h", "100h", "500h", "2000h"]
    fig.update_layout(
        title="Distribución de Horas Jugadas por Sentimiento (escala log)",
        yaxis=dict(tickvals=tick_vals, ticktext=tick_labels),
        yaxis_title="Horas Jugadas",
        xaxis_title="Tipo de Voto",
        showlegend=False,
    )
    return _theme(fig, height=480)


def fig_dev_ranking(data: dict) -> go.Figure:
    pdf = (
        data["gm_developer_performance"]
        .nlargest(20, "total_reviews")
        .sort_values("total_reviews", ascending=True)
    )
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=("Positividad Media", "Engagement (horas)"),
        horizontal_spacing=0.08,
    )
    # Positividad
    n = len(pdf)
    green_colors = [f"rgba(52,211,153,{0.4 + 0.6 * i / n:.2f})" for i in range(n)]
    fig.add_trace(go.Bar(
        x=pdf["positivity_rate"], y=pdf["developer_name"],
        orientation="h", marker=dict(color=green_colors),
        text=[f'{v:.2f}' for v in pdf["positivity_rate"]],
        textposition="outside", textfont=dict(size=9, color=TEXT_SECONDARY),
        hovertemplate="<b>%{y}</b><br>Positividad: %{x:.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    # Horas
    blue_colors = [f"rgba(56,189,248,{0.4 + 0.6 * i / n:.2f})" for i in range(n)]
    fig.add_trace(go.Bar(
        x=pdf["avg_playtime_hours"], y=pdf["developer_name"],
        orientation="h", marker=dict(color=blue_colors),
        text=[f'{v:.0f}h' for v in pdf["avg_playtime_hours"]],
        textposition="outside", textfont=dict(size=9, color=TEXT_SECONDARY),
        hovertemplate="<b>%{y}</b><br>Media horas: %{x:.0f}h<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text="Ranking de Desarrolladoras (Top 20 por volumen)", y=0.99),
        margin=dict(l=10, r=10, t=80, b=10),
    )
    fig.update_xaxes(range=[0, 1.1], row=1, col=1)
    fig.add_vline(x=0.7, line=dict(color=TEXT_SECONDARY, dash="dash", width=0.8), row=1, col=1)
    # Subplot titles — bajarlas para que no choquen con el título principal
    for annotation in fig.layout.annotations:
        annotation.font = dict(color=TEXT_SECONDARY, size=12, family="Inter")
        annotation.y = annotation.y - 0.02
    return _theme(fig, height=600)


def fig_dev_hater_paradox(data: dict) -> go.Figure:
    pdf_ratings = data["gm_game_ratings"][["appid", "developer_name"]].copy()
    pdf_hater = data["gm_hater_paradox"][["appid", "avg_hours_positive", "avg_hours_negative"]].copy()
    pdf_dev = data["gm_developer_performance"].copy()

    pdf_merged = (
        pdf_ratings
        .merge(pdf_hater, on="appid", how="inner")
        .dropna(subset=["developer_name"])
    )
    pdf_agg = (
        pdf_merged
        .groupby("developer_name", as_index=False)
        .agg(
            avg_hours_positive=("avg_hours_positive", "mean"),
            avg_hours_negative=("avg_hours_negative", "mean"),
        )
        .merge(
            pdf_dev[["developer_name", "num_games", "positivity_rate", "total_reviews"]],
            on="developer_name", how="inner",
        )
    )

    fig = go.Figure(go.Scatter(
        x=pdf_agg["avg_hours_positive"],
        y=pdf_agg["avg_hours_negative"],
        mode="markers",
        marker=dict(
            size=pdf_agg["num_games"].clip(upper=50) * 2 + 5,
            color=pdf_agg["positivity_rate"],
            colorscale="RdYlGn", cmin=0, cmax=1,
            colorbar=dict(title="Positividad"),
            opacity=0.7, line=dict(width=0.5, color="#333"),
        ),
        text=pdf_agg["developer_name"],
        customdata=np.stack([
            pdf_agg["num_games"], pdf_agg["total_reviews"],
            pdf_agg["positivity_rate"],
        ], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Horas positivos: %{x:.1f}<br>"
            "Horas negativos: %{y:.1f}<br>"
            "Juegos: %{customdata[0]:.0f}<br>"
            "Reseñas: %{customdata[1]:,.0f}<br>"
            "Positividad: %{customdata[2]:.1%}<extra></extra>"
        ),
    ))
    max_val = max(
        pdf_agg["avg_hours_positive"].max(),
        pdf_agg["avg_hours_negative"].max(),
    )
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color=TEXT_SECONDARY, dash="dash", width=1),
    )
    fig.update_layout(
        title="Hater Paradox por Desarrolladora",
        xaxis_title="Media Horas (Votos Positivos)",
        yaxis_title="Media Horas (Votos Negativos)",
    )
    return _theme(fig, height=540)


# ── Serialización ──────────────────────────────────────


def _make_serializable(obj):
    """Convierte recursivamente tipos numpy/pandas a tipos Python nativos."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_make_serializable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def _serialize_fig(fig: go.Figure) -> dict:
    """Convierte un Figure a dict JSON-serializable (sin typed arrays bdata)."""
    return _make_serializable(fig.to_dict())


# ── Plantilla HTML ─────────────────────────────────────


def generate_html(charts: dict[str, dict], kpis: list[dict]) -> str:
    """Genera el HTML completo del dashboard."""
    charts_json = json.dumps(charts, ensure_ascii=False)
    kpis_json = json.dumps(kpis, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Steam Analytics Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0a0a0a;--bg-card:#141414;--bg-card-hover:#1a1a1a;
  --orange:#f97316;--orange-dim:#c2410c;--orange-glow:rgba(249,115,22,0.15);
  --text:#e5e5e5;--text-sec:#a3a3a3;--border:#1f1f1f;--border-hover:#333;
}}
html{{font-size:15px}}
body{{
  font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);
  display:flex;min-height:100vh;overflow-x:hidden;
}}

/* ── Sidebar ── */
.sidebar{{
  width:250px;min-width:250px;height:100vh;position:fixed;top:0;left:0;
  background:linear-gradient(180deg,#0c0c0c 0%,#111 100%);
  border-right:1px solid var(--border);z-index:100;
  display:flex;flex-direction:column;
  transition:transform .3s ease;
}}
.sidebar-header{{
  padding:2rem 1.5rem 1.5rem;border-bottom:1px solid var(--border);
}}
.sidebar-header h1{{
  font-size:1.4rem;font-weight:700;color:var(--orange);letter-spacing:-0.02em;
  line-height:1.2;
}}
.sidebar-header h1 span{{color:var(--text);font-weight:300;display:block;font-size:0.85rem;margin-top:0.25rem;letter-spacing:0.05em;}}
.sidebar nav{{flex:1;padding:1rem 0;overflow-y:auto}}
.nav-item{{
  display:flex;align-items:center;gap:0.75rem;
  padding:0.75rem 1.5rem;cursor:pointer;
  color:var(--text-sec);font-size:0.85rem;font-weight:500;
  border-left:3px solid transparent;
  transition:all .2s ease;text-decoration:none;
}}
.nav-item:hover{{background:rgba(255,255,255,0.03);color:var(--text)}}
.nav-item.active{{
  color:var(--orange);background:var(--orange-glow);
  border-left-color:var(--orange);
}}
.nav-dot{{
  width:6px;height:6px;border-radius:50%;
  background:var(--text-sec);flex-shrink:0;
  transition:background .2s ease;
}}
.nav-item.active .nav-dot{{background:var(--orange);box-shadow:0 0 8px var(--orange)}}
.sidebar-footer{{
  padding:1rem 1.5rem;border-top:1px solid var(--border);
  font-size:0.7rem;color:#555;text-align:center;
}}

/* ── Main content ── */
.content{{margin-left:250px;flex:1;min-height:100vh;padding:2rem}}

/* ── Tab sections ── */
.tab{{display:none;animation:fadeIn .4s ease}}
.tab.active{{display:block}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}

.tab-header{{margin-bottom:1.5rem}}
.tab-header h2{{font-size:1.5rem;font-weight:600;color:var(--text);letter-spacing:-0.02em}}
.tab-header p{{color:var(--text-sec);font-size:0.85rem;margin-top:0.25rem}}
.tab-header::after{{
  content:'';display:block;width:50px;height:3px;
  background:var(--orange);border-radius:2px;margin-top:0.75rem;
}}

/* ── KPI Cards ── */
.kpi-grid{{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:1rem;margin-bottom:1.5rem;
}}
.kpi-card{{
  background:var(--bg-card);border:1px solid var(--border);border-radius:12px;
  padding:1.25rem;text-align:center;
  transition:border-color .3s ease,transform .2s ease;
}}
.kpi-card:hover{{border-color:var(--orange);transform:translateY(-2px)}}
.kpi-value{{
  font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:600;
  color:var(--orange);line-height:1;
}}
.kpi-label{{
  color:var(--text-sec);font-size:0.75rem;text-transform:uppercase;
  letter-spacing:0.06em;margin-top:0.5rem;
}}
.kpi-detail{{color:#555;font-size:0.7rem;margin-top:0.25rem}}

/* ── Chart grid ── */
.chart-grid{{
  display:grid;grid-template-columns:repeat(2,1fr);gap:1.25rem;
}}
.chart-card{{
  background:var(--bg-card);border:1px solid var(--border);border-radius:12px;
  padding:0.75rem;overflow:hidden;
  transition:border-color .3s ease;
}}
.chart-card:hover{{border-color:var(--border-hover)}}
.chart-card.span-2{{grid-column:span 2}}
.chart-container{{width:100%;height:380px}}

/* ── Responsive ── */
.hamburger{{
  display:none;position:fixed;top:1rem;left:1rem;z-index:200;
  background:var(--bg-card);border:1px solid var(--border);border-radius:8px;
  padding:0.5rem 0.75rem;cursor:pointer;color:var(--text);font-size:1.2rem;
}}
@media(max-width:900px){{
  .sidebar{{transform:translateX(-100%)}}
  .sidebar.open{{transform:translateX(0)}}
  .content{{margin-left:0;padding:1rem;padding-top:3.5rem}}
  .hamburger{{display:block}}
  .chart-grid{{grid-template-columns:1fr}}
  .chart-card.span-2{{grid-column:span 1}}
}}
@media(max-width:600px){{
  .kpi-grid{{grid-template-columns:repeat(2,1fr)}}
  .kpi-value{{font-size:1.5rem}}
}}

/* ── Scrollbar ── */
::-webkit-scrollbar{{width:6px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:#333;border-radius:3px}}
::-webkit-scrollbar-thumb:hover{{background:#555}}

/* ── Visual polish ── */
.content{{
  background-image:radial-gradient(rgba(249,115,22,0.022) 1px,transparent 1px);
  background-size:28px 28px;
}}
.sidebar-header{{
  background:linear-gradient(135deg,rgba(249,115,22,0.08) 0%,transparent 70%);
  border-bottom:1px solid rgba(249,115,22,0.18) !important;
  position:relative;
}}
.sidebar-header::after{{
  content:'';position:absolute;bottom:-1px;left:1.5rem;right:1.5rem;
  height:1px;background:linear-gradient(90deg,var(--orange),transparent);
  opacity:0.3;
}}
.sidebar-badge{{
  display:inline-block;font-size:0.6rem;font-weight:600;letter-spacing:0.12em;
  color:var(--orange);background:rgba(249,115,22,0.12);
  border:1px solid rgba(249,115,22,0.25);border-radius:4px;
  padding:0.15rem 0.5rem;margin-bottom:0.5rem;text-transform:uppercase;
}}
.nav-item.active{{
  color:var(--orange);background:var(--orange-glow);
  border-left-color:var(--orange);
  box-shadow:inset 0 0 30px rgba(249,115,22,0.05);
}}
.sidebar-footer{{
  padding:1rem 1.5rem;border-top:1px solid rgba(249,115,22,0.1);
  font-size:0.68rem;color:#444;text-align:center;letter-spacing:0.04em;
}}
.kpi-card{{
  background:linear-gradient(145deg,#1d1d1d 0%,#141414 100%) !important;
  position:relative;overflow:hidden;
}}
.kpi-card::before{{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--orange) 0%,rgba(249,115,22,0.3) 60%,transparent 100%);
}}
.kpi-card:hover{{
  border-color:rgba(249,115,22,0.4) !important;
  box-shadow:0 6px 24px rgba(249,115,22,0.07);
  transform:translateY(-3px) !important;
}}
.kpi-value{{text-shadow:0 0 30px rgba(249,115,22,0.3)}}
.chart-card{{
  position:relative;
}}
.chart-card::before{{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;z-index:1;
  background:linear-gradient(90deg,rgba(249,115,22,0.45) 0%,rgba(249,115,22,0.1) 50%,transparent 100%);
  border-radius:12px 12px 0 0;
}}
.chart-card:hover{{
  border-color:rgba(249,115,22,0.2) !important;
  box-shadow:0 8px 32px rgba(0,0,0,0.35);
}}
.tab-header h2{{
  background:linear-gradient(90deg,#ffffff 30%,#aaaaaa 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}}
.tab-header::after{{
  background:linear-gradient(90deg,var(--orange),rgba(249,115,22,0.2),transparent);
  width:80px;height:3px;
}}

/* ── Search bar ── */
.search-bar{{
  display:flex;align-items:center;gap:0.5rem;
  padding:0.6rem 0.75rem 0.35rem;
  border-bottom:1px solid var(--border);
}}
.search-input{{
  flex:1;background:#0d0d0d;
  border:1px solid #2a2a2a;border-radius:8px;
  color:var(--text);padding:0.45rem 0.85rem 0.45rem 2rem;
  font-family:'Inter',sans-serif;font-size:0.82rem;
  outline:none;transition:border-color .2s,box-shadow .2s;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23555' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:0.6rem center;
}}
.search-input:focus{{
  border-color:var(--orange);
  box-shadow:0 0 0 3px rgba(249,115,22,0.12);
}}
.search-input::placeholder{{color:#3a3a3a}}
.search-clear{{
  background:none;border:1px solid #2a2a2a;
  color:#555;border-radius:6px;
  padding:0.38rem 0.65rem;cursor:pointer;font-size:0.75rem;
  transition:all .2s ease;flex-shrink:0;line-height:1;
}}
.search-clear:hover{{border-color:var(--orange);color:var(--orange)}}
.search-hint{{
  font-size:0.68rem;color:#3a3a3a;
  padding:0.2rem 0.75rem 0.1rem;font-style:italic;
}}
.search-matches{{
  font-size:0.72rem;color:var(--orange);
  padding:0.2rem 0.75rem 0.1rem;min-height:1.2em;
  font-family:'JetBrains Mono',monospace;
}}

/* ── Entrance animations ── */
@keyframes slideUpFade{{
  from{{opacity:0;transform:translateY(18px)}}
  to{{opacity:1;transform:translateY(0)}}
}}
@keyframes fadeInScale{{
  from{{opacity:0;transform:scale(0.96)}}
  to{{opacity:1;transform:scale(1)}}
}}
@keyframes dotPulse{{
  0%,100%{{box-shadow:0 0 6px var(--orange)}}
  50%{{box-shadow:0 0 16px var(--orange),0 0 28px rgba(249,115,22,0.35)}}
}}
@keyframes navShimmer{{
  0%{{background-position:-200% center}}
  100%{{background-position: 200% center}}
}}
@keyframes kpiGlow{{
  from{{text-shadow:0 0 0 rgba(249,115,22,0)}}
  to{{text-shadow:0 0 30px rgba(249,115,22,0.4)}}
}}
@keyframes borderPulse{{
  0%,100%{{border-color:var(--border)}}
  50%{{border-color:rgba(249,115,22,0.3)}}
}}

/* Tab header y KPI grid */
.tab.active .tab-header{{animation:slideUpFade .35s ease both}}
.tab.active .kpi-grid{{animation:slideUpFade .4s ease .04s both}}

/* KPI cards escalonadas */
.tab.active .kpi-card{{animation:slideUpFade .45s cubic-bezier(.22,.68,0,1.2) both}}
.tab.active .kpi-card:nth-child(1){{animation-delay:.06s}}
.tab.active .kpi-card:nth-child(2){{animation-delay:.12s}}
.tab.active .kpi-card:nth-child(3){{animation-delay:.18s}}
.tab.active .kpi-card:nth-child(4){{animation-delay:.24s}}
.tab.active .kpi-card:nth-child(5){{animation-delay:.30s}}
.tab.active .kpi-card:nth-child(6){{animation-delay:.36s}}
.tab.active .kpi-card:nth-child(7){{animation-delay:.42s}}
.tab.active .kpi-value{{animation:slideUpFade .45s ease both,kpiGlow .9s ease .4s both}}

/* Chart cards escalonadas */
.tab.active .chart-card{{animation:fadeInScale .5s cubic-bezier(.22,.68,0,1.2) both}}
.tab.active .chart-card:nth-child(1){{animation-delay:.10s}}
.tab.active .chart-card:nth-child(2){{animation-delay:.19s}}
.tab.active .chart-card:nth-child(3){{animation-delay:.28s}}
.tab.active .chart-card:nth-child(4){{animation-delay:.37s}}
.tab.active .chart-card:nth-child(5){{animation-delay:.46s}}

/* Nav items sidebar */
.sidebar nav .nav-item{{animation:slideUpFade .3s ease both}}
.sidebar nav .nav-item:nth-child(1){{animation-delay:.05s}}
.sidebar nav .nav-item:nth-child(2){{animation-delay:.10s}}
.sidebar nav .nav-item:nth-child(3){{animation-delay:.15s}}
.sidebar nav .nav-item:nth-child(4){{animation-delay:.20s}}
.sidebar nav .nav-item:nth-child(5){{animation-delay:.25s}}
.sidebar nav .nav-item:nth-child(6){{animation-delay:.30s}}
.sidebar-header{{animation:slideUpFade .4s ease both}}
.sidebar-footer{{animation:slideUpFade .35s ease .35s both}}

/* Nav active — shimmer + dot pulsante */
.nav-item{{position:relative;overflow:hidden}}
.nav-item.active::after{{
  content:'';position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(90deg,transparent 0%,rgba(249,115,22,0.07) 50%,transparent 100%);
  background-size:200% 100%;
  animation:navShimmer 3s linear infinite;
}}
.nav-item.active .nav-dot{{animation:dotPulse 2.5s ease infinite}}

/* Hover border pulse on chart cards */
.chart-card:hover{{animation:borderPulse 2s ease infinite}}

/* Search input — micro interacción */
.search-input{{transition:border-color .2s,box-shadow .2s,transform .15s}}
.search-input:focus{{transform:scaleX(1.005);transform-origin:left}}
</style>
</head>
<body>

<!-- Hamburger (mobile) -->
<button class="hamburger" onclick="document.querySelector('.sidebar').classList.toggle('open')" aria-label="Menú">&#9776;</button>

<!-- Sidebar -->
<aside class="sidebar">
  <div class="sidebar-header">
    <div class="sidebar-badge">Big Data · CBD 2025</div>
    <h1>STEAM ANALYTICS<span>Analytics Dashboard</span></h1>
  </div>
  <nav>
    <a class="nav-item active" onclick="switchTab('overview',this)">
      <span class="nav-dot"></span>Resumen General
    </a>
    <a class="nav-item" onclick="switchTab('games',this)">
      <span class="nav-dot"></span>Análisis de Juegos
    </a>
    <a class="nav-item" onclick="switchTab('genres',this)">
      <span class="nav-dot"></span>Géneros
    </a>
    <a class="nav-item" onclick="switchTab('toxicity',this)">
      <span class="nav-dot"></span>Toxicidad
    </a>
    <a class="nav-item" onclick="switchTab('users',this)">
      <span class="nav-dot"></span>Usuarios
    </a>
    <a class="nav-item" onclick="switchTab('developers',this)">
      <span class="nav-dot"></span>Desarrolladoras
    </a>
  </nav>
  <div class="sidebar-footer">
    Proyecto CBD &middot; PySpark &middot; 2025
  </div>
</aside>

<!-- Content -->
<main class="content">

  <!-- ── Overview ── -->
  <section id="tab-overview" class="tab active">
    <div class="tab-header">
      <h2>Resumen General</h2>
      <p>Vista panorámica del ecosistema de reseñas de Steam</p>
    </div>
    <div class="kpi-grid" id="kpi-container"></div>
    <div class="chart-grid">
      <div class="chart-card"><div id="chart-early_access" class="chart-container"></div></div>
      <div class="chart-card"><div id="chart-top_genres" class="chart-container"></div></div>
    </div>
  </section>

  <!-- ── Games ── -->
  <section id="tab-games" class="tab">
    <div class="tab-header">
      <h2>Análisis de Juegos</h2>
      <p>Rankings y patrones de valoración por título</p>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><div id="chart-best_games" class="chart-container"></div></div>
      <div class="chart-card"><div id="chart-worst_games" class="chart-container"></div></div>
      <div class="chart-card span-2">
        <div class="search-bar">
          <input type="text" class="search-input" id="search-hater_paradox"
                 placeholder="Buscar juego por nombre..."
                 oninput="filterScatter('chart-hater_paradox', this.value, 'text')">
          <button class="search-clear" onclick="clearSearch('hater_paradox')">&#x2715;</button>
        </div>
        <div class="search-matches" id="matches-hater_paradox"></div>
        <div id="chart-hater_paradox" class="chart-container"></div>
      </div>
    </div>
  </section>

  <!-- ── Genres ── -->
  <section id="tab-genres" class="tab">
    <div class="tab-header">
      <h2>Géneros</h2>
      <p>Distribución de sentimiento y evolución temporal</p>
    </div>
    <div class="chart-grid">
      <div class="chart-card span-2"><div id="chart-sentiment_genre" class="chart-container"></div></div>
      <div class="chart-card span-2"><div id="chart-genre_timeline" class="chart-container"></div></div>
    </div>
  </section>

  <!-- ── Toxicity ── -->
  <section id="tab-toxicity" class="tab">
    <div class="tab-header">
      <h2>Toxicidad</h2>
      <p>Patrones de negatividad en la comunidad de Steam</p>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><div id="chart-toxicity_price" class="chart-container"></div></div>
      <div class="chart-card"><div id="chart-toxicity_heatmap" class="chart-container"></div></div>
      <div class="chart-card span-2"><div id="chart-rate_bombing" class="chart-container"></div></div>
      <div class="chart-card span-2"><div id="chart-cdf_price" class="chart-container"></div></div>
    </div>
  </section>

  <!-- ── Users ── -->
  <section id="tab-users" class="tab">
    <div class="tab-header">
      <h2>Usuarios</h2>
      <p>Perfiles y comportamiento de los revisores de Steam</p>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><div id="chart-top_users" class="chart-container"></div></div>
      <div class="chart-card"><div id="chart-playtime_violin" class="chart-container"></div></div>
      <div class="chart-card span-2"><div id="chart-user_profiles" class="chart-container"></div></div>
    </div>
  </section>

  <!-- ── Developers ── -->
  <section id="tab-developers" class="tab">
    <div class="tab-header">
      <h2>Desarrolladoras</h2>
      <p>Métricas de rendimiento y engagement por estudio</p>
    </div>
    <div class="chart-grid">
      <div class="chart-card span-2"><div id="chart-dev_ranking" class="chart-container"></div></div>
      <div class="chart-card span-2">
        <div class="search-bar">
          <input type="text" class="search-input" id="search-dev_hater_paradox"
                 placeholder="Buscar desarrolladora por nombre..."
                 oninput="filterScatter('chart-dev_hater_paradox', this.value, 'text')">
          <button class="search-clear" onclick="clearSearch('dev_hater_paradox')">&#x2715;</button>
        </div>
        <div class="search-matches" id="matches-dev_hater_paradox"></div>
        <div id="chart-dev_hater_paradox" class="chart-container"></div>
      </div>
    </div>
  </section>

</main>

<script>
// ── Datos embebidos ──
const CHARTS = {charts_json};
const KPIS = {kpis_json};

// ── Mapa de pestañas a gráficas ──
const TAB_CHARTS = {{
  overview:    ['early_access', 'top_genres'],
  games:       ['best_games', 'worst_games', 'hater_paradox'],
  genres:      ['sentiment_genre', 'genre_timeline'],
  toxicity:    ['toxicity_price', 'toxicity_heatmap', 'rate_bombing', 'cdf_price'],
  users:       ['top_users', 'playtime_violin', 'user_profiles'],
  developers:  ['dev_ranking', 'dev_hater_paradox'],
}};

const PLOTLY_CONFIG = {{
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  toImageButtonOptions: {{ format: 'png', scale: 2 }},
}};

const rendered = new Set();

// ── KPI rendering ──
function renderKPIs() {{
  const container = document.getElementById('kpi-container');
  KPIS.forEach(kpi => {{
    const card = document.createElement('div');
    card.className = 'kpi-card';
    card.innerHTML = `
      <div class="kpi-value">${{kpi.value}}</div>
      <div class="kpi-label">${{kpi.label}}</div>
      <div class="kpi-detail">${{kpi.detail}}</div>
    `;
    container.appendChild(card);
  }});
}}

// ── Tab switching ──
function switchTab(tabId, el) {{
  // Update navigation
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  if (el) el.classList.add('active');

  // Update content
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');

  // Render charts (lazy)
  if (!rendered.has(tabId)) {{
    renderTab(tabId);
    rendered.add(tabId);
  }} else {{
    // Resize existing
    (TAB_CHARTS[tabId] || []).forEach(id => {{
      const div = document.getElementById('chart-' + id);
      if (div && div.data) Plotly.Plots.resize(div);
    }});
  }}

  // Close mobile sidebar
  document.querySelector('.sidebar').classList.remove('open');
}}

function renderTab(tabId) {{
  const chartIds = TAB_CHARTS[tabId] || [];
  chartIds.forEach(id => {{
    const chartData = CHARTS[id];
    const div = document.getElementById('chart-' + id);
    if (chartData && div) {{
      Plotly.newPlot(div, chartData.data, chartData.layout, PLOTLY_CONFIG);
    }}
  }});
}}

// ── Buscadores para scatter plots ──
// Cache de trazas originales (decodificadas por Plotly tras el primer render)
const searchOriginals = {{}};

function filterScatter(chartId, query, textField) {{
  const div = document.getElementById(chartId);
  const baseId = chartId.replace('chart-', '');
  const matchEl = document.getElementById('matches-' + baseId);

  if (!div || !div.data) return;

  // Guardar copia de la traza original la primera vez que se llama
  if (!searchOriginals[baseId]) {{
    searchOriginals[baseId] = JSON.parse(JSON.stringify(div.data[0]));
  }}

  const originalTrace = searchOriginals[baseId];
  const names = originalTrace.text || [];

  query = (query || '').toLowerCase().trim();

  if (!query) {{
    // Restaurar datos originales completos
    Plotly.react(div, [originalTrace], div.layout, PLOTLY_CONFIG);
    if (matchEl) matchEl.textContent = '';
    return;
  }}

  // Índices que coinciden con la búsqueda
  const matching = [];
  names.forEach((n, i) => {{
    if (typeof n === 'string' && n.toLowerCase().includes(query)) matching.push(i);
  }});

  // Construir traza filtrada: sólo se retienen los índices coincidentes
  const filtered = {{}};
  for (const key of Object.keys(originalTrace)) {{
    const val = originalTrace[key];
    if (Array.isArray(val) && val.length === names.length) {{
      filtered[key] = matching.map(i => val[i]);
    }} else {{
      filtered[key] = val;
    }}
  }}

  Plotly.react(div, [filtered], div.layout, PLOTLY_CONFIG);
  if (matchEl) matchEl.textContent = matching.length > 0
    ? matching.length + ' resultado(s) encontrado(s)'
    : 'Sin coincidencias';
}}

function clearSearch(baseId) {{
  const input = document.getElementById('search-' + baseId);
  const matchEl = document.getElementById('matches-' + baseId);
  if (input) {{ input.value = ''; filterScatter('chart-' + baseId, '', 'text'); }}
  if (matchEl) matchEl.textContent = '';
}}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {{
  renderKPIs();
  renderTab('overview');
  rendered.add('overview');

  // Resize on window change
  window.addEventListener('resize', () => {{
    document.querySelectorAll('.chart-container').forEach(div => {{
      if (div.data) Plotly.Plots.resize(div);
    }});
  }});
}});
</script>
</body>
</html>"""


# ── Orquestador ────────────────────────────────────────


def main() -> None:
    print("=" * 55)
    print("  Steam Analytics — Generando Dashboard Interactivo")
    print("=" * 55)

    print("\n[1/3] Cargando tablas Gold...")
    data = load_data()

    print("\n[2/3] Construyendo gráficas interactivas...")
    figures = {
        "early_access":     fig_early_access(data),
        "top_genres":       fig_top_genres(data),
        "hater_paradox":    fig_hater_paradox(data),
        "best_games":       fig_best_games(data),
        "worst_games":      fig_worst_games(data),
        "sentiment_genre":  fig_sentiment_genre(data),
        "genre_timeline":   fig_genre_timeline(data),
        "toxicity_price":   fig_toxicity_price(data),
        "toxicity_heatmap": fig_toxicity_heatmap(data),
        "rate_bombing":     fig_rate_bombing(data),
        "cdf_price":        fig_cdf_price(data),
        "top_users":        fig_top_users(data),
        "user_profiles":    fig_user_profiles(data),
        "playtime_violin":  fig_playtime_violin(data),
        "dev_ranking":      fig_dev_ranking(data),
        "dev_hater_paradox": fig_dev_hater_paradox(data),
    }
    print(f"  [OK] {len(figures)} graficas generadas")

    print("\n[3/3] Generando HTML...")
    kpis = compute_kpis(data)
    charts_serialized = {k: _serialize_fig(v) for k, v in figures.items()}
    html = generate_html(charts_serialized, kpis)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n  [OK] Dashboard guardado en: {OUTPUT_PATH}")
    print(f"  [OK] Tamano: {size_mb:.1f} MB")
    print(f"  [OK] Abrelo directamente en el navegador.")
    print("=" * 55)


if __name__ == "__main__":
    main()

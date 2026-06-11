from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis import sentiment_by_period, topic_distribution_by_period


def create_visualizations(df: pd.DataFrame, config: dict[str, Any], figures_dir: str | Path) -> list[Path]:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    sentiment_path = plot_sentiment_period_comparison(df, config, figures_dir)
    if sentiment_path:
        created.append(sentiment_path)

    created.extend(plot_wordclouds(df, config, figures_dir))
    created.extend(plot_topic_distribution(df, config, figures_dir))
    return created


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _period_label(config: dict[str, Any], period: str) -> str:
    return config.get("periods", {}).get(period, {}).get("label", period)


def plot_sentiment_period_comparison(df: pd.DataFrame, config: dict[str, Any], figures_dir: Path) -> Path | None:
    summary = sentiment_by_period(df)
    if summary.empty:
        print("Sentiment visualization skipped: no sentiment summary data.")
        return None

    plt = _setup_matplotlib()
    pivot = summary.pivot(index="sentiment_label", columns="period", values="percentage").fillna(0)
    pivot = pivot.rename(columns={period: _period_label(config, period) for period in pivot.columns})

    ax = pivot.plot(kind="bar", figsize=(9, 5), width=0.75)
    ax.set_title("Perbandingan Persentase Sentimen per Periode")
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Persentase (%)")
    ax.set_ylim(0, max(100, float(pivot.max().max()) + 10))
    ax.legend(title="Periode")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    path = figures_dir / "sentiment_period_comparison.png"
    plt.savefig(path, dpi=int(config.get("visualization", {}).get("figure_dpi", 150)))
    plt.close()
    return path


def plot_wordclouds(df: pd.DataFrame, config: dict[str, Any], figures_dir: Path) -> list[Path]:
    try:
        from wordcloud import WordCloud
    except ImportError as exc:
        raise RuntimeError("Install wordcloud to create word cloud figures.") from exc

    plt = _setup_matplotlib()
    created: list[Path] = []
    settings = config.get("visualization", {})
    for period in config.get("periods", {}):
        scoped = df[df["period"] == period]
        text = " ".join(scoped.get("clean_text", pd.Series(dtype=str)).dropna().astype(str).tolist())
        if not text.strip():
            print(f"Word cloud skipped for {period}: no clean_text data.")
            continue

        cloud = WordCloud(
            width=int(settings.get("wordcloud_width", 1200)),
            height=int(settings.get("wordcloud_height", 800)),
            background_color="white",
            collocations=False,
        ).generate(text)

        plt.figure(figsize=(10, 6))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud {_period_label(config, period)}")
        plt.tight_layout()

        path = figures_dir / f"wordcloud_{period}.png"
        plt.savefig(path, dpi=int(settings.get("figure_dpi", 150)))
        plt.close()
        created.append(path)
    return created


def plot_topic_distribution(df: pd.DataFrame, config: dict[str, Any], figures_dir: Path) -> list[Path]:
    summary = topic_distribution_by_period(df)
    if summary.empty:
        print("Topic distribution visualization skipped: no topic data.")
        return []

    plt = _setup_matplotlib()
    created: list[Path] = []
    for period in config.get("periods", {}):
        scoped = summary[summary["period"] == period]
        if scoped.empty:
            continue

        plt.figure(figsize=(8, 5))
        plt.bar(scoped["dominant_topic"].astype(str), scoped["percentage"])
        plt.title(f"Distribusi Topik {_period_label(config, period)}")
        plt.xlabel("Topik Dominan")
        plt.ylabel("Persentase (%)")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        path = figures_dir / f"topic_distribution_{period}.png"
        plt.savefig(path, dpi=int(config.get("visualization", {}).get("figure_dpi", 150)))
        plt.close()
        created.append(path)
    return created

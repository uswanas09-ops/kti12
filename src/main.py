from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}. {hint}")


def write_csv(df: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def command_scrape(config: dict[str, Any]) -> Path:
    import pandas as pd

    from src.config import configured_path
    from src.scrapers.csv_loader import empty_raw_frame, load_manual_csvs, save_raw_csv
    from src.scrapers.instagram_scraper import scrape_instagram
    from src.scrapers.twitter_scraper import scrape_twitter
    from src.scrapers.youtube_scraper import scrape_youtube

    raw_dir = configured_path(config, "raw_dir")
    manual_dir = configured_path(config, "manual_raw_dir")

    frames = []
    scraper_outputs = [
        ("youtube", scrape_youtube(config), raw_dir / "youtube_comments.csv"),
        ("twitter", scrape_twitter(config), raw_dir / "twitter_comments.csv"),
        ("instagram", scrape_instagram(config), raw_dir / "instagram_comments.csv"),
    ]

    for name, df, path in scraper_outputs:
        if df.empty:
            print(f"{name}: no rows collected.")
        else:
            save_raw_csv(df, path)
            print(f"{name}: wrote {len(df)} rows to {path}")
            frames.append(df)

    manual_df = load_manual_csvs(manual_dir)
    if not manual_df.empty:
        print(f"manual: loaded {len(manual_df)} rows from {manual_dir}")
        frames.append(manual_df)

    combined = pd.concat(frames, ignore_index=True) if frames else empty_raw_frame()
    output = raw_dir / "comments_raw.csv"
    save_raw_csv(combined, output)
    print(f"raw combined: wrote {len(combined)} rows to {output}")
    return output


def command_preprocess(config: dict[str, Any]) -> Path:
    import pandas as pd

    from src.config import output_path
    from src.preprocessing import preprocess_dataframe

    input_path = output_path(config, "raw", "comments_raw.csv")
    output = output_path(config, "processed", "comments_clean.csv")
    require_file(input_path, "Run scrape first or add manual CSV files and run scrape.")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No rows to preprocess in {input_path}")
    processed = preprocess_dataframe(df, config)
    write_csv(processed, output)
    print(f"preprocess: wrote {len(processed)} rows to {output}")
    return output


def command_sentiment(config: dict[str, Any]) -> Path:
    import pandas as pd

    from src.config import output_path
    from src.sentiment import analyze_sentiment_dataframe

    input_path = output_path(config, "processed", "comments_clean.csv")
    output = output_path(config, "processed", "comments_sentiment.csv")
    require_file(input_path, "Run preprocess first.")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No rows to analyze in {input_path}")
    analyzed = analyze_sentiment_dataframe(df, config)
    write_csv(analyzed, output)
    print(f"sentiment: wrote {len(analyzed)} rows to {output}")
    return output


def command_topics(config: dict[str, Any]) -> Path:
    from src.config import configured_path, output_path
    from src.topic_modeling import run_topic_modeling

    input_path = output_path(config, "processed", "comments_sentiment.csv")
    processed_dir = configured_path(config, "processed_dir")
    results_dir = configured_path(config, "results_dir")
    require_file(input_path, "Run sentiment first.")

    run_topic_modeling(input_path, config, processed_dir=processed_dir, results_dir=results_dir)
    output = processed_dir / "comments_topics.csv"
    print(f"topics: wrote topic data to {output}")
    return output


def _best_analysis_input(config: dict[str, Any]) -> Path:
    from src.config import output_path

    candidates = [
        output_path(config, "processed", "comments_topics.csv"),
        output_path(config, "processed", "comments_sentiment.csv"),
        output_path(config, "processed", "comments_clean.csv"),
        output_path(config, "raw", "comments_raw.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No analysis input found. Run scrape/preprocess/sentiment/topics first.")


def command_analyze(config: dict[str, Any]) -> Path:
    from src.analysis import run_analysis
    from src.config import configured_path
    from src.visualization import create_visualizations

    input_path = _best_analysis_input(config)
    results_dir = configured_path(config, "results_dir")
    figures_dir = configured_path(config, "figures_dir")
    analyzed = run_analysis(input_path, config, results_dir=results_dir)
    create_visualizations(analyzed, config, figures_dir=figures_dir)
    print(f"analyze: wrote summaries to {results_dir}")
    print(f"analyze: wrote figures to {figures_dir}")
    return results_dir / "comments_analyzed.csv"


def command_all(config: dict[str, Any]) -> None:
    command_scrape(config)
    command_preprocess(config)
    command_sentiment(config)
    command_topics(config)
    command_analyze(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KTI ADEM 3T text mining pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["scrape", "preprocess", "sentiment", "topics", "analyze", "all"]:
        subparsers.add_parser(command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from src.config import load_config

    config = load_config(args.config)

    if args.command == "scrape":
        command_scrape(config)
    elif args.command == "preprocess":
        command_preprocess(config)
    elif args.command == "sentiment":
        command_sentiment(config)
    elif args.command == "topics":
        command_topics(config)
    elif args.command == "analyze":
        command_analyze(config)
    elif args.command == "all":
        command_all(config)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

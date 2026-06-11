from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis import normalize_and_assign_periods


def parse_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(token) for token in value if str(token).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = text.split()
    if not isinstance(parsed, list):
        return []
    return [str(token) for token in parsed if str(token).strip()]


def prepare_period_documents(df: pd.DataFrame, period: str) -> tuple[pd.Index, list[list[str]]]:
    scoped = df[df["period"] == period]
    indexes: list[Any] = []
    documents: list[list[str]] = []
    for index, value in scoped["tokens"].items():
        tokens = parse_tokens(value)
        if tokens:
            indexes.append(index)
            documents.append(tokens)
    return pd.Index(indexes), documents


def train_best_lda(documents: list[list[str]], config: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel, LdaModel
    except ImportError as exc:
        raise RuntimeError("Install gensim and scipy to run topic modeling.") from exc

    settings = config.get("topic_modeling", {})
    min_topics = int(settings.get("min_topics", 3))
    max_topics = int(settings.get("max_topics", 8))
    no_below = int(settings.get("no_below", 2))
    no_above = float(settings.get("no_above", 0.8))

    if len(documents) < 2 or sum(len(document) for document in documents) < 5:
        return None

    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=min(no_below, len(documents)), no_above=no_above)
    if len(dictionary) < min_topics:
        return None

    corpus = [dictionary.doc2bow(document) for document in documents]
    corpus = [bow for bow in corpus if bow]
    if len(corpus) < 2:
        return None

    candidates = range(min_topics, min(max_topics, len(dictionary)) + 1)
    best: dict[str, Any] | None = None
    coherence_rows = []
    for num_topics in candidates:
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=int(settings.get("random_state", 42)),
            passes=int(settings.get("passes", 10)),
            per_word_topics=False,
        )
        coherence_model = CoherenceModel(
            model=model,
            texts=documents,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence = float(coherence_model.get_coherence())
        coherence_rows.append({"num_topics": num_topics, "coherence": coherence})
        if best is None or coherence > best["coherence"]:
            best = {
                "model": model,
                "dictionary": dictionary,
                "corpus": [dictionary.doc2bow(document) for document in documents],
                "coherence": coherence,
                "num_topics": num_topics,
                "coherence_rows": coherence_rows.copy(),
            }

    return best


def extract_topic_keywords(model: Any, period: str, num_topics: int, coherence: float, topn: int) -> list[dict[str, Any]]:
    rows = []
    for topic_id, words in model.show_topics(num_topics=num_topics, num_words=topn, formatted=False):
        rows.append(
            {
                "period": period,
                "num_topics": num_topics,
                "coherence": round(coherence, 4),
                "topic_id": topic_id,
                "keywords": ", ".join(word for word, _weight in words),
            }
        )
    return rows


def assign_dominant_topics(df: pd.DataFrame, indexes: pd.Index, model_bundle: dict[str, Any]) -> pd.DataFrame:
    output = df.copy()
    model = model_bundle["model"]
    corpus = model_bundle["corpus"]
    for index, bow in zip(indexes, corpus):
        topic_probs = model.get_document_topics(bow)
        if topic_probs:
            output.at[index, "dominant_topic"] = int(max(topic_probs, key=lambda item: item[1])[0])
    return output


def run_topic_modeling(input_path: str | Path, config: dict[str, Any], processed_dir: str | Path, results_dir: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    processed_dir = Path(processed_dir)
    results_dir = Path(results_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "tokens" not in df.columns:
        raise ValueError(f"Input file missing tokens column: {input_path}")
    if "created_at_utc" not in df.columns or "period" not in df.columns:
        df = normalize_and_assign_periods(df, config, invalid_output_path=results_dir / "invalid_dates.csv")

    output = df.copy()
    output["dominant_topic"] = pd.NA
    topic_rows: list[dict[str, Any]] = []
    coherence_rows: list[dict[str, Any]] = []

    topn = int(config.get("topic_modeling", {}).get("topn_words", 10))
    for period in config.get("periods", {}):
        indexes, documents = prepare_period_documents(output, period)
        bundle = train_best_lda(documents, config)
        if bundle is None:
            print(f"Topic modeling skipped for {period}: not enough usable tokens.")
            pd.DataFrame(columns=["period", "num_topics", "coherence", "topic_id", "keywords"]).to_csv(
                results_dir / f"topics_{period}.csv",
                index=False,
            )
            continue

        output = assign_dominant_topics(output, indexes, bundle)
        period_topics = extract_topic_keywords(
            bundle["model"],
            period,
            int(bundle["num_topics"]),
            float(bundle["coherence"]),
            topn,
        )
        topic_rows.extend(period_topics)
        pd.DataFrame(period_topics).to_csv(results_dir / f"topics_{period}.csv", index=False)

        for row in bundle["coherence_rows"]:
            coherence_rows.append({"period": period, **row})

    pd.DataFrame(topic_rows).to_csv(results_dir / "topics_all_periods.csv", index=False)
    pd.DataFrame(coherence_rows).to_csv(results_dir / "topic_coherence.csv", index=False)
    output.to_csv(processed_dir / "comments_topics.csv", index=False)
    return output

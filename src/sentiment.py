from __future__ import annotations

from typing import Any

import pandas as pd


class SentimentAnalyzer:
    def __init__(
        self,
        model_name: str,
        label_map: dict[str, str],
        batch_size: int = 16,
        max_length: int = 256,
        classifier: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.label_map = label_map
        self.batch_size = batch_size
        self.max_length = max_length
        self.classifier = classifier

    def load(self) -> None:
        if self.classifier is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install transformers and torch to run sentiment analysis.") from exc

        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Hugging Face sentiment model. "
                f"Check internet/cache and model name: {self.model_name}"
            ) from exc

    def normalize_label(self, raw_label: str) -> str:
        if raw_label in self.label_map:
            return self.label_map[raw_label]
        lowered = raw_label.lower()
        if lowered in {"positive", "neutral", "negative"}:
            return lowered
        return lowered

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        self.load()
        assert self.classifier is not None

        results: list[dict[str, Any]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            non_empty_indexes = [index for index, text in enumerate(batch) if text.strip()]
            non_empty_texts = [batch[index] for index in non_empty_indexes]

            batch_results = [{"label": "neutral", "score": 0.0} for _ in batch]
            if non_empty_texts:
                predictions = self.classifier(
                    non_empty_texts,
                    truncation=True,
                    max_length=self.max_length,
                )
                for local_index, prediction in zip(non_empty_indexes, predictions):
                    raw_label = str(prediction.get("label", "neutral"))
                    batch_results[local_index] = {
                        "label": self.normalize_label(raw_label),
                        "score": float(prediction.get("score", 0.0)),
                    }

            results.extend(batch_results)
        return results


def build_analyzer(config: dict[str, Any], classifier: Any | None = None) -> SentimentAnalyzer:
    settings = config.get("sentiment", {})
    return SentimentAnalyzer(
        model_name=settings.get("model_name", "mdhugol/indonesia-bert-sentiment-classification"),
        label_map=settings.get(
            "label_map",
            {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"},
        ),
        batch_size=int(settings.get("batch_size", 16)),
        max_length=int(settings.get("max_length", 256)),
        classifier=classifier,
    )


def analyze_sentiment_dataframe(
    df: pd.DataFrame,
    config: dict[str, Any],
    classifier: Any | None = None,
) -> pd.DataFrame:
    text_column = "clean_text" if "clean_text" in df.columns else "text"
    analyzer = build_analyzer(config, classifier=classifier)
    texts = df[text_column].fillna("").astype(str).tolist()
    predictions = analyzer.predict(texts)

    output = df.copy()
    output["sentiment_label"] = [item["label"] for item in predictions]
    output["sentiment_score"] = [item["score"] for item in predictions]
    return output


def analyze_sentiment_file(input_path: str, output_path: str, config: dict[str, Any]) -> str:
    df = pd.read_csv(input_path)
    output = analyze_sentiment_dataframe(df, config)
    output.to_csv(output_path, index=False)
    return output_path

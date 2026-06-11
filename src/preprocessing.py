from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import pandas as pd


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
SPACE_RE = re.compile(r"\s+")

FALLBACK_STOPWORDS = {
    "ada",
    "adalah",
    "akan",
    "aku",
    "anda",
    "atau",
    "dalam",
    "dan",
    "dari",
    "dengan",
    "di",
    "ini",
    "itu",
    "ke",
    "kita",
    "untuk",
    "yang",
}


class IdentityStemmer:
    def stem(self, text: str) -> str:
        return text


def clean_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    value = str(text).lower()
    try:
        import emoji

        value = emoji.replace_emoji(value, replace=" ")
    except ImportError:
        # Regex below removes most emoji/non alphabetic characters even without emoji package.
        pass

    value = URL_RE.sub(" ", value)
    value = MENTION_RE.sub(" ", value)
    value = HASHTAG_RE.sub(r"\1", value)
    value = NON_ALPHA_RE.sub(" ", value)
    value = SPACE_RE.sub(" ", value).strip()
    return value


def tokenize(text: str) -> list[str]:
    return [token for token in text.split() if token]


def load_stopwords(config: dict[str, Any]) -> set[str]:
    stopwords = set(FALLBACK_STOPWORDS)
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

        stopwords.update(StopWordRemoverFactory().get_stop_words())
    except ImportError:
        try:
            from nltk.corpus import stopwords as nltk_stopwords

            stopwords.update(nltk_stopwords.words("indonesian"))
        except Exception:
            print("Sastrawi/NLTK stopwords unavailable. Using fallback stopword list.")

    stopwords.update(config.get("preprocessing", {}).get("custom_stopwords", []))
    return {clean_text(word) for word in stopwords if clean_text(word)}


def load_stemmer() -> Any:
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        return StemmerFactory().create_stemmer()
    except ImportError:
        print("Sastrawi unavailable. Stemming disabled until Sastrawi is installed.")
        return IdentityStemmer()


@lru_cache(maxsize=20000)
def _stem_cached(token: str, stemmer_name: str) -> str:
    # This function is only used for identity fallback. Real Sastrawi stemmer is not hashable.
    return token


def preprocess_text(text: Any, stopwords: set[str], stemmer: Any) -> dict[str, Any]:
    cleaned = clean_text(text)
    raw_tokens = tokenize(cleaned)
    filtered = [token for token in raw_tokens if token not in stopwords and len(token) > 1]

    if isinstance(stemmer, IdentityStemmer):
        tokens = [_stem_cached(token, "identity") for token in filtered]
    else:
        tokens = [stemmer.stem(token) for token in filtered]

    tokens = [token for token in tokens if token and token not in stopwords and len(token) > 1]
    return {
        "clean_text": " ".join(tokens),
        "tokens": tokens,
    }


def preprocess_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    stopwords = load_stopwords(config)
    stemmer = load_stemmer()
    output = df.copy()

    clean_values: list[str] = []
    token_values: list[str] = []
    for text in output["text"].tolist():
        result = preprocess_text(text, stopwords, stemmer)
        clean_values.append(result["clean_text"])
        token_values.append(json.dumps(result["tokens"], ensure_ascii=False))

    output["clean_text"] = clean_values
    output["tokens"] = token_values
    return output


def preprocess_file(input_path: str, output_path: str, config: dict[str, Any]) -> str:
    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError(f"Input file missing text column: {input_path}")
    processed = preprocess_dataframe(df, config)
    processed.to_csv(output_path, index=False)
    return output_path

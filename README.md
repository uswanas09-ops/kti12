# KTI ADEM 3T Text Mining

Program Python modular untuk penelitian:

"Analisis Opini Publik terhadap Implementasi Beasiswa ADEM 3T: Studi Text Mining Komentar Media Sosial Kemendikdasmen."

Pipeline mencakup scraping, preprocessing teks bahasa Indonesia, sentiment analysis dengan model IndoBERT/Transformer, topic modeling LDA, analisis perbandingan periode, dan visualisasi.

## Instalasi

Gunakan virtual environment, lalu install dependency:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Jika memakai GPU, install `torch` sesuai instruksi resmi PyTorch untuk versi CUDA yang tersedia.

## Konfigurasi API

Salin `.env.example` menjadi `.env`, lalu isi token yang tersedia:

```text
YOUTUBE_API_KEY=
X_BEARER_TOKEN=
APIFY_TOKEN=
```

Token kosong tidak menghentikan pipeline. Scraper terkait akan dilewati, dan data tetap bisa dimasukkan lewat CSV manual.

## Format CSV Manual

Letakkan file CSV manual di:

```text
data/raw/manual/
```

Kolom minimal:

```csv
source,username,text,created_at
manual,user1,"Beasiswa ADEM sangat membantu","2025-06-01T08:00:00+07:00"
manual,user2,"Informasinya kurang jelas","2024-05-10T10:00:00Z"
```

Kolom tambahan yang didukung:

```text
platform_id,url,query
```

## Periode Penelitian

Program menormalisasi semua timestamp dengan:

```python
pandas.to_datetime(created_at, utc=True, errors="coerce")
```

Ini penting agar tanggal dari YouTube, X/Twitter, Instagram, dan CSV manual bisa dibandingkan dalam satu timezone.

Periode:

- Periode 1: `2023-06-01T00:00:00Z <= created_at < 2025-06-01T00:00:00Z`
- Periode 2: `2025-06-01T00:00:00Z <= created_at < 2026-06-01T00:00:00Z`

Data tepat pada `2025-06-01T00:00:00Z` masuk Periode 2.

## Cara Menjalankan

Jalankan dari root proyek.

```bash
python -m src.main scrape
python -m src.main preprocess
python -m src.main sentiment
python -m src.main topics
python -m src.main analyze
```

Atau jalankan semua tahap:

```bash
python -m src.main all
```

## Output

Raw dan processed data:

- `data/raw/comments_raw.csv`
- `data/processed/comments_clean.csv`
- `data/processed/comments_sentiment.csv`
- `data/processed/comments_topics.csv`

Ringkasan:

- `data/results/comments_analyzed.csv`
- `data/results/sentiment_by_period.csv`
- `data/results/sentiment_by_platform.csv`
- `data/results/topic_distribution_by_period.csv`
- `data/results/invalid_dates.csv`
- `data/results/topics_period_1.csv`
- `data/results/topics_period_2.csv`
- `data/results/topic_coherence.csv`

Visualisasi:

- `data/figures/sentiment_period_comparison.png`
- `data/figures/wordcloud_period_1.png`
- `data/figures/wordcloud_period_2.png`
- `data/figures/topic_distribution_period_1.png`
- `data/figures/topic_distribution_period_2.png`

## Catatan Scraping

- YouTube memakai YouTube Data API v3.
- X/Twitter memakai Tweepy API v2. Akses data lama sering membutuhkan paket API tertentu.
- Instagram memakai Apify jika token tersedia. Scraping Instagram sering dibatasi platform.
- Untuk penelitian yang harus reproducible, simpan dataset mentah final sebagai CSV dan cantumkan tanggal pengambilan data.

## Testing

```bash
pytest -q
```

Test tidak membutuhkan API key dan tidak mengunduh model Hugging Face.

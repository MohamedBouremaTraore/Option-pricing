import time
import logging
from datetime import date, timedelta
from typing import List, Tuple, Union
import pandas as pd
import requests
from urllib.parse import quote
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text_vader(text: Union[str, float, None]) -> str:
    """
    Clean text for VADER sentiment analysis.

    Steps:
    - lowercase
    - remove URLs
    - remove tickers / FX pairs
    - remove special characters
    - normalize whitespace
    """

    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove stock tickers ($AAPL) and FX pairs (EUR/USD)
    text = re.sub(r"\$\w+|\b[A-Z]{2,}/[A-Z]{2,}\b", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_vader_score(text: Union[str, float, None]) -> float:
    """
    Compute VADER sentiment score (compound).

    Returns:
        float: sentiment score in [-1, 1]
    """

    if not isinstance(text, str) or not text.strip():
        return 0.0

    return sia.polarity_scores(text)["compound"]


def build_gdelt_url(query: str, start: str, end: str) -> str:
    """Build GDELT API URL."""
    encoded_query = quote(query)
    return (
        f"https://api.gdeltproject.org/api/v2/doc/doc?"
        f"query={encoded_query}&mode=ArtList&maxrecords=100&format=json&"
        f"startdatetime={start}000000&enddatetime={end}235959"
    )


def fetch_articles(url: str, retries: int, pause: float) -> List[dict]:
    """Fetch articles from GDELT API with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])

            logger.info(f"{len(articles)} articles fetched")
            return articles

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(pause)

    return []


def preprocess_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich articles DataFrame."""
    if df.empty:
        return df

    # Datetime processing
    df["datetime"] = pd.to_datetime(df["seendate"], errors="coerce")
    df["date"] = df["datetime"].dt.date

    # Keep only English articles
    df = df[df["language"] == "English"]

    # Text preprocessing
    df["clean_title"] = df["title"].apply(clean_text_vader)

    # Sentiment
    df["sentiment"] = df["clean_title"].apply(get_vader_score)

    df["label"] = df["sentiment"].apply(
        lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
    )

    return df


def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment metrics by date."""
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("date").agg(
        n_articles=("clean_title", "count"),
        pct_positive=("label", lambda x: (x == "positive").mean() * 100),
        pct_negative=("label", lambda x: (x == "negative").mean() * 100),
        sentiment_mean=("sentiment", "mean"),
        sentiment_min=("sentiment", "min"),
        sentiment_max=("sentiment", "max"),
        sentiment_median=("sentiment", "median"),
        sentiment_std=("sentiment", "std"),
    ).reset_index()

    return agg.sort_values("date")


def fetch_daily_sentiment(
    keywords: List[str],
    start_date: date = date.today(),
    days: int = 7,
    pause: float = 1.0,
    retries: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch news from GDELT, compute sentiment, and aggregate daily statistics.

    Args:
        keywords: List of keywords (e.g., ["Apple", "AAPL"])
        start_date: Reference date
        days: Number of past days to fetch
        pause: Delay between API calls
        retries: Number of retries per request

    Returns:
        Tuple (aggregated_df, articles_df)
    """
    query = "(" + " OR ".join(keywords) + ")"
    all_articles = []

    for i in range(days):
        day_start = start_date - timedelta(days=i + 1)
        day_end = start_date - timedelta(days=i)

        # Skip weekends
        if day_start.weekday() > 4 or day_end.weekday() > 4:
            logger.info(f"Skipping weekend: {day_start} → {day_end}")
            continue

        start_str = day_start.strftime("%Y%m%d")
        end_str = day_end.strftime("%Y%m%d")

        logger.info(f"Fetching {start_str} → {end_str}")

        url = build_gdelt_url(query, start_str, end_str)
        articles = fetch_articles(url, retries=retries, pause=pause)

        all_articles.extend(articles)

    if not all_articles:
        logger.warning("No articles found.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(all_articles)

    # Preprocess
    df = preprocess_articles(df)

    # Aggregate
    agg = aggregate_sentiment(df)

    return agg, df
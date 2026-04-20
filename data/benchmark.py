"""
Benchmark: old vs new architecture at different data scales.

Scales tested: 5K, 50K, 100K, 500K, 1M posts.

Old approach: SELECT * FROM posts → iterate all rows in Go → keyword/trend compute
New approach: SELECT posts LEFT JOIN post_analyses → trend from daily_platform_stats

Both are simulated directly against SQLite (no HTTP overhead) so we measure pure
DB query + computation time.

Usage:
    python3 benchmark.py
"""

import sqlite3
import csv
import time
import random
import os
import statistics

DB_PATH = "/tmp/listenai_bench.db"
CSV_PATH = os.path.join(os.path.dirname(__file__), "posts.csv")

PLATFORMS = ["twitter", "facebook", "instagram", "threads", "ptt"]
SENTIMENTS = ["positive", "neutral", "negative"]
FROM_DATE = "2024-01-01"
TO_DATE   = "2026-12-31"
SCALES    = [5_000, 50_000, 100_000, 500_000, 1_000_000]
REPEATS   = 5   # repeat each query to get stable timing

# Measured from evaluate.py: Lexicon-based = 0.0546 ms/sample
NLP_LATENCY_MS_PER_SAMPLE = 0.0546


# ─── Setup ────────────────────────────────────────────────────────────────────

def load_template_contents():
    contents = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            contents.append(row["content"])
    return contents


def build_db(n: int, contents: list[str]) -> sqlite3.Connection:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            author   TEXT NOT NULL,
            content  TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX idx_posts_created_at ON posts(created_at);")
    conn.execute("""
        CREATE TABLE post_analyses (
            post_id         INTEGER PRIMARY KEY,
            sentiment       TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            analyzed_at     TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE daily_platform_stats (
            date           TEXT NOT NULL,
            platform       TEXT NOT NULL,
            post_count     INTEGER NOT NULL DEFAULT 0,
            positive_count INTEGER NOT NULL DEFAULT 0,
            neutral_count  INTEGER NOT NULL DEFAULT 0,
            negative_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (date, platform)
        )
    """)

    print(f"  Inserting {n:,} rows...", end="", flush=True)
    t0 = time.perf_counter()

    batch_posts = []
    batch_analyses = []
    daily = {}  # (date, platform) → [total, pos, neu, neg]

    rng = random.Random(42)
    num_contents = len(contents)

    for i in range(n):
        platform = rng.choice(PLATFORMS)
        # Spread dates evenly over ~2 years (730 days)
        day_offset = i % 730
        date = f"202{4 + day_offset // 365}-{((day_offset % 365) // 30) + 1:02d}-{(day_offset % 30) + 1:02d}"
        created_at = f"{date}T00:00:00Z"
        content = contents[i % num_contents]
        sentiment = rng.choice(SENTIMENTS)
        score = round(rng.uniform(0.55, 0.99), 4)

        batch_posts.append((platform, f"user_{i}", content, created_at))
        batch_analyses.append((i + 1, sentiment, score, created_at))

        key = (date, platform)
        if key not in daily:
            daily[key] = [0, 0, 0, 0]
        daily[key][0] += 1
        if sentiment == "positive":
            daily[key][1] += 1
        elif sentiment == "neutral":
            daily[key][2] += 1
        else:
            daily[key][3] += 1

        if len(batch_posts) == 10_000:
            conn.executemany(
                "INSERT INTO posts (platform, author, content, created_at) VALUES (?,?,?,?)",
                batch_posts,
            )
            conn.executemany(
                "INSERT INTO post_analyses (post_id, sentiment, sentiment_score, analyzed_at) VALUES (?,?,?,?)",
                batch_analyses,
            )
            conn.commit()
            batch_posts.clear()
            batch_analyses.clear()

    if batch_posts:
        conn.executemany(
            "INSERT INTO posts (platform, author, content, created_at) VALUES (?,?,?,?)",
            batch_posts,
        )
        conn.executemany(
            "INSERT INTO post_analyses (post_id, sentiment, sentiment_score, analyzed_at) VALUES (?,?,?,?)",
            batch_analyses,
        )
        conn.commit()

    daily_rows = [
        (k[0], k[1], v[0], v[1], v[2], v[3]) for k, v in daily.items()
    ]
    conn.executemany(
        "INSERT INTO daily_platform_stats VALUES (?,?,?,?,?,?)", daily_rows
    )
    conn.commit()

    elapsed = time.perf_counter() - t0
    print(f" done in {elapsed:.1f}s")
    return conn


# ─── Benchmark queries ────────────────────────────────────────────────────────

def bench_old_stats(conn: sqlite3.Connection) -> tuple[float, float, float]:
    """
    Old approach (per dashboard request):
      1. DB: fetch up to 500 posts (no sentiment cached)
      2. DB: build trends by iterating fetched posts
      3. NLP: run sentiment inference on ALL 500 posts (simulated via measured latency)

    Returns (db_ms, nlp_ms, total_ms).
    """
    db_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        rows = conn.execute(
            """SELECT id, platform, author, content, created_at
               FROM posts
               WHERE date(created_at) BETWEEN date(?) AND date(?)
               ORDER BY datetime(created_at) DESC
               LIMIT 500""",
            (FROM_DATE, TO_DATE),
        ).fetchall()
        # Go-side trend building — iterate fetched rows in memory
        counts: dict[str, int] = {}
        for row in rows:
            date = row[4][:10] if len(row[4]) >= 10 else ""
            counts[date] = counts.get(date, 0) + 1
        db_times.append(time.perf_counter() - t0)

    db_ms = statistics.mean(db_times) * 1000
    # NLP must be called for every fetched post (no cache)
    nlp_ms = len(rows) * NLP_LATENCY_MS_PER_SAMPLE
    return db_ms, nlp_ms, db_ms + nlp_ms


def bench_new_stats(conn: sqlite3.Connection) -> tuple[float, float, float]:
    """
    New approach (per dashboard request):
      1. DB: fetch posts LEFT JOIN post_analyses (sentiment already stored)
      2. DB: trends from daily_platform_stats (O(days) aggregation)
      3. NLP: 0 calls (all posts have cached analysis in steady state)

    Returns (db_ms, nlp_ms, total_ms).
    """
    db_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        rows = conn.execute(
            """SELECT p.id, p.platform, p.author, p.content, p.created_at,
                      COALESCE(a.sentiment,''), COALESCE(a.sentiment_score,0)
               FROM posts p
               LEFT JOIN post_analyses a ON p.id = a.post_id
               WHERE date(p.created_at) BETWEEN date(?) AND date(?)
               ORDER BY datetime(p.created_at) DESC
               LIMIT 500""",
            (FROM_DATE, TO_DATE),
        ).fetchall()
        # Trends from materialized table — tiny result set regardless of N
        conn.execute(
            """SELECT date, SUM(post_count)
               FROM daily_platform_stats
               WHERE date BETWEEN ? AND ?
               GROUP BY date ORDER BY date""",
            (FROM_DATE, TO_DATE),
        ).fetchall()
        db_times.append(time.perf_counter() - t0)

    db_ms = statistics.mean(db_times) * 1000
    nlp_ms = 0.0   # steady state: all posts have cached sentiment
    return db_ms, nlp_ms, db_ms + nlp_ms


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    contents = load_template_contents()
    print(f"Template: {len(contents)} unique post contents loaded.\n")

    results = []

    for scale in SCALES:
        print(f"{'─'*50}")
        print(f"Scale: {scale:,} posts")
        conn = build_db(scale, contents)

        old_db, old_nlp, old_total = bench_old_stats(conn)
        new_db, new_nlp, new_total = bench_new_stats(conn)
        speedup = old_total / new_total if new_total > 0 else float("inf")
        results.append((scale, old_db, old_nlp, old_total, new_db, new_nlp, new_total, speedup))

        conn.close()
        os.remove(DB_PATH)

        print(f"  OLD → DB: {old_db:7.2f} ms | NLP: {old_nlp:7.2f} ms | Total: {old_total:8.2f} ms")
        print(f"  NEW → DB: {new_db:7.2f} ms | NLP: {new_nlp:7.2f} ms | Total: {new_total:8.2f} ms")
        print(f"  End-to-end speedup: {speedup:.1f}×")

    print(f"\n{'='*72}")
    print(f"  {'Scale':>10}  {'Old DB':>8}  {'Old NLP':>8}  {'Old Tot':>8}  {'New DB':>8}  {'New NLP':>8}  {'New Tot':>8}  {'Speedup':>8}")
    print(f"  {'-'*70}")
    for row in results:
        scale, old_db, old_nlp, old_tot, new_db, new_nlp, new_tot, spd = row
        print(f"  {scale:>10,}  {old_db:>8.1f}  {old_nlp:>8.1f}  {old_tot:>8.1f}  {new_db:>8.1f}  {new_nlp:>8.1f}  {new_tot:>8.1f}  {spd:>7.1f}×")
    print(f"\n  NLP latency assumed: {NLP_LATENCY_MS_PER_SAMPLE} ms/sample (measured from Lexicon-based classifier)")
    print(f"  New approach NLP=0 assumes steady state (all posts pre-analyzed at write time)\n")

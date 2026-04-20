package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"

	_ "modernc.org/sqlite"
)

// ─── Data structures ──────────────────────────────────────────────────────────

type Post struct {
	ID             int     `json:"id"`
	Platform       string  `json:"platform"`
	Author         string  `json:"author"`
	Content        string  `json:"content"`
	CreatedAt      string  `json:"created_at"`
	Sentiment      string  `json:"sentiment,omitempty"`
	SentimentScore float64 `json:"sentiment_score,omitempty"`
}

type StatsRequest struct {
	IncludeKeywords []string `json:"include_keywords"`
	ExcludeKeywords []string `json:"exclude_keywords"`
	FromDate        string   `json:"from_date"`
	ToDate          string   `json:"to_date"`
	ExampleLimit    int      `json:"example_limit"`
	PostLimit       int      `json:"post_limit"`
}

type KeywordCount struct {
	Keyword string `json:"keyword"`
	Count   int    `json:"count"`
}

type TrendPoint struct {
	Date  string `json:"date"`
	Count int    `json:"count"`
}

type StatsResponse struct {
	MentionCount int            `json:"mention_count"`
	TopKeywords  []KeywordCount `json:"top_keywords"`
	Trends       []TrendPoint   `json:"trends"`
	ExamplePosts []Post         `json:"example_posts"`
	Posts        []Post         `json:"posts"`
}

type InsertPostRequest struct {
	Platform  string `json:"platform"`
	Author    string `json:"author"`
	Content   string `json:"content"`
	CreatedAt string `json:"created_at"`
}

type InsertPostResponse struct {
	ID int `json:"id"`
}

// AnalysisRequest stores pre-computed NLP results for a single post.
// Called by Gateway after it inserts a post and receives the NLP result.
type AnalysisRequest struct {
	PostID         int     `json:"post_id"`
	Sentiment      string  `json:"sentiment"`
	SentimentScore float64 `json:"sentiment_score"`
}

// ─── Stop words & helpers ─────────────────────────────────────────────────────

var stopWords = map[string]bool{
	"the": true, "a": true, "an": true, "and": true, "or": true, "to": true,
	"of": true, "in": true, "on": true, "for": true, "with": true, "is": true,
	"are": true, "it": true, "this": true, "that": true, "my": true, "our": true,
	"your": true, "but": true, "from": true, "at": true, "was": true,
}

func parseDateRange(fromDate, toDate string) (string, string, error) {
	layout := "2006-01-02"
	now := time.Now()
	if fromDate == "" {
		fromDate = now.AddDate(0, 0, -30).Format(layout)
	}
	if toDate == "" {
		toDate = now.Format(layout)
	}
	if _, err := time.Parse(layout, fromDate); err != nil {
		return "", "", fmt.Errorf("invalid from_date: %w", err)
	}
	if _, err := time.Parse(layout, toDate); err != nil {
		return "", "", fmt.Errorf("invalid to_date: %w", err)
	}
	return fromDate, toDate, nil
}

func containsAny(text string, words []string) bool {
	if len(words) == 0 {
		return true
	}
	l := strings.ToLower(text)
	for _, w := range words {
		w = strings.TrimSpace(strings.ToLower(w))
		if w == "" {
			continue
		}
		if strings.Contains(l, w) {
			return true
		}
	}
	return false
}

func containsNone(text string, words []string) bool {
	l := strings.ToLower(text)
	for _, w := range words {
		w = strings.TrimSpace(strings.ToLower(w))
		if w == "" {
			continue
		}
		if strings.Contains(l, w) {
			return false
		}
	}
	return true
}

func extractTopKeywords(posts []Post, include, exclude []string, topN int) []KeywordCount {
	re := regexp.MustCompile(`[a-zA-Z']+|[\p{Han}]+`)
	freq := map[string]int{}
	excludedMap := map[string]bool{}
	for _, w := range exclude {
		w = strings.ToLower(strings.TrimSpace(w))
		if w != "" {
			excludedMap[w] = true
		}
	}
	for _, post := range posts {
		tokens := extractKeywordTokens(post.Content, re)
		for _, t := range tokens {
			if isTooShortKeyword(t) || stopWords[t] || excludedMap[t] {
				continue
			}
			freq[t]++
		}
	}
	items := make([]KeywordCount, 0, len(freq))
	for k, c := range freq {
		items = append(items, KeywordCount{Keyword: k, Count: c})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].Count == items[j].Count {
			return items[i].Keyword < items[j].Keyword
		}
		return items[i].Count > items[j].Count
	})
	if len(items) > topN {
		items = items[:topN]
	}
	return items
}

func extractKeywordTokens(content string, re *regexp.Regexp) []string {
	matches := re.FindAllString(strings.ToLower(content), -1)
	tokens := make([]string, 0, len(matches)*2)
	for _, m := range matches {
		if isHanOnly(m) {
			tokens = append(tokens, hanBigrams(m)...)
			continue
		}
		tokens = append(tokens, m)
	}
	return tokens
}

func isHanOnly(text string) bool {
	if text == "" {
		return false
	}
	for _, r := range text {
		if !unicode.Is(unicode.Han, r) {
			return false
		}
	}
	return true
}

func hanBigrams(text string) []string {
	runes := []rune(text)
	if len(runes) < 2 {
		return nil
	}
	if len(runes) == 2 {
		return []string{text}
	}
	bigrams := make([]string, 0, len(runes)-1)
	for i := 0; i < len(runes)-1; i++ {
		bigrams = append(bigrams, string(runes[i:i+2]))
	}
	return bigrams
}

func isTooShortKeyword(token string) bool {
	if isHanOnly(token) {
		return len([]rune(token)) < 2
	}
	return len(token) <= 2
}

func buildTrends(posts []Post) []TrendPoint {
	counts := map[string]int{}
	for _, p := range posts {
		if len(p.CreatedAt) >= 10 {
			counts[p.CreatedAt[:10]]++
		}
	}
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	trends := make([]TrendPoint, 0, len(keys))
	for _, d := range keys {
		trends = append(trends, TrendPoint{Date: d, Count: counts[d]})
	}
	return trends
}

// buildTrendsFromCache reads pre-aggregated daily_platform_stats instead of
// scanning all posts — O(days) instead of O(posts).
func buildTrendsFromCache(db *sql.DB, fromDate, toDate string) ([]TrendPoint, error) {
	rows, err := db.Query(
		`SELECT date, SUM(post_count) AS cnt
		 FROM daily_platform_stats
		 WHERE date BETWEEN ? AND ?
		 GROUP BY date
		 ORDER BY date`,
		fromDate, toDate,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var trends []TrendPoint
	for rows.Next() {
		var tp TrendPoint
		if err := rows.Scan(&tp.Date, &tp.Count); err != nil {
			return nil, err
		}
		trends = append(trends, tp)
	}
	return trends, nil
}

// ─── Database helpers ─────────────────────────────────────────────────────────

// fetchFilteredPosts returns posts enriched with pre-computed sentiment via
// a LEFT JOIN on post_analyses. Posts without an analysis entry have empty
// Sentiment (Gateway falls back to live NLP for those).
func fetchFilteredPosts(db *sql.DB, req StatsRequest) ([]Post, error) {
	fromDate, toDate, err := parseDateRange(req.FromDate, req.ToDate)
	if err != nil {
		return nil, err
	}

	rows, err := db.Query(
		`SELECT p.id, p.platform, p.author, p.content, p.created_at,
		        COALESCE(a.sentiment, ''), COALESCE(a.sentiment_score, 0.0)
		 FROM posts p
		 LEFT JOIN post_analyses a ON p.id = a.post_id
		 WHERE date(p.created_at) BETWEEN date(?) AND date(?)
		 ORDER BY datetime(p.created_at) DESC`,
		fromDate, toDate,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	posts := []Post{}
	for rows.Next() {
		var p Post
		if err := rows.Scan(&p.ID, &p.Platform, &p.Author, &p.Content,
			&p.CreatedAt, &p.Sentiment, &p.SentimentScore); err != nil {
			return nil, err
		}
		if !containsAny(p.Content, req.IncludeKeywords) {
			continue
		}
		if !containsNone(p.Content, req.ExcludeKeywords) {
			continue
		}
		posts = append(posts, p)
	}

	if req.PostLimit <= 0 {
		req.PostLimit = 500
	}
	if len(posts) > req.PostLimit {
		posts = posts[:req.PostLimit]
	}
	return posts, nil
}

func setupDatabase(db *sql.DB) error {
	_, err := db.Exec(`PRAGMA journal_mode=WAL;`)
	if err != nil {
		return err
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS posts (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			platform   TEXT NOT NULL,
			author     TEXT NOT NULL,
			content    TEXT NOT NULL,
			created_at TEXT NOT NULL
		);

		-- Index for date-range queries (the most common filter)
		CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);

		-- Pre-computed per-post NLP results (written once at insert time)
		CREATE TABLE IF NOT EXISTS post_analyses (
			post_id         INTEGER PRIMARY KEY,
			sentiment       TEXT    NOT NULL,
			sentiment_score REAL    NOT NULL,
			analyzed_at     TEXT    NOT NULL,
			FOREIGN KEY (post_id) REFERENCES posts(id)
		);

		-- Materialized daily aggregate — enables O(days) trend queries
		CREATE TABLE IF NOT EXISTS daily_platform_stats (
			date           TEXT NOT NULL,
			platform       TEXT NOT NULL,
			post_count     INTEGER NOT NULL DEFAULT 0,
			positive_count INTEGER NOT NULL DEFAULT 0,
			neutral_count  INTEGER NOT NULL DEFAULT 0,
			negative_count INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (date, platform)
		);
	`)
	return err
}

func writeJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func normalizeCreatedAt(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Now().UTC().Format(time.RFC3339), nil
	}
	t, err := time.Parse(time.RFC3339, value)
	if err != nil {
		return "", fmt.Errorf("created_at must be RFC3339 format")
	}
	return t.UTC().Format(time.RFC3339), nil
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	port := os.Getenv("STAT_PORT")
	if port == "" {
		port = "8002"
	}
	sqlitePath := os.Getenv("SQLITE_PATH")
	if sqlitePath == "" {
		sqlitePath = "./listenai.db"
	}

	db, err := sql.Open("sqlite", sqlitePath)
	if err != nil {
		log.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	if err := setupDatabase(db); err != nil {
		log.Fatalf("failed to setup database: %v", err)
	}

	// ── GET /health ──────────────────────────────────────────────────────────
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "service": "stat", "port": port})
	})

	// ── POST /stats ──────────────────────────────────────────────────────────
	// Returns posts enriched with pre-computed sentiment (no NLP call needed).
	// Trends are read from daily_platform_stats when no keyword filter is active
	// (O(days) query), otherwise built from the filtered post list.
	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
			return
		}

		var req StatsRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
			return
		}

		posts, err := fetchFilteredPosts(db, req)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}

		exampleLimit := req.ExampleLimit
		if exampleLimit <= 0 {
			exampleLimit = 5
		}
		examplePosts := posts
		if len(examplePosts) > exampleLimit {
			examplePosts = examplePosts[:exampleLimit]
		}

		// Use materialized trend table when there's no keyword filter (fast path).
		var trends []TrendPoint
		if len(req.IncludeKeywords) == 0 && len(req.ExcludeKeywords) == 0 {
			fromDate, toDate, _ := parseDateRange(req.FromDate, req.ToDate)
			trends, _ = buildTrendsFromCache(db, fromDate, toDate)
		}
		if len(trends) == 0 {
			trends = buildTrends(posts)
		}

		resp := StatsResponse{
			MentionCount: len(posts),
			TopKeywords:  extractTopKeywords(posts, req.IncludeKeywords, req.ExcludeKeywords, 10),
			Trends:       trends,
			ExamplePosts: examplePosts,
			Posts:        posts,
		}
		writeJSON(w, http.StatusOK, resp)
	})

	// ── POST /posts ──────────────────────────────────────────────────────────
	http.HandleFunc("/posts", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
			return
		}

		var req InsertPostRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
			return
		}

		req.Platform = strings.TrimSpace(req.Platform)
		req.Author = strings.TrimSpace(req.Author)
		req.Content = strings.TrimSpace(req.Content)

		if req.Platform == "" || req.Author == "" || req.Content == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "platform, author, and content are required"})
			return
		}

		createdAt, err := normalizeCreatedAt(req.CreatedAt)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}

		result, err := db.Exec(
			`INSERT INTO posts (platform, author, content, created_at) VALUES (?, ?, ?, ?)`,
			req.Platform, req.Author, req.Content, createdAt,
		)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to insert post"})
			return
		}

		id64, err := result.LastInsertId()
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to retrieve inserted id"})
			return
		}

		writeJSON(w, http.StatusCreated, InsertPostResponse{ID: int(id64)})
	})

	// ── POST /analyses ───────────────────────────────────────────────────────
	// Called by Gateway after it receives the NLP result for a newly inserted
	// post. Stores the result in post_analyses and updates daily_platform_stats.
	http.HandleFunc("/analyses", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
			return
		}

		var req AnalysisRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
			return
		}

		if req.PostID <= 0 || req.Sentiment == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "post_id and sentiment are required"})
			return
		}

		// Fetch the post's platform and date so we can update daily_platform_stats.
		var platform, createdAt string
		err := db.QueryRow(
			`SELECT platform, created_at FROM posts WHERE id = ?`, req.PostID,
		).Scan(&platform, &createdAt)
		if err != nil {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "post not found"})
			return
		}

		date := createdAt
		if len(date) >= 10 {
			date = date[:10]
		}

		analyzedAt := time.Now().UTC().Format(time.RFC3339)

		// Insert analysis (idempotent via INSERT OR REPLACE).
		_, err = db.Exec(
			`INSERT OR REPLACE INTO post_analyses (post_id, sentiment, sentiment_score, analyzed_at)
			 VALUES (?, ?, ?, ?)`,
			req.PostID, req.Sentiment, req.SentimentScore, analyzedAt,
		)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to store analysis"})
			return
		}

		// Update materialized daily aggregate atomically.
		positiveInc, neutralInc, negativeInc := 0, 0, 0
		switch req.Sentiment {
		case "positive":
			positiveInc = 1
		case "neutral":
			neutralInc = 1
		case "negative":
			negativeInc = 1
		}

		_, err = db.Exec(
			`INSERT INTO daily_platform_stats (date, platform, post_count, positive_count, neutral_count, negative_count)
			 VALUES (?, ?, 1, ?, ?, ?)
			 ON CONFLICT(date, platform) DO UPDATE SET
			   post_count     = post_count     + 1,
			   positive_count = positive_count + excluded.positive_count,
			   neutral_count  = neutral_count  + excluded.neutral_count,
			   negative_count = negative_count + excluded.negative_count`,
			date, platform, positiveInc, neutralInc, negativeInc,
		)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to update daily stats"})
			return
		}

		writeJSON(w, http.StatusCreated, map[string]string{"status": "ok"})
	})

	addr := ":" + port
	log.Printf("stat service listening on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("server failed: %v", err)
	}
}

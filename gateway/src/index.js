const express = require("express");
const cors = require("cors");
const jwt = require("jsonwebtoken");
const axios = require("axios");
const dotenv = require("dotenv");

dotenv.config();

const app = express();
const port = process.env.GATEWAY_PORT || 8000;
const statUrl = process.env.STAT_URL || "http://localhost:8002";
const nlpUrl = process.env.NLP_URL || "http://localhost:8001";
const jwtSecret = process.env.JWT_SECRET || "supersecret";
const demoUser = process.env.DEMO_USER || "admin";
const demoPass = process.env.DEMO_PASS || "admin123";

app.use(cors());
app.use(express.json());

function authMiddleware(req, res, next) {
  const authHeader = req.headers.authorization || "";
  const [, token] = authHeader.split(" ");

  if (!token) {
    return res.status(401).json({ error: "Missing bearer token" });
  }

  try {
    const payload = jwt.verify(token, jwtSecret);
    req.user = payload;
    return next();
  } catch (err) {
    return res.status(401).json({ error: "Invalid or expired token" });
  }
}

app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "gateway", port });
});

app.post("/auth/login", (req, res) => {
  const { username, password } = req.body || {};

  if (username !== demoUser || password !== demoPass) {
    return res.status(401).json({ error: "Invalid credentials" });
  }

  const token = jwt.sign({ username }, jwtSecret, { expiresIn: "12h" });
  return res.json({ token });
});

// ── POST /api/dashboard ───────────────────────────────────────────────────────
// Optimized: stat now returns posts enriched with pre-computed sentiment.
// NLP is only called for posts that have no cached analysis yet (fallback).
app.post("/api/dashboard", authMiddleware, async (req, res) => {
  const {
    includeKeywords = [],
    excludeKeywords = [],
    fromDate = "",
    toDate = "",
    sampleSize = 5,
  } = req.body || {};

  try {
    const statResp = await axios.post(`${statUrl}/stats`, {
      include_keywords: includeKeywords,
      exclude_keywords: excludeKeywords,
      from_date: fromDate,
      to_date: toDate,
      example_limit: sampleSize,
      post_limit: 500,
    });

    const stats = statResp.data;
    const posts = Array.isArray(stats.posts) ? stats.posts : [];

    // Separate posts that already have cached sentiment from those that don't.
    const cached = posts.filter((p) => p.sentiment !== "");
    const uncached = posts.filter((p) => p.sentiment === "");

    // Run NLP only for uncached posts (write-time miss — should be rare at scale).
    let sentimentMap = {};
    if (uncached.length > 0) {
      const texts = uncached.map((p) => p.content);
      const nlpResp = await axios.post(`${nlpUrl}/sentiment`, { texts });
      nlpResp.data.classifications.forEach((cls, idx) => {
        sentimentMap[uncached[idx].id] = {
          label: cls.label,
          score: cls.score,
        };
      });
    }

    // Merge cached + live NLP results.
    const classifiedPosts = posts.map((post) => {
      if (post.sentiment !== "") {
        return {
          ...post,
          sentiment: post.sentiment,
          sentiment_score: post.sentiment_score,
        };
      }
      const live = sentimentMap[post.id] || { label: "neutral", score: 0 };
      return { ...post, sentiment: live.label, sentiment_score: live.score };
    });

    // Build sentiment percentage from the classified list (no extra NLP call).
    const counts = { positive: 0, neutral: 0, negative: 0 };
    classifiedPosts.forEach((p) => {
      if (counts[p.sentiment] !== undefined) counts[p.sentiment]++;
    });
    const total = Math.max(1, classifiedPosts.length);
    const sentimentPercentage = {
      positive: Math.round((counts.positive / total) * 10000) / 100,
      neutral: Math.round((counts.neutral / total) * 10000) / 100,
      negative: Math.round((counts.negative / total) * 10000) / 100,
    };

    const examples = classifiedPosts.slice(0, sampleSize);

    return res.json({
      sentimentPercentage,
      topKeywords: stats.top_keywords || [],
      trends: stats.trends || [],
      examplePosts: examples,
      mentionCount: stats.mention_count || 0,
      totalAnalyzedPosts: classifiedPosts.length,
    });
  } catch (err) {
    const detail = err.response?.data || err.message;
    return res.status(500).json({
      error: "Failed to build dashboard response",
      detail,
    });
  }
});

// ── POST /api/posts ───────────────────────────────────────────────────────────
// Optimized write path: insert post → call NLP once → store analysis in stat.
// Subsequent reads never need to re-run NLP for this post.
app.post("/api/posts", authMiddleware, async (req, res) => {
  const { platform = "", author = "", content = "", createdAt = "" } = req.body || {};

  try {
    // 1. Insert the raw post into the stat service.
    const statResp = await axios.post(`${statUrl}/posts`, {
      platform,
      author,
      content,
      created_at: createdAt,
    });
    const postId = statResp.data.id;

    // 2. Run NLP on this single post (one inference, amortized across all reads).
    let sentiment = "neutral";
    let sentimentScore = 0;
    try {
      const nlpResp = await axios.post(`${nlpUrl}/sentiment`, {
        texts: [content],
      });
      const cls = nlpResp.data.classifications?.[0];
      if (cls) {
        sentiment = cls.label;
        sentimentScore = cls.score;
      }
    } catch (nlpErr) {
      // Non-fatal: analysis can be retried later; post is already stored.
      console.warn(`NLP analysis failed for post ${postId}:`, nlpErr.message);
    }

    // 3. Persist the pre-computed analysis (updates post_analyses + daily_platform_stats).
    try {
      await axios.post(`${statUrl}/analyses`, {
        post_id: postId,
        sentiment,
        sentiment_score: sentimentScore,
      });
    } catch (analysisErr) {
      console.warn(`Failed to store analysis for post ${postId}:`, analysisErr.message);
    }

    return res.status(201).json({ id: postId, sentiment, sentiment_score: sentimentScore });
  } catch (err) {
    const status = err.response?.status || 500;
    const detail = err.response?.data || err.message;
    return res.status(status).json({
      error: "Failed to insert post",
      detail,
    });
  }
});

app.listen(port, () => {
  console.log(`gateway listening on :${port}`);
});

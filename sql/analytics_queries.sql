-- Analytics queries used by the dashboard and src/analytics.py.
-- Each query starts with "-- name: <query_name>" so it can be loaded by name.
-- Parameters use SQLAlchemy's :param style.

-- name: kpi_overview
SELECT
    COUNT(*)                                                         AS total_feedback,
    AVG(f.rating)                                                    AS avg_rating,
    AVG(CASE WHEN a.sentiment_label = 'negative' THEN 1.0 ELSE 0 END) AS negative_share,
    AVG(CASE WHEN f.rating >= 4 THEN 1.0 ELSE 0 END)                 AS csat,
    100.0 * (AVG(CASE WHEN f.nps_group = 'promoter'  THEN 1.0 ELSE 0 END)
           - AVG(CASE WHEN f.nps_group = 'detractor' THEN 1.0 ELSE 0 END)) AS nps
FROM feedback f
JOIN feedback_analysis a ON a.feedback_id = f.feedback_id;

-- name: monthly_trend
SELECT
    f.feedback_month,
    COUNT(*)                                                          AS feedback_count,
    AVG(f.rating)                                                     AS avg_rating,
    AVG(a.sentiment_score)                                            AS avg_sentiment,
    SUM(CASE WHEN a.is_complaint = 1 THEN 1 ELSE 0 END)               AS complaints,
    100.0 * (AVG(CASE WHEN f.nps_group = 'promoter'  THEN 1.0 ELSE 0 END)
           - AVG(CASE WHEN f.nps_group = 'detractor' THEN 1.0 ELSE 0 END)) AS nps
FROM feedback f
JOIN feedback_analysis a ON a.feedback_id = f.feedback_id
GROUP BY f.feedback_month
ORDER BY f.feedback_month;

-- name: complaints_by_category
SELECT
    a.complaint_category,
    COUNT(*)          AS complaints,
    AVG(f.rating)     AS avg_rating
FROM feedback f
JOIN feedback_analysis a ON a.feedback_id = f.feedback_id
WHERE a.is_complaint = 1
GROUP BY a.complaint_category
ORDER BY complaints DESC;

-- name: category_by_month
SELECT
    f.feedback_month,
    a.complaint_category,
    COUNT(*) AS complaints
FROM feedback f
JOIN feedback_analysis a ON a.feedback_id = f.feedback_id
WHERE a.is_complaint = 1
GROUP BY f.feedback_month, a.complaint_category
ORDER BY f.feedback_month, complaints DESC;

-- name: segment_satisfaction
SELECT
    f.customer_segment,
    f.plan,
    COUNT(*)                                                          AS feedback_count,
    AVG(f.rating)                                                     AS avg_rating,
    AVG(CASE WHEN a.sentiment_label = 'negative' THEN 1.0 ELSE 0 END) AS negative_share,
    100.0 * (AVG(CASE WHEN f.nps_group = 'promoter'  THEN 1.0 ELSE 0 END)
           - AVG(CASE WHEN f.nps_group = 'detractor' THEN 1.0 ELSE 0 END)) AS nps
FROM feedback f
JOIN feedback_analysis a ON a.feedback_id = f.feedback_id
GROUP BY f.customer_segment, f.plan
ORDER BY nps ASC;

-- name: channel_sentiment
SELECT
    f.channel,
    a.sentiment_label,
    COUNT(*) AS feedback_count
FROM feedback f
JOIN feedback_analysis a ON a.feedback_id = f.feedback_id
GROUP BY f.channel, a.sentiment_label
ORDER BY f.channel;

-- name: daily_complaints
SELECT
    f.feedback_date,
    COUNT(*)                                             AS feedback_count,
    SUM(CASE WHEN a.is_complaint = 1 THEN 1 ELSE 0 END)  AS complaints
FROM feedback f
JOIN feedback_analysis a ON a.feedback_id = f.feedback_id
GROUP BY f.feedback_date
ORDER BY f.feedback_date;

-- name: topic_overview
SELECT
    t.topic_id,
    t.topic_label,
    t.top_terms,
    COUNT(a.feedback_id)                                               AS feedback_count,
    AVG(a.sentiment_score)                                             AS avg_sentiment,
    AVG(CASE WHEN a.sentiment_label = 'negative' THEN 1.0 ELSE 0 END)  AS negative_share
FROM topics t
LEFT JOIN feedback_analysis a ON a.topic_id = t.topic_id
GROUP BY t.topic_id, t.topic_label, t.top_terms
ORDER BY feedback_count DESC;

-- name: negative_examples_for_month
SELECT f.feedback_id, f.created_at, f.channel, a.complaint_category, f.rating, f.text
FROM feedback f
JOIN feedback_analysis a ON a.feedback_id = f.feedback_id
WHERE a.sentiment_label = 'negative' AND f.feedback_month = :month
ORDER BY a.sentiment_score ASC
LIMIT :limit;

-- name: latest_model_run
SELECT *
FROM model_runs
ORDER BY run_at DESC
LIMIT 1;

-- name: feedback_full
SELECT
    f.feedback_id, f.created_at, f.feedback_date, f.feedback_week, f.feedback_month,
    f.channel, f.plan, f.region, f.customer_segment, f.rating, f.nps_group,
    f.text, f.clean_text,
    a.sentiment_label, a.sentiment_score, a.sentiment_confidence,
    a.topic_id, t.topic_label, a.complaint_category, a.is_complaint
FROM feedback f
JOIN feedback_analysis a ON a.feedback_id = f.feedback_id
LEFT JOIN topics t ON t.topic_id = a.topic_id;

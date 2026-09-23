# Data dictionary

## `feedback`

| Column | Type | Description |
|---|---|---|
| feedback_id | varchar | Unique id |
| created_at | timestamp | When the feedback was given |
| feedback_date / feedback_week / feedback_month | date / date / 'YYYY-MM' | Precomputed time buckets |
| channel | varchar | app_store, email, call_center, social_media, survey (or `unknown`) |
| plan | varchar | prepaid, postpaid, data_only, family |
| region | varchar | Customer region |
| customer_segment | varchar | new_customer, loyal_customer, at_risk, business |
| rating | int 1-5 | Star rating (may be NULL for datasets without ratings) |
| nps_group | varchar | promoter (5★), passive (4★), detractor (1-3★) |
| text | text | Original text with PII masked |
| clean_text | text | Lower-cased, normalised text used by the models |
| word_count | int | Words in `clean_text` |

## `feedback_analysis`

| Column | Type | Description |
|---|---|---|
| sentiment_label | varchar | negative / neutral / positive (model prediction) |
| sentiment_score | real | P(positive) − P(negative), from −1 to 1 |
| sentiment_confidence | real | Probability of the predicted class |
| complaint_category | varchar | Taxonomy category or `other` |
| is_complaint | 0/1 | Negative sentiment **or** rating ≤ 2 |
| topic_id | int | Complaint topic (NULL for non-complaints) |

## KPI definitions

| KPI | Formula |
|---|---|
| CSAT | % of ratings ≥ 4 |
| NPS (simulated) | % promoters − % detractors (range −100 to +100) |
| Negative share | % of feedback predicted negative |
| Complaint share | % of feedback flagged as complaint |
| Anomaly | Day/week where complaints > rolling median + 3 × robust spread |

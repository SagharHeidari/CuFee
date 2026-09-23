# Data

Files in this folder are git-ignored (except this README). The pipeline writes the
SQLite database `cx_analytics.db` here by default.

## Datasets

| Dataset | Why | Columns to map |
|---|---|---|
| Synthetic (built in) | Runs anywhere, has a known outage to detect | none |
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) (e.g. *Cell_Phones_and_Accessories*) | Ratings + timestamps, millions of reviews | `--map timestamp=created_at` |
| [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) | Real support conversations, including telecom brands | `--map tweet_id=feedback_id` + `--sentiment-backend transformer` (no ratings) |

Tips:
* Start with `--limit 20000` to iterate quickly, then scale up.
* For the Twitter dataset, keep only customer messages (`inbound == True`) and a few
  telecom brands before running the pipeline. Doing this in a small script
  (`src/prepare_twitter.py`) is a good first extension.
* Always check a dataset's licence and terms of use, and mention them in the README.

# Step-by-step guide

The whole pipeline already works on synthetic data. Go through these steps in order.
Each one builds a skill you can honestly put on your CV and explain in an interview.

## Phase 1: Get it running (day 1)

1. **Merge the branch.** On GitHub, open a pull request from `claude/customer-experience-ai`
   into `main` and merge it. Then check the **Actions** tab: the CI pipeline (lint, tests on
   SQLite and PostgreSQL, Docker build) should turn green.
2. **Clone and install locally.**
   ```bash
   git clone https://github.com/sagharheidari/cufee.git && cd cufee
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```
3. **Run the pipeline and the dashboard.**
   ```bash
   python -m src.pipeline
   streamlit run dashboard/app.py
   ```
   Click through all 8 tabs. Try the filters (e.g. region = berlin) and look at the anomaly tab.
4. **Run the tests.** Run `pytest`. Then open one test file and read how it checks the code.
5. **Read the code in pipeline order:** `ingest.py` → `preprocessing.py` → `sentiment.py` →
   `topic_modeling.py` → `analytics.py` → `insights.py` → `dashboard/app.py`.
   You must be able to explain every line in an interview. If something is unclear, change it
   and see what breaks.

## Phase 2: PostgreSQL and Docker (day 2)

6. Install Docker Desktop, then run `docker compose up --build` and open http://localhost:8501.
7. Connect to the database with a SQL client (DBeaver, pgAdmin or `psql`) using the
   credentials in `docker-compose.yml`. Run the queries in `sql/analytics_queries.sql` by hand.
8. **Write 3 new SQL queries yourself**, for example:
   * complaint share per channel per month
   * the 10 regions with the biggest NPS drop month over month (use a window function such as `LAG`)
   * the average rating of complaints vs. non-complaints per plan

## Phase 3: Real data (days 3-5). This matters most

9. Download a real dataset (see `data/README.md`). **Amazon Reviews 2023, Cell Phones** is the
   easiest, because it has ratings. The **Twitter customer support** data is closer to a
   telecom job (congstar!).
10. Run the pipeline on it with `--limit 20000`, then look at the results critically:
    * Do the taxonomy categories fit? Edit `CATEGORY_KEYWORDS` in `topic_modeling.py` for your data.
    * Are the topics readable? Try `--topics 8/12/15` and pick the best number.
    * What is the real sentiment accuracy? It will be lower than on the synthetic data. That is normal.
11. Update the notebook `notebooks/exploratory_analysis.ipynb` with **your own** findings on the
    real data (charts, data quality problems you found, decisions you took).
12. Replace the "Results" section in the README with the real numbers and new screenshots.

## Phase 4: Improve the models (week 2)

13. `pip install -r requirements-optional.txt`, then compare
    `--sentiment-backend transformer` with the TF-IDF model on the same labelled data.
    Write a small table in the README: accuracy, macro-F1, runtime. This is your
    **model evaluation** story.
14. Try `--topic-backend bertopic` (sentence-transformers embeddings) and compare its topics with NMF.
15. Optional: create a free Anthropic API key, put it in `.env`, and try the **Ask AI** tab.
    Write down 3 questions where it helped and 1 where it failed, and explain why.

## Phase 5: Make it visible (week 2-3)

16. **Deploy the dashboard** on [Streamlit Community Cloud](https://streamlit.io/cloud)
    (free). Point it at `dashboard/app.py`. On first start the app runs the pipeline on the
    sample data by itself. Put the live link at the top of the README.
17. Pin the repository on your GitHub profile, and consider renaming it to
    `customer-experience-ai` (Settings → Rename).
18. Add it to your CV, for example:
    > **Customer Experience AI Platform** (Python, SQL/PostgreSQL, scikit-learn, NLP, Streamlit, Docker, GitHub Actions)
    > End-to-end pipeline that analyses 50k+ customer reviews: sentiment classification (macro-F1 0.xx),
    > complaint topic modeling, anomaly detection of complaint spikes, NPS/CSAT KPIs and an interactive
    > dashboard with LLM-generated insights. 30+ automated tests, CI on PostgreSQL, Dockerised.
19. Prepare a 2-minute demo for interviews: business question → dashboard → one insight → how you
    would act on it (e.g. "network complaints in Berlin rose 5× during outage week; the team gets alerted within a day").

## Optional extensions (pick 1-2)

* **REST API:** add FastAPI with `/kpis`, `/predict-sentiment` and `/ask` endpoints, plus tests.
* **Power BI:** connect Power BI Desktop to the PostgreSQL database and rebuild 2-3 pages. Many
  analyst job ads ask for Power BI.
* **German language support:** congstar customers write in German. Try a multilingual model
  (e.g. `nlptown/bert-base-multilingual-uncased-sentiment`).
* **Scheduling:** run the pipeline daily with a GitHub Actions `schedule:` trigger.
* **Churn angle:** does negative sentiment predict the `at_risk` segment?

## Git habits to show

* Work on feature branches (`feature/fastapi`), open pull requests, and let CI run.
* Write small commits with clear messages ("Add weekly NPS query", not "update").
* Never commit `.env` or large datasets (already covered by `.gitignore`).

"""Synthetic telecom customer-feedback generator.

Used for local development, tests and CI so the whole pipeline runs without
downloading anything. The data is clearly synthetic: swap in a real public
dataset with ``src.ingest.load_csv`` once the pipeline works end to end.

A network-outage spike is injected on purpose so that the anomaly detection
has a known event to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE

CHANNELS = ["app_store", "email", "call_center", "social_media", "survey"]
PLANS = ["prepaid", "postpaid", "data_only", "family"]
REGIONS = ["North", "South", "East", "West", "Berlin", "Munich"]
SEGMENTS = ["new_customer", "loyal_customer", "at_risk", "business"]

# (category, negative phrases, positive phrases)
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "network": {
        "neg": [
            "No signal at home for three days, the network coverage is terrible",
            "Mobile data keeps dropping, internet connection is extremely slow",
            "Constant network outage in my area, calls drop all the time",
            "5G coverage is a joke, speed is slower than 3G",
            "Lost connection again today, the network is down every evening",
        ],
        "pos": [
            "Network coverage is great even in rural areas",
            "Fast mobile internet and stable connection, very happy",
            "Signal strength improved a lot, 5G speed is excellent",
        ],
    },
    "billing": {
        "neg": [
            "I was charged twice this month, the invoice is wrong",
            "Unexpected extra fees on my bill, nobody can explain the charges",
            "Overcharged for roaming even though I had the EU package",
            "Refund still not processed after six weeks, billing is a mess",
            "My direct debit was taken twice and the invoice shows hidden costs",
        ],
        "pos": [
            "Billing is transparent and the invoice is easy to understand",
            "Got my refund quickly, fair and clear pricing",
        ],
    },
    "customer_service": {
        "neg": [
            "Waited on hold for 45 minutes and the hotline agent was rude",
            "Customer service never answers emails, support is useless",
            "The support agent hung up on me, terrible service",
            "Nobody at the hotline could solve my problem, I called five times",
        ],
        "pos": [
            "The support agent was friendly and solved my issue in minutes",
            "Excellent customer service, quick and helpful answer via chat",
            "Hotline staff were patient and very helpful",
        ],
    },
    "contract": {
        "neg": [
            "Cancelling my contract is impossible, they keep extending it",
            "Tariff change was not applied although it was confirmed",
            "Contract termination confirmation never arrived",
            "They switched my plan without asking, very unfair contract terms",
        ],
        "pos": [
            "Flexible contract, monthly cancellation is a big plus",
            "Changing my tariff online was simple and fast",
        ],
    },
    "sim_activation": {
        "neg": [
            "My new SIM card still is not activated after a week",
            "eSIM activation failed and the QR code does not work",
            "Number porting took forever and I had no phone service",
        ],
        "pos": [
            "SIM card arrived next day and activation was instant",
            "eSIM setup worked perfectly in two minutes",
        ],
    },
    "app_website": {
        "neg": [
            "The app crashes every time I try to log in",
            "Website login does not work and the app is very slow",
            "Cannot see my data usage in the app, it shows an error",
        ],
        "pos": [
            "The app is clean and shows my data usage clearly",
            "Great app, easy to book extra data",
        ],
    },
    "price_value": {
        "neg": [
            "Too expensive for so little data, competitors are cheaper",
            "Price increase again, not worth the money anymore",
        ],
        "pos": [
            "Great value for money, cheap tariff with lots of data",
            "Affordable price and good offers for students",
            "Best price performance ratio I have found",
        ],
    },
}

NEUTRAL = [
    "It is okay, nothing special but works most of the time",
    "Average service, some days good some days not",
    "Switched recently, too early to say but so far fine",
    "Service is acceptable, price is average",
]

SUFFIXES = [
    "",
    "",
    "",
    " Please fix this.",
    " Thanks.",
    " Would recommend.",
    " Considering switching provider.",
    " I expected better.",
    " Keep it up!",
]


def generate_feedback(
    n: int = 5000,
    start: str = "2025-01-01",
    end: str = "2025-12-31",
    seed: int = RANDOM_STATE,
    outage_week: str | None = "2025-09-08",
) -> pd.DataFrame:
    """Create ``n`` synthetic feedback records with ratings, metadata and text."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="h")
    categories = list(TEMPLATES)
    cat_weights = np.array([0.24, 0.18, 0.18, 0.1, 0.08, 0.1, 0.12])

    rows = []
    for i in range(n):
        created_at = dates[rng.integers(len(dates))]
        segment = rng.choice(SEGMENTS, p=[0.25, 0.4, 0.2, 0.15])
        # at-risk customers are more negative; loyal ones more positive
        p_neg = {"new_customer": 0.35, "loyal_customer": 0.25, "at_risk": 0.6, "business": 0.35}[segment]
        tone = rng.choice(["neg", "neu", "pos"], p=[p_neg, 0.15, 0.85 - p_neg])
        category = rng.choice(categories, p=cat_weights)

        if tone == "neu":
            text = rng.choice(NEUTRAL)
            rating = int(rng.choice([3, 3, 3, 2, 4]))
        else:
            text = rng.choice(TEMPLATES[category][tone])
            rating = (
                int(rng.choice([1, 1, 2, 2, 3], p=[0.35, 0.2, 0.2, 0.15, 0.1]))
                if tone == "neg"
                else int(rng.choice([4, 5, 5, 3], p=[0.35, 0.35, 0.25, 0.05]))
            )
        text = text + rng.choice(SUFFIXES)

        rows.append(
            {
                "feedback_id": f"FB{i:06d}",
                "created_at": created_at,
                "channel": rng.choice(CHANNELS, p=[0.3, 0.15, 0.2, 0.15, 0.2]),
                "plan": rng.choice(PLANS),
                "region": rng.choice(REGIONS),
                "customer_segment": segment,
                "rating": rating,
                "text": text,
            }
        )

    df = pd.DataFrame(rows)

    if outage_week:
        df = pd.concat([df, _outage_spike(outage_week, n // 25, rng, start_id=n)], ignore_index=True)

    # a few realistic data-quality problems for preprocessing to handle
    duplicates = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)
    df.loc[df.sample(frac=0.005, random_state=seed + 1).index, "text"] = ""
    return df.sort_values("created_at").reset_index(drop=True)


def _outage_spike(week_start: str, k: int, rng: np.random.Generator, start_id: int) -> pd.DataFrame:
    days = pd.date_range(week_start, periods=7 * 24, freq="h")
    texts = [
        "Complete network outage since this morning, no internet and no calls",
        "Network down in the whole city, mobile data not working at all",
        "Still no signal after the outage, this is unacceptable",
    ]
    return pd.DataFrame(
        {
            "feedback_id": [f"FB{start_id + j:06d}" for j in range(k)],
            "created_at": [days[rng.integers(len(days))] for _ in range(k)],
            "channel": rng.choice(["social_media", "call_center", "app_store"], size=k),
            "plan": rng.choice(PLANS, size=k),
            "region": rng.choice(["Berlin", "East"], size=k),
            "customer_segment": rng.choice(SEGMENTS, size=k),
            "rating": rng.choice([1, 1, 2], size=k),
            "text": [rng.choice(texts) for _ in range(k)],
        }
    )


if __name__ == "__main__":
    from src.config import DATA_DIR

    out = DATA_DIR / "raw_feedback.csv"
    generate_feedback().to_csv(out, index=False)
    print(f"Wrote {out}")

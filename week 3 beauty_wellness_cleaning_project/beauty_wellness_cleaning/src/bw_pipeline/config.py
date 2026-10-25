"""Central configuration for the Beauty & Wellness data-cleaning pipeline.

Every business rule, canonical vocabulary and threshold lives here so that the
cleaning logic stays free of "magic numbers" and the whole run is reproducible
from a single file (its hash is stored in the run manifest).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_FILE = DATA_RAW / "beauty_wellness_raw.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
REFERENCE_DATE = pd.Timestamp("2026-10-20")      # "today" for future-date checks
PII_SALT = os.environ.get("BW_PII_SALT", "bw-demo-salt-change-me")  # use env var in production

# --------------------------------------------------------------------------- #
# Missing-value tokens that really mean "no data"
# --------------------------------------------------------------------------- #
MISSING_TOKENS = {
    "", "na", "n/a", "#n/a", "nan", "null", "none", "nil", "-", "--", "?",
    "unknown", "not available", "missing", "not specified",
}

# --------------------------------------------------------------------------- #
# Canonical vocabularies and synonyms (keys are normalised: lower-case, no punctuation)
# --------------------------------------------------------------------------- #
CATEGORIES = ["Skincare", "Haircare", "Makeup", "Fragrance", "Bath & Body", "Wellness Supplements"]
CATEGORY_SYNONYMS = {
    "skin care": "Skincare", "face care": "Skincare", "facial care": "Skincare",
    "hair care": "Haircare", "hair": "Haircare",
    "cosmetics": "Makeup", "make up": "Makeup",
    "perfume": "Fragrance", "perfumes": "Fragrance", "fragrances": "Fragrance",
    "bath and body": "Bath & Body", "body care": "Bath & Body",
    "supplements": "Wellness Supplements", "wellness": "Wellness Supplements",
    "nutraceuticals": "Wellness Supplements",
}

BRANDS = [
    "Lumiere Botanics", "Aurava", "Veda Glow", "PureRoots", "Kovai Naturals",
    "Silk & Sage", "Zenora", "Nila Organics", "Dermavita", "Bloomhaus",
    "Ayuri", "Velvet Leaf", "Glowpath", "Herbiva",
]
BRAND_SYNONYMS = {"silk and sage": "Silk & Sage", "kovai natural": "Kovai Naturals"}

SKIN_TYPES = ["Oily", "Dry", "Combination", "Normal", "Sensitive"]
SKIN_TYPE_SYNONYMS = {"combo": "Combination", "comb": "Combination", "mixed": "Combination",
                      "oily skin": "Oily", "dry skin": "Dry", "sensitive skin": "Sensitive"}

SKIN_CONCERNS = ["Acne", "Aging", "Dryness", "Pigmentation", "Sun Protection",
                 "Hair Fall", "Dandruff", "Hydration", "General Wellness"]

GENDERS = ["Female", "Male", "Other"]
GENDER_SYNONYMS = {"f": "Female", "woman": "Female", "girl": "Female", "fem": "Female",
                   "m": "Male", "man": "Male", "boy": "Male",
                   "non binary": "Other", "nonbinary": "Other", "nb": "Other",
                   "prefer not to say": "Other", "o": "Other"}

CITIES = ["Coimbatore", "Chennai", "Bengaluru", "Mumbai", "Delhi", "Kolkata",
          "Hyderabad", "Pune", "Jaipur", "Kochi"]
CITY_SYNONYMS = {"cbe": "Coimbatore", "kovai": "Coimbatore", "madras": "Chennai",
                 "bangalore": "Bengaluru", "blr": "Bengaluru", "bombay": "Mumbai",
                 "new delhi": "Delhi", "calcutta": "Kolkata", "cochin": "Kochi"}

CHANNELS = ["Online", "Store", "App"]
CHANNEL_SYNONYMS = {"website": "Online", "web": "Online", "e commerce": "Online", "ecommerce": "Online",
                    "retail": "Store", "offline": "Store", "in store": "Store", "instore": "Store",
                    "mobile app": "App", "application": "App"}

YES_TOKENS = {"y", "yes", "true", "1", "t", "recommend", "yeah"}
NO_TOKENS = {"n", "no", "false", "0", "f", "nope"}

FUZZY_CUTOFF = 0.78  # difflib similarity needed to accept a fuzzy label match

# --------------------------------------------------------------------------- #
# Numeric parsing and domain rules
# --------------------------------------------------------------------------- #
FX_TO_INR = {"INR": 1.0, "USD": 83.0, "EUR": 90.0}   # fixed demo rates (document in report)
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

# Conversion of size units to base units (ml for volume, g for mass)
UNIT_TO_BASE = {
    "ml": ("ml", 1.0), "l": ("ml", 1000.0), "ltr": ("ml", 1000.0), "litre": ("ml", 1000.0),
    "liter": ("ml", 1000.0), "fl oz": ("ml", 29.5735),
    "g": ("g", 1.0), "gm": ("g", 1.0), "gms": ("g", 1.0), "kg": ("g", 1000.0), "oz": ("g", 28.3495),
}

AGE_RANGE = (13, 90)
RATING_RANGE = (1.0, 5.0)
SPF_RANGE = (0, 100)
SIZE_RANGE = (1.0, 2000.0)
PRICE_MIN_INR = 20.0
PRICE_HARD_MAX_INR = 100_000.0
PRICE_MAX_BY_CATEGORY = {"Skincare": 8_000, "Haircare": 6_000, "Makeup": 8_000,
                         "Fragrance": 25_000, "Bath & Body": 4_000, "Wellness Supplements": 8_000}
PRICE_PRODUCT_RATIO = (0.1, 10.0)   # price vs median price of the same product

# Outlier detection
IQR_K = 1.5
MAD_Z_THRESHOLD = 3.5
ISOLATION_FOREST_CONTAMINATION = 0.01

# --------------------------------------------------------------------------- #
# Ingredient normalisation (INCI-style synonyms) and safety/claim flags
# --------------------------------------------------------------------------- #
INGREDIENT_SYNONYMS = {
    "aqua": "water", "water (aqua)": "water", "eau": "water", "purified water": "water",
    "vit c": "ascorbic acid", "vitamin c": "ascorbic acid",
    "vitamin b3": "niacinamide", "nicotinamide": "niacinamide", "vit b3": "niacinamide",
    "vit e": "tocopherol", "vitamin e": "tocopherol",
    "sodium hyaluronate": "hyaluronic acid", "hyaluronic": "hyaluronic acid",
    "parfum": "fragrance", "perfume": "fragrance",
    "aloe barbadensis": "aloe vera", "aloe barbadensis leaf juice": "aloe vera",
    "glycerine": "glycerin", "salicylic": "salicylic acid",
    "sls": "sodium lauryl sulfate", "sles": "sodium laureth sulfate",
    "retinol (vitamin a)": "retinol", "vitamin a": "retinol",
}
INGREDIENT_FLAGS = {
    "has_fragrance": ["fragrance"],
    "has_alcohol": ["alcohol denat", "ethanol", "isopropyl alcohol"],
    "has_paraben": ["methylparaben", "propylparaben", "butylparaben"],
    "has_sulfate": ["sodium lauryl sulfate", "sodium laureth sulfate"],
    "has_retinol": ["retinol"],
    "has_niacinamide": ["niacinamide"],
    "has_hyaluronic_acid": ["hyaluronic acid"],
    "has_vitamin_c": ["ascorbic acid"],
}

# --------------------------------------------------------------------------- #
# Simple review-sentiment lexicon (transparent, dependency-free)
# --------------------------------------------------------------------------- #
POSITIVE_WORDS = {"love", "loved", "great", "amazing", "excellent", "good", "best", "perfect",
                  "smooth", "glowing", "hydrating", "worth", "happy", "recommend", "fantastic",
                  "gentle", "effective", "wonderful", "soft", "nourishing"}
NEGATIVE_WORDS = {"bad", "worst", "terrible", "awful", "irritation", "rash", "breakout", "breakouts",
                  "greasy", "sticky", "waste", "disappointed", "allergic", "burning", "harsh",
                  "dry", "overpriced", "fake", "smell", "useless", "itchy"}

# Indian seasons -> used to capture beauty/wellness seasonality
SEASON_BY_MONTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
                   6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-Monsoon", 11: "Post-Monsoon"}

AGE_BINS = [0, 24, 34, 44, 54, 120]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Dates are assumed DAY-FIRST (Indian convention) when ambiguous
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%b %d, %Y", "%Y/%m/%d", "%d.%m.%Y", "%d %B %Y"]

# --------------------------------------------------------------------------- #
# ML export
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.20
HIGH_RATING_THRESHOLD = 4.0
REQUIRED_COLUMNS = ["review_id", "customer_id", "product_id", "review_date", "rating", "category", "price_inr"]

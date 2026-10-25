"""Synthetic *raw* beauty & wellness review data with realistic quality problems.

Real public beauty/wellness data (e-commerce reviews, product catalogues, survey
responses) is messy in predictable ways.  This module builds a clean "ground
truth" first and then deliberately corrupts it, which lets us (a) demonstrate the
pipeline end-to-end and (b) measure how much of the damage the pipeline repairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

ADJECTIVES = ["Radiant", "Pure", "Deep", "Daily", "Gentle", "Intense", "Herbal", "Ultra", "Silk", "Glow"]
PRODUCT_TYPES = {
    "Skincare": ["Face Serum", "Moisturizer", "Sunscreen", "Face Wash", "Night Cream", "Toner"],
    "Haircare": ["Shampoo", "Conditioner", "Hair Oil", "Hair Mask", "Hair Serum"],
    "Makeup": ["Lipstick", "Foundation", "Kajal", "Compact Powder", "Mascara"],
    "Fragrance": ["Eau de Parfum", "Body Mist", "Deodorant Spray"],
    "Bath & Body": ["Body Lotion", "Body Wash", "Hand Cream", "Bath Salts"],
    "Wellness Supplements": ["Collagen Powder", "Biotin Tablets", "Ashwagandha Capsules", "Multivitamin"],
}
PRICE_RANGE = {"Skincare": (300, 2500), "Haircare": (200, 1800), "Makeup": (250, 3000),
               "Fragrance": (500, 6000), "Bath & Body": (150, 1200), "Wellness Supplements": (400, 3500)}
SIZE_CHOICES = {"Skincare": [("ml", [30, 50, 100]), ("g", [30, 50])],
                "Haircare": [("ml", [100, 200, 250, 400])],
                "Makeup": [("g", [4, 8, 15, 30]), ("ml", [15, 30])],
                "Fragrance": [("ml", [30, 50, 100])],
                "Bath & Body": [("ml", [100, 200, 300, 500])],
                "Wellness Supplements": [("g", [100, 200, 300, 500])]}
INGREDIENT_POOL = {
    "Skincare": ["water", "glycerin", "niacinamide", "hyaluronic acid", "ascorbic acid", "salicylic acid",
                 "retinol", "tocopherol", "aloe vera", "fragrance", "alcohol denat", "methylparaben", "zinc oxide"],
    "Haircare": ["water", "sodium lauryl sulfate", "sodium laureth sulfate", "argan oil", "coconut oil",
                 "biotin", "keratin", "fragrance", "aloe vera", "propylparaben", "glycerin"],
    "Makeup": ["talc", "mica", "dimethicone", "tocopherol", "castor oil", "iron oxides", "fragrance", "methylparaben"],
    "Fragrance": ["alcohol denat", "fragrance", "water", "limonene", "linalool", "ethanol"],
    "Bath & Body": ["water", "glycerin", "shea butter", "fragrance", "aloe vera", "sodium laureth sulfate",
                    "tocopherol", "butylparaben"],
    "Wellness Supplements": ["collagen peptides", "biotin", "ashwagandha", "ascorbic acid", "zinc", "vitamin d3"],
}
ING_VARIANTS = {"water": ["Aqua", "Water (Aqua)", "Purified Water", "water"],
                "ascorbic acid": ["Vitamin C", "Vit C", "Ascorbic Acid"],
                "niacinamide": ["Vitamin B3", "Nicotinamide", "Niacinamide"],
                "fragrance": ["Parfum", "Fragrance", "Perfume"],
                "hyaluronic acid": ["Sodium Hyaluronate", "Hyaluronic Acid"],
                "tocopherol": ["Vitamin E", "Tocopherol"], "glycerin": ["Glycerine", "Glycerin"],
                "aloe vera": ["Aloe Barbadensis", "Aloe Vera"]}
REVIEW_TEXT = {
    5: ["Absolutely love this, my skin is glowing and soft!", "Excellent product, perfect for daily use. Highly recommend.",
        "Amazing results in two weeks, worth every rupee.", "Best purchase this year, gentle and effective."],
    4: ["Good product, hydrating and smooth on skin.", "Happy with it, works well though the packaging could be better.",
        "Great texture and nice results, would buy again."],
    3: ["Average product, okay for the price.", "It is fine, nothing special but not bad either.",
        "Decent but takes a long time to show any effect."],
    2: ["Disappointed, feels sticky and the smell is strong.", "Not very effective, a bit overpriced for what it does.",
        "Caused slight irritation, would not repeat."],
    1: ["Terrible, caused a rash and breakouts. Waste of money.", "Worst product ever, burning sensation and itchy skin.",
        "Awful, fake smell and totally useless."],
}
SKIN_CONCERN_BY_CAT = {"Skincare": ["Acne", "Aging", "Pigmentation", "Hydration", "Sun Protection"],
                       "Haircare": ["Hair Fall", "Dandruff", "Dryness"], "Makeup": ["General Wellness", "Pigmentation"],
                       "Fragrance": ["General Wellness"], "Bath & Body": ["Dryness", "Hydration"],
                       "Wellness Supplements": ["General Wellness", "Aging", "Hair Fall"]}


def _build_products(rng: np.random.Generator, n_products: int = 120) -> pd.DataFrame:
    rows = []
    for i in range(n_products):
        cat = C.CATEGORIES[i % len(C.CATEGORIES)]
        brand = str(rng.choice(C.BRANDS))
        ptype = str(rng.choice(PRODUCT_TYPES[cat]))
        lo, hi = PRICE_RANGE[cat]
        unit, sizes = SIZE_CHOICES[cat][int(rng.integers(len(SIZE_CHOICES[cat])))]
        size = int(rng.choice(sizes))
        spf = int(rng.choice([15, 30, 50])) if ptype == "Sunscreen" else 0
        pool = INGREDIENT_POOL[cat]
        ings = list(rng.choice(pool, size=int(rng.integers(4, min(8, len(pool) + 1))), replace=False))
        launch = pd.Timestamp("2021-01-01") + pd.Timedelta(days=int(rng.integers(0, 1500)))
        rows.append(dict(product_id=f"P{1000 + i}", product_name=f"{brand.split()[0]} {rng.choice(ADJECTIVES)} {ptype}"
                         + (f" SPF {spf}" if spf else ""), brand=brand, category=cat,
                         price=int(round(rng.uniform(lo, hi), -1)), size_value=size, size_unit=unit, spf=spf,
                         ingredients=ings, launch_date=launch))
    return pd.DataFrame(rows)


def _build_customers(rng: np.random.Generator, n_customers: int) -> pd.DataFrame:
    rows = []
    for i in range(n_customers):
        phone = f"{rng.choice([6, 7, 8, 9])}{rng.integers(10**8, 10**9)}"
        rows.append(dict(customer_id=f"C{20000 + i}", email=f"user{20000 + i}@{rng.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'rediffmail.com'])}",
                         phone=phone, age=int(np.clip(rng.normal(31, 9), 18, 65)),
                         gender=str(rng.choice(C.GENDERS, p=[0.72, 0.25, 0.03])),
                         city=str(rng.choice(C.CITIES, p=[0.22, 0.14, 0.14, 0.12, 0.1, 0.06, 0.08, 0.06, 0.03, 0.05])),
                         skin_type=str(rng.choice(C.SKIN_TYPES, p=[0.25, 0.2, 0.3, 0.15, 0.1]))))
    return pd.DataFrame(rows)


def _messy_label(rng, value: str, p_case=0.12, p_space=0.08, p_typo=0.04) -> str:
    """Case/whitespace/typo noise for categorical labels."""
    v = value
    r = rng.random()
    if r < p_typo and len(v) > 4:
        i = int(rng.integers(1, len(v) - 1))
        v = v[:i] + v[i + 1:]                       # dropped character
    elif r < p_typo + p_case:
        v = v.upper() if rng.random() < 0.5 else v.lower()
    if rng.random() < p_space:
        v = f"  {v} "
    return v


def _fmt_price(rng, p: float) -> str:
    r = rng.random()
    if r < 0.45: return str(int(p))
    if r < 0.60: return f"₹{int(p):,}"
    if r < 0.72: return f"Rs. {int(p)}"
    if r < 0.82: return f"INR {int(p):,}.00"
    if r < 0.90: return f"{int(p)} /-"
    if r < 0.96: return f"${p / 83:.2f}"
    return f"€{p / 90:.2f}"


def _fmt_size(rng, value: float, unit: str) -> str:
    r = rng.random()
    if unit == "ml":
        if r < 0.55: return f"{value:g} ml"
        if r < 0.70: return f"{value:g}ml"
        if r < 0.85: return f"{value / 29.5735:.1f} fl oz"
        return f"{value / 1000:g} L"
    if r < 0.60: return f"{value:g} g"
    if r < 0.75: return f"{value:g}gm"
    if r < 0.88: return f"{value / 28.3495:.1f} oz"
    return f"{value / 1000:g} kg"


def _fmt_date(rng, d: pd.Timestamp) -> str:
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%b %d, %Y", "%d.%m.%Y"]
    w = [0.35, 0.2, 0.12, 0.13, 0.1, 0.1]
    return d.strftime(fmts[int(rng.choice(len(fmts), p=w))])


def _fmt_rating(rng, r: int) -> str:
    x = rng.random()
    if x < 0.55: return str(r)
    if x < 0.68: return f"{r}.0"
    if x < 0.78: return f"{r} stars"
    if x < 0.86: return f"{r}/5"
    if x < 0.91: return f"{r * 2}/10"
    if x < 0.95: return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}[r]
    return str(rng.choice([0, 6, 55, 44, -1, 11]))      # impossible values / double-key typos


def _fmt_ingredients(rng, ings: list[str]) -> str:
    names = []
    for ing in ings:
        if ing in ING_VARIANTS and rng.random() < 0.6:
            names.append(str(rng.choice(ING_VARIANTS[ing])))
        elif rng.random() < 0.5:
            names.append(ing.title())
        else:
            names.append(ing)
    if rng.random() < 0.1:
        names.append(names[0])                        # duplicated ingredient
    sep = str(rng.choice([", ", "; ", " | ", ",", "\n"], p=[0.5, 0.2, 0.15, 0.1, 0.05]))
    return sep.join(names)


def _messy_text(rng, t: str) -> str:
    r = rng.random()
    if r < 0.08: t = f"<p>{t}</p>"
    elif r < 0.14: t = t + " https://bit.ly/offer123"
    elif r < 0.22: t = t + " 😍😍👍"
    elif r < 0.27: t = t.upper()
    elif r < 0.31: t = t.replace("!", "!!!!!")
    elif r < 0.35: t = "   " + t + "   \n"
    return t


def generate_raw_data(n_rows: int = 5000, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Return a deliberately messy DataFrame of string-like raw review records."""
    rng = np.random.default_rng(seed)
    products = _build_products(rng)
    customers = _build_customers(rng, n_customers=max(600, n_rows // 3))
    p_idx = rng.integers(0, len(products), n_rows)
    c_idx = rng.integers(0, len(customers), n_rows)
    prod = products.iloc[p_idx].reset_index(drop=True)
    cust = customers.iloc[c_idx].reset_index(drop=True)

    ratings = rng.choice([1, 2, 3, 4, 5], size=n_rows, p=[0.06, 0.08, 0.14, 0.30, 0.42])
    review_dates = [launch + pd.Timedelta(days=int(rng.integers(5, 1400))) for launch in prod["launch_date"]]
    review_dates = [min(d, C.REFERENCE_DATE - pd.Timedelta(days=1)) for d in review_dates]
    channel = rng.choice(["Online", "Store", "App"], size=n_rows, p=[0.55, 0.25, 0.20])
    texts = [str(rng.choice(REVIEW_TEXT[int(r)])) for r in ratings]
    recommend = np.where(ratings >= 4, "Yes", np.where(ratings <= 2, "No", rng.choice(["Yes", "No"], n_rows)))

    df = pd.DataFrame({
        "Review ID": [f"R{100000 + i}" for i in range(n_rows)],
        "Customer ID": cust["customer_id"], "Customer Email": cust["email"], "Customer Phone": cust["phone"],
        "Customer Age": cust["age"].astype(object), "Gender": cust["gender"], "City": cust["city"],
        "Skin Type": cust["skin_type"],
        "Skin Concern": [str(rng.choice(SKIN_CONCERN_BY_CAT[c])) for c in prod["category"]],
        "Review Date": review_dates, "Product ID": prod["product_id"], "Product Name": prod["product_name"],
        "Brand": prod["brand"], "Category": prod["category"], "Price": prod["price"].astype(float).astype(object),
        "Size": [(v, u) for v, u in zip(prod["size_value"], prod["size_unit"])],
        "Launch Date": prod["launch_date"], "SPF": prod["spf"].astype(object),
        "Ingredients": prod["ingredients"], "Rating": ratings.astype(object), "Review Text": texts,
        "Would Recommend": recommend, "Purchase Channel": channel,
    })

    # ------------------------------------------------------------------ corruption
    out = df.copy().astype(object)
    ix = out.index

    out["Price"] = [_fmt_price(rng, p) for p in df["Price"]]
    out["Size"] = [_fmt_size(rng, v, u) for v, u in df["Size"]]
    out["Review Date"] = [_fmt_date(rng, d) for d in df["Review Date"]]
    out["Launch Date"] = [_fmt_date(rng, d) for d in df["Launch Date"]]
    out["Rating"] = [_fmt_rating(rng, int(r)) for r in df["Rating"]]
    out["Ingredients"] = [_fmt_ingredients(rng, i) for i in df["Ingredients"]]
    out["Review Text"] = [_messy_text(rng, t) for t in df["Review Text"]]
    out["SPF"] = [("SPF " + str(s) if s and rng.random() < 0.5 else (f"{s}+" if s else "0")) for s in df["SPF"]]
    for col in ["Brand", "Category", "Skin Type", "City", "Gender", "Purchase Channel"]:
        out[col] = [_messy_label(rng, str(v)) for v in df[col]]
    # synonym-style inconsistencies
    syn_cat = {"Skincare": "skin care", "Haircare": "Hair-Care", "Makeup": "Make up", "Bath & Body": "Bath and Body",
               "Fragrance": "Perfume", "Wellness Supplements": "Supplements"}
    m = rng.random(n_rows) < 0.08
    out.loc[m, "Category"] = [syn_cat[c] for c in df.loc[m, "Category"]]
    m = rng.random(n_rows) < 0.10
    out.loc[m, "City"] = [{"Coimbatore": "CBE", "Bengaluru": "Bangalore", "Chennai": "Madras"}.get(c, c) for c in df.loc[m, "City"]]
    m = rng.random(n_rows) < 0.10
    out.loc[m, "Gender"] = [{"Female": "F", "Male": "M"}.get(g, g) for g in df.loc[m, "Gender"]]
    out["Would Recommend"] = [str(rng.choice({"Yes": ["Yes", "Y", "1", "TRUE", "yes"], "No": ["No", "N", "0", "FALSE", "no"]}[v])) for v in df["Would Recommend"]]

    # impossible / extreme values
    m = rng.random(n_rows) < 0.020
    out.loc[m, "Price"] = [str(rng.choice(["99999", "1299000", "-499", "0"])) for _ in range(m.sum())]
    m = rng.random(n_rows) < 0.012
    out.loc[m, "Customer Age"] = [str(rng.choice([-5, 0, 7, 150, 999, 5])) for _ in range(m.sum())]
    m = rng.random(n_rows) < 0.010
    out.loc[m, "Size"] = [str(rng.choice(["0 ml", "5000 ml", "-30 g"])) for _ in range(m.sum())]
    m = rng.random(n_rows) < 0.012
    out.loc[m, "Review Date"] = [(C.REFERENCE_DATE + pd.Timedelta(days=int(rng.integers(30, 800)))).strftime("%Y-%m-%d") for _ in range(m.sum())]
    m = rng.random(n_rows) < 0.010
    out.loc[m, "Review Date"] = [(pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.integers(0, 500)))).strftime("%d/%m/%Y") for _ in range(m.sum())]

    # invalid contact data
    bad_email = rng.random(n_rows) < 0.04
    out.loc[bad_email, "Customer Email"] = [str(rng.choice(["john@@gmail", "user.gmail.com", "abc@", "@yahoo.com", "no email"])) for _ in range(bad_email.sum())]
    upper = rng.random(n_rows) < 0.08
    out.loc[upper, "Customer Email"] = out.loc[upper, "Customer Email"].str.upper()
    out["Customer Phone"] = [(f"+91 {p[:5]}-{p[5:]}" if rng.random() < 0.3 else f"0{p}" if rng.random() < 0.2 else p) for p in df["Customer Phone"]]
    m = rng.random(n_rows) < 0.03
    out.loc[m, "Customer Phone"] = "12345"

    # missing values: MCAR everywhere + MAR (age / skin type depend on channel & category)
    tokens = ["", "N/A", "NA", "null", "-", "?", "None", "not available"]
    for col, p in {"Customer Age": 0.05, "Gender": 0.03, "City": 0.03, "Skin Type": 0.05, "Skin Concern": 0.04,
                   "Price": 0.025, "Size": 0.025, "Brand": 0.015, "Launch Date": 0.03, "SPF": 0.02,
                   "Ingredients": 0.04, "Rating": 0.02, "Review Text": 0.05, "Would Recommend": 0.04,
                   "Purchase Channel": 0.02, "Review Date": 0.01, "Customer Email": 0.02, "Customer Phone": 0.03}.items():
        m = rng.random(n_rows) < p
        out.loc[m, col] = [str(rng.choice(tokens)) for _ in range(m.sum())]
    mar_age = (df["Purchase Channel"] == "App") & (rng.random(n_rows) < 0.25)       # app users hide age
    out.loc[mar_age, "Customer Age"] = "N/A"
    mar_skin = (df["Category"] == "Wellness Supplements") & (rng.random(n_rows) < 0.30)
    out.loc[mar_skin, "Skin Type"] = ""

    # duplicates: exact copies + "near duplicates" (same customer/product/date, new review id, text noise)
    exact = out.sample(frac=0.03, random_state=seed)
    near = out.sample(frac=0.02, random_state=seed + 1).copy()
    near["Review ID"] = [f"R{900000 + i}" for i in range(len(near))]
    near["Review Text"] = near["Review Text"].astype(str).str.strip().str.lower()
    out = pd.concat([out, exact, near], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # make column order slightly untidy like real exports
    return out.astype(str).replace({"nan": ""})


def write_raw_csv(path=C.RAW_FILE, n_rows: int = 5000, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_raw_data(n_rows=n_rows, seed=seed)
    df.to_csv(path, index=False)
    return df

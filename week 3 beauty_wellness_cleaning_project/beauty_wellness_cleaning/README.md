# Beauty & Wellness – Data Cleaning & Preprocessing Pipeline

**Author:** Sumit Kumar Sharma – Data Science with Python Analyst Intern (Coimbatore)
**Task:** Week 3 – Data Cleaning and Preprocessing Pipeline Development

A reproducible, tested pandas/NumPy/scikit-learn pipeline that turns messy beauty & wellness
review data (product reviews + product catalogue + customer attributes) into
(1) an analysis-ready clean table and (2) a scaled, encoded, train/test-split ML matrix.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --strict                 # generate 5,000 synthetic raw rows, clean, validate, export, plot
python -m unittest discover -s tests -v # 19 unit/integration tests
python main.py --input my_raw.csv       # clean your own CSV (same column layout as data/raw/*.csv)
```

## What it fixes
| Problem | Solution |
|---|---|
| `N/A`, `?`, `-`, `null`, blanks | token normalisation → true `NaN` |
| exact + near duplicates | `drop_duplicates`, then (customer, product, date) key keeping the most complete row |
| `skin care`/`SKINCARE`/`Skincre`, `CBE`/`Kovai` | exact → synonym → fuzzy (`difflib`) mapping to canonical lists |
| `₹1,299`, `Rs. 999`, `$12.99`, `499 /-` | regex parse + FX conversion → INR float |
| `1.7 fl oz`, `0.1 L`, `100gm` | unit conversion → ml / g |
| ratings `4 stars`, `8/10`, `four`, `55`, `0` | rescale to 1-5, typo repair, invalid → quarantine |
| 8 different date formats, future dates, review-before-launch | multi-format parser, rule checks |
| HTML, URLs, emoji, SHOUTING in reviews | text normaliser |
| ingredient lists (`Aqua`, `Vitamin C`, mixed separators) | INCI synonym map, de-dup, claim flags |
| raw e-mail / phone (PII) | validation + salted SHA-256 pseudonyms, raw columns dropped |
| missing values | entity reconciliation → group median → `Unknown` + flags |
| outliers | domain rules + product-median ratio + IQR + MAD + Isolation Forest |
| scaling / encoding | log, StandardScaler, RobustScaler (fit on **train only**), one-hot |

## Outputs
| File | Content |
|---|---|
| `data/processed/beauty_wellness_clean.csv` | cleaned, feature-rich analysis table |
| `data/processed/beauty_wellness_ml_ready.csv` | scaled + encoded model matrix with `split` column |
| `data/processed/quarantine_rows.csv` | rejected rows with the reason (audit trail) |
| `reports/cleaning_log.csv` | rows / missing cells / seconds for each step |
| `reports/issue_register.csv` | every issue found, how many records, what was done |
| `reports/validation_checks.csv`, `quality_scorecard.csv`, `quality_report.json` | quality evidence |
| `reports/run_manifest.json` | library versions, seed, SHA-256 of config + inputs + outputs |
| `reports/data_dictionary.csv` | column dtype + description |
| `reports/figures/0*.png` | flowchart and before/after charts |

## Project layout
```
main.py                    CLI entry point
src/bw_pipeline/
  config.py                all rules, vocabularies, thresholds
  data_generator.py        synthetic raw data with injected defects
  cleaning.py              parsers, standardisers, text/PII helpers
  imputation.py            entity reconciliation, group median, constants
  outliers.py              IQR, MAD, grouped winsorising, Isolation Forest
  features.py              price/100ml, sentiment, season, ingredient flags
  validation.py            22 rule-based checks + quality scorecard
  pipeline.py              15 ordered steps, logging, ML export, manifest
  visualization.py         flowchart and charts
tests/test_pipeline.py     19 tests
```

## Design principles
1. Read everything as text – never let `read_csv` guess types.
2. Never destroy information silently – invalid values become `NaN`, unusable rows go to quarantine.
3. Keep originals next to treated values (`price_inr` vs `price_inr_winsor`) and flag, don't delete, outliers.
4. Fit scalers on the training split only (no leakage).
5. Privacy by design – PII is pseudonymised before anything is saved.
6. Everything configurable in one file; seeds and hashes recorded for reproducibility.

## Limitations
The bundled data is **synthetic** (generated with realistic defects) so results are demonstrative; FX rates are fixed;
dates are assumed day-first; the sentiment lexicon is deliberately simple.

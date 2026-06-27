# =============================================================================
# Synthetic Respondent Dataset Generator
# Freeze-Dried Home-Cooked Meal Service - Customer Research Questionnaire
# =============================================================================
# Generates >=500 synthetic survey responses built around 4 underlying
# "persona" archetypes (real but hidden market segments), so that downstream
# techniques have genuine signal to find:
#   - Classification target : likely_to_use (Yes/No), derived from E1
#   - Regression targets    : max_price_per_kg_inr, monthly_spend_inr
#   - Clustering input      : D1-D8 attitude scores + demographics
#   - Association rules     : multi-select dish/occasion/channel baskets
#
# Realism is added via:
#   - Gaussian noise on every persona-driven trait (so it isn't perfectly
#     separable)
#   - Lognormal distributions for income/price/spend -> natural right skew
#   - ~3% outlier injection on key numeric columns
#   - ~2-4% missing values on optional/soft fields (brand used, comments,
#     locality) to mimic real survey non-response
#
# Run this directly in Google Colab - no special setup needed beyond:
#   !pip install mlxtend -q   (only needed if you run the association-rule
#                               validation section at the bottom)
# =============================================================================

import numpy as np
import pandas as pd

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N = 520  # >= 500 respondents; a handful above 500 mimics real-world response counts

# -----------------------------------------------------------------------------
# 1. PERSONA DEFINITIONS (hidden ground-truth market segments)
# -----------------------------------------------------------------------------
# These are NOT a survey question - they represent the real underlying
# customer types we are trying to discover via clustering. Kept in the
# output as `true_persona` purely so you can sanity-check / validate your
# own clustering results against a known answer. Drop it before running
# unsupervised clustering to avoid leakage; use it afterwards to check
# how well your clusters recovered the real segments.

PERSONAS = {
    "NRI_Quality":       {"weight": 0.28, "segment_probs": {"NRI_Family": 0.80, "Traveler_Student": 0.10, "Neither": 0.10}},
    "Traveler_Budget":   {"weight": 0.27, "segment_probs": {"NRI_Family": 0.10, "Traveler_Student": 0.80, "Neither": 0.10}},
    "Convenience_Family":{"weight": 0.20, "segment_probs": {"NRI_Family": 0.45, "Traveler_Student": 0.15, "Neither": 0.40}},
    "Skeptical_LowIntent":{"weight": 0.25, "segment_probs": {"NRI_Family": 0.15, "Traveler_Student": 0.15, "Neither": 0.70}},
}

persona_names = list(PERSONAS.keys())
persona_weights = [PERSONAS[p]["weight"] for p in persona_names]
true_persona = rng.choice(persona_names, size=N, p=persona_weights)

def seg_for_row(p):
    probs = PERSONAS[p]["segment_probs"]
    return rng.choice(list(probs.keys()), p=list(probs.values()))

segment = np.array([seg_for_row(p) for p in true_persona])

# -----------------------------------------------------------------------------
# 2. DEMOGRAPHICS
# -----------------------------------------------------------------------------
AGE_PARAMS = {  # (mean, sd) by persona
    "NRI_Quality": (42, 9), "Traveler_Budget": (26, 5),
    "Convenience_Family": (38, 8), "Skeptical_LowIntent": (35, 11),
}
age = np.array([np.clip(rng.normal(*AGE_PARAMS[p]), 18, 68) for p in true_persona]).round(0)

def bucket_age(a):
    if a < 25: return "Under 25"
    if a < 35: return "25-34"
    if a < 45: return "35-44"
    if a < 55: return "45-54"
    return "55 and above"
age_group = np.array([bucket_age(a) for a in age])

gender = rng.choice(["Male", "Female", "Prefer not to say"], size=N, p=[0.48, 0.48, 0.04])

HYD_AREAS = ["Gachibowli", "Madhapur", "Banjara Hills", "Jubilee Hills", "Kukatpally",
             "Begumpet", "Secunderabad", "Kondapur", "LB Nagar", "Dilsukhnagar",
             "Ameerpet", "Miyapur", "Uppal", "Manikonda"]
locality = rng.choice(HYD_AREAS, size=N)

OCC_PROBS = {  # occupation distribution by persona
    "NRI_Quality":        {"Salaried / Employed": 0.45, "Self-employed / Business owner": 0.30, "Homemaker": 0.15, "Student": 0.02, "Retired": 0.06, "Other": 0.02},
    "Traveler_Budget":     {"Salaried / Employed": 0.35, "Self-employed / Business owner": 0.10, "Homemaker": 0.05, "Student": 0.45, "Retired": 0.01, "Other": 0.04},
    "Convenience_Family":  {"Salaried / Employed": 0.50, "Self-employed / Business owner": 0.20, "Homemaker": 0.20, "Student": 0.03, "Retired": 0.05, "Other": 0.02},
    "Skeptical_LowIntent": {"Salaried / Employed": 0.40, "Self-employed / Business owner": 0.15, "Homemaker": 0.20, "Student": 0.10, "Retired": 0.10, "Other": 0.05},
}
occupation = np.array([rng.choice(list(OCC_PROBS[p].keys()), p=list(OCC_PROBS[p].values())) for p in true_persona])

INCOME_LOGPARAMS = {  # (mean_log, sigma_log) in INR -> lognormal gives natural right skew
    "NRI_Quality": (11.7, 0.45), "Traveler_Budget": (10.9, 0.40),
    "Convenience_Family": (11.3, 0.40), "Skeptical_LowIntent": (11.0, 0.50),
}
monthly_income_inr = np.array([rng.lognormal(*INCOME_LOGPARAMS[p]) for p in true_persona]).round(-2)
monthly_income_inr = np.clip(monthly_income_inr, 15000, 1000000)

def bucket_income(v):
    if v < 50000: return "Below Rs 50,000"
    if v < 100000: return "Rs 50,000 - 1,00,000"
    if v < 200000: return "Rs 1,00,001 - 2,00,000"
    if v < 350000: return "Rs 2,00,001 - 3,50,000"
    return "Above Rs 3,50,000"
income_bracket = np.array([bucket_income(v) for v in monthly_income_inr])

household_size_bracket = rng.choice(["1-2", "3-4", "5-6", "7 or more"], size=N, p=[0.25, 0.45, 0.22, 0.08])

# -----------------------------------------------------------------------------
# 3. SCREENING & AWARENESS (Section A1, B1-B4)
# -----------------------------------------------------------------------------
EVER_SENT_PROB = {"NRI_Quality": 0.92, "Traveler_Budget": 0.75, "Convenience_Family": 0.55, "Skeptical_LowIntent": 0.30}
ever_sent_homecooked_food = np.array([rng.random() < EVER_SENT_PROB[p] for p in true_persona])
ever_sent_homecooked_food = np.where(ever_sent_homecooked_food, "Yes", "No")

HEARD_PROB = {"NRI_Quality": 0.70, "Traveler_Budget": 0.60, "Convenience_Family": 0.40, "Skeptical_LowIntent": 0.20}
heard_of_freeze_dry = np.array([rng.random() < HEARD_PROB[p] for p in true_persona])
heard_of_freeze_dry = np.where(heard_of_freeze_dry, "Yes", "No")

used_freeze_dry_before = np.where(
    (heard_of_freeze_dry == "Yes") & (rng.random(N) < 0.30), "Yes", "No"
)

BRAND_NAMES = ["Spice Up Foods", "My Taste My Meal", "Desi Khana", "Dryfii", "Leela Instant Foods"]
brand_used = np.array([
    rng.choice(BRAND_NAMES) if used_freeze_dry_before[i] == "Yes" else np.nan
    for i in range(N)
], dtype=object)

FREQ_PROBS = {
    "NRI_Quality":        {"Never": 0.05, "Rarely (once or twice a year)": 0.20, "Occasionally (every 2-3 months)": 0.40, "Frequently (monthly or more)": 0.35},
    "Traveler_Budget":     {"Never": 0.10, "Rarely (once or twice a year)": 0.30, "Occasionally (every 2-3 months)": 0.40, "Frequently (monthly or more)": 0.20},
    "Convenience_Family":  {"Never": 0.20, "Rarely (once or twice a year)": 0.35, "Occasionally (every 2-3 months)": 0.30, "Frequently (monthly or more)": 0.15},
    "Skeptical_LowIntent": {"Never": 0.45, "Rarely (once or twice a year)": 0.35, "Occasionally (every 2-3 months)": 0.15, "Frequently (monthly or more)": 0.05},
}
frequency_sending_food = np.array([rng.choice(list(FREQ_PROBS[p].keys()), p=list(FREQ_PROBS[p].values())) for p in true_persona])

# Multi-select B5: methods currently used (binary columns -> basket data)
METHOD_OPTIONS = ["method_no_preservation", "method_freeze_athome", "method_vacuum_self",
                   "method_freezedry_service", "method_none"]
METHOD_BASE_PROB = {  # baseline probability per option, persona modifies it
    "method_no_preservation": 0.45, "method_freeze_athome": 0.30, "method_vacuum_self": 0.15,
    "method_freezedry_service": 0.10, "method_none": 0.20,
}
METHOD_PERSONA_MULT = {
    "NRI_Quality":        {"method_freezedry_service": 2.5, "method_vacuum_self": 1.5, "method_none": 0.3},
    "Traveler_Budget":     {"method_no_preservation": 1.3, "method_freeze_athome": 1.2, "method_none": 0.6},
    "Convenience_Family":  {"method_freeze_athome": 1.3},
    "Skeptical_LowIntent": {"method_none": 2.0, "method_no_preservation": 0.7},
}
method_cols = {opt: np.zeros(N, dtype=int) for opt in METHOD_OPTIONS}
for i, p in enumerate(true_persona):
    for opt in METHOD_OPTIONS:
        base = METHOD_BASE_PROB[opt]
        mult = METHOD_PERSONA_MULT.get(p, {}).get(opt, 1.0)
        prob = np.clip(base * mult, 0.02, 0.95)
        method_cols[opt][i] = int(rng.random() < prob)

# -----------------------------------------------------------------------------
# 4. PREFERENCES - MULTI-SELECT BASKETS (Section C1-C3) + C4 single-select
# -----------------------------------------------------------------------------
DISH_OPTIONS = ["dish_dal", "dish_rice_biryani", "dish_rotis", "dish_sabzi",
                 "dish_nonveg_curry", "dish_sweets_snacks", "dish_pickles_chutneys"]
DISH_BASE_PROB = {opt: 0.35 for opt in DISH_OPTIONS}
DISH_PERSONA_MULT = {
    "NRI_Quality":        {"dish_dal": 1.6, "dish_rice_biryani": 1.5, "dish_rotis": 1.6, "dish_nonveg_curry": 1.4, "dish_pickles_chutneys": 1.3},
    "Traveler_Budget":     {"dish_rice_biryani": 1.4, "dish_sweets_snacks": 1.6, "dish_pickles_chutneys": 1.2, "dish_dal": 0.8},
    "Convenience_Family":  {"dish_sabzi": 1.4, "dish_dal": 1.2, "dish_rotis": 1.1},
    "Skeptical_LowIntent": {opt: 0.6 for opt in DISH_OPTIONS},
}

OCCASION_OPTIONS = ["occasion_abroad_family", "occasion_own_travel", "occasion_festival_gifting",
                     "occasion_emergency_stock", "occasion_elderly_parents"]
OCCASION_BASE_PROB = {opt: 0.30 for opt in OCCASION_OPTIONS}
OCCASION_PERSONA_MULT = {
    "NRI_Quality":        {"occasion_abroad_family": 2.2, "occasion_festival_gifting": 1.6, "occasion_elderly_parents": 1.4},
    "Traveler_Budget":     {"occasion_own_travel": 2.3, "occasion_abroad_family": 1.2, "occasion_emergency_stock": 0.7},
    "Convenience_Family":  {"occasion_festival_gifting": 1.5, "occasion_emergency_stock": 1.6, "occasion_elderly_parents": 1.3},
    "Skeptical_LowIntent": {opt: 0.5 for opt in OCCASION_OPTIONS},
}

CHANNEL_OPTIONS = ["channel_instagram", "channel_google", "channel_referral",
                    "channel_kirana_partner", "channel_college_rwa", "channel_influencer"]
CHANNEL_BASE_PROB = {opt: 0.30 for opt in CHANNEL_OPTIONS}
CHANNEL_PERSONA_MULT = {
    "NRI_Quality":        {"channel_referral": 1.8, "channel_instagram": 1.2, "channel_kirana_partner": 1.3},
    "Traveler_Budget":     {"channel_instagram": 1.8, "channel_google": 1.5, "channel_college_rwa": 1.7, "channel_influencer": 1.4},
    "Convenience_Family":  {"channel_referral": 1.4, "channel_kirana_partner": 1.3},
    "Skeptical_LowIntent": {opt: 0.5 for opt in CHANNEL_OPTIONS},
}

def build_multiselect(options, base_prob, persona_mult):
    cols = {opt: np.zeros(N, dtype=int) for opt in options}
    for i, p in enumerate(true_persona):
        any_selected = False
        for opt in options:
            base = base_prob[opt]
            mult = persona_mult.get(p, {}).get(opt, 1.0)
            prob = np.clip(base * mult, 0.03, 0.95)
            val = int(rng.random() < prob)
            cols[opt][i] = val
            any_selected = any_selected or val
        if not any_selected:  # ensure at least one selection, as a real respondent would
            forced = rng.choice(options)
            cols[forced][i] = 1
    return cols

dish_cols = build_multiselect(DISH_OPTIONS, DISH_BASE_PROB, DISH_PERSONA_MULT)
occasion_cols = build_multiselect(OCCASION_OPTIONS, OCCASION_BASE_PROB, OCCASION_PERSONA_MULT)
channel_cols = build_multiselect(CHANNEL_OPTIONS, CHANNEL_BASE_PROB, CHANNEL_PERSONA_MULT)

PACKAGING_PROBS = {
    "NRI_Quality":        {"Sealed pouches (lightweight, compact)": 0.55, "Jars/containers (reusable)": 0.30, "No strong preference": 0.15},
    "Traveler_Budget":     {"Sealed pouches (lightweight, compact)": 0.70, "Jars/containers (reusable)": 0.10, "No strong preference": 0.20},
    "Convenience_Family":  {"Sealed pouches (lightweight, compact)": 0.45, "Jars/containers (reusable)": 0.35, "No strong preference": 0.20},
    "Skeptical_LowIntent": {"Sealed pouches (lightweight, compact)": 0.40, "Jars/containers (reusable)": 0.25, "No strong preference": 0.35},
}
packaging_pref = np.array([rng.choice(list(PACKAGING_PROBS[p].keys()), p=list(PACKAGING_PROBS[p].values())) for p in true_persona])

# -----------------------------------------------------------------------------
# 5. ATTITUDES - LIKERT 1-5 (Section D1-D8) -> main clustering input
# -----------------------------------------------------------------------------
# (mean, sd) per persona per statement - deliberately distinct profiles so
# clustering recovers interpretable, well-separated segments
LIKERT_PARAMS = {
    "taste_importance":      {"NRI_Quality": (4.6, 0.35), "Traveler_Budget": (3.0, 0.50), "Convenience_Family": (3.9, 0.40), "Skeptical_LowIntent": (2.0, 0.60)},
    "price_importance":      {"NRI_Quality": (2.6, 0.50), "Traveler_Budget": (4.7, 0.35), "Convenience_Family": (3.6, 0.45), "Skeptical_LowIntent": (3.7, 0.60)},
    "shelf_life_importance": {"NRI_Quality": (4.5, 0.35), "Traveler_Budget": (3.6, 0.50), "Convenience_Family": (3.8, 0.45), "Skeptical_LowIntent": (2.2, 0.60)},
    "hygiene_importance":    {"NRI_Quality": (4.8, 0.25), "Traveler_Budget": (3.7, 0.50), "Convenience_Family": (4.3, 0.40), "Skeptical_LowIntent": (2.6, 0.70)},
    "convenience_importance":{"NRI_Quality": (3.8, 0.50), "Traveler_Budget": (4.2, 0.40), "Convenience_Family": (4.7, 0.30), "Skeptical_LowIntent": (2.4, 0.60)},
    "brand_trust_importance":{"NRI_Quality": (4.4, 0.40), "Traveler_Budget": (3.2, 0.50), "Convenience_Family": (3.8, 0.45), "Skeptical_LowIntent": (2.0, 0.60)},
    "variety_importance":    {"NRI_Quality": (4.3, 0.40), "Traveler_Budget": (3.4, 0.50), "Convenience_Family": (3.6, 0.45), "Skeptical_LowIntent": (1.9, 0.60)},
    "speed_importance":      {"NRI_Quality": (3.6, 0.50), "Traveler_Budget": (4.2, 0.40), "Convenience_Family": (3.9, 0.45), "Skeptical_LowIntent": (2.3, 0.70)},
}

likert_cols = {}
for trait, params in LIKERT_PARAMS.items():
    vals = np.array([rng.normal(*params[p]) for p in true_persona])
    vals = np.clip(np.round(vals), 1, 5).astype(int)
    likert_cols[trait] = vals

# -----------------------------------------------------------------------------
# 6. PURCHASE INTENT & WILLINGNESS TO PAY (Section E1-E6)
# -----------------------------------------------------------------------------
LIKELIHOOD_PARAMS = {"NRI_Quality": (4.3, 0.7), "Traveler_Budget": (3.8, 0.8),
                      "Convenience_Family": (3.2, 0.9), "Skeptical_LowIntent": (1.9, 0.9)}
likelihood_raw = np.array([rng.normal(*LIKELIHOOD_PARAMS[p]) for p in true_persona])
likelihood_to_use = np.clip(np.round(likelihood_raw), 1, 5).astype(int)
likely_to_use = np.where(likelihood_to_use >= 4, "Yes", "No")  # classification target

income_mean, income_std = monthly_income_inr.mean(), monthly_income_inr.std()
income_z = (monthly_income_inr - income_mean) / income_std
taste_z = (likert_cols["taste_importance"] - 3) / 1.0
hygiene_z = (likert_cols["hygiene_importance"] - 3) / 1.0
brand_z = (likert_cols["brand_trust_importance"] - 3) / 1.0
PERSONA_PRICE_OFFSET = {"NRI_Quality": 0.04, "Traveler_Budget": -0.02, "Convenience_Family": 0.00, "Skeptical_LowIntent": -0.05}
persona_offset = np.array([PERSONA_PRICE_OFFSET[p] for p in true_persona])

log_price = (6.05
             + 0.13 * taste_z
             + 0.10 * hygiene_z
             + 0.07 * brand_z
             + 0.14 * income_z
             + persona_offset
             + rng.normal(0, 0.22, size=N))
max_price_per_kg_inr = np.exp(log_price).round(0)
max_price_per_kg_inr = np.clip(max_price_per_kg_inr, 200, 900)

def bucket_price(v):
    if v < 300: return "Below Rs 300"
    if v < 400: return "Rs 300-400"
    if v < 500: return "Rs 401-500"
    if v < 600: return "Rs 501-600"
    return "Above Rs 600"
price_bracket = np.array([bucket_price(v) for v in max_price_per_kg_inr])

QTY_PARAMS = {"NRI_Quality": (5.5, 2.0), "Traveler_Budget": (6.5, 2.5),
              "Convenience_Family": (4.0, 1.8), "Skeptical_LowIntent": (3.0, 1.5)}
min_order_qty_kg = np.array([np.clip(rng.normal(*QTY_PARAMS[p]), 1, 20) for p in true_persona]).round(1)

def bucket_qty(v):
    if v < 3: return "1-2 kg"
    if v < 6: return "3-5 kg"
    if v < 11: return "6-10 kg"
    return "More than 10 kg"
qty_bracket = np.array([bucket_qty(v) for v in min_order_qty_kg])

FREQ_FACTOR = {"Never": 0.05, "Rarely (once or twice a year)": 0.15,
               "Occasionally (every 2-3 months)": 0.45, "Frequently (monthly or more)": 1.0}
freq_factor = np.array([FREQ_FACTOR[f] for f in frequency_sending_food])
monthly_spend_inr = (max_price_per_kg_inr * min_order_qty_kg * freq_factor
                      * rng.lognormal(0, 0.18, size=N)).round(-1)
monthly_spend_inr = np.clip(monthly_spend_inr, 100, 15000)

def bucket_spend(v):
    if v < 500: return "Below Rs 500"
    if v < 1500: return "Rs 500-1,500"
    if v < 3000: return "Rs 1,501-3,000"
    if v < 5000: return "Rs 3,001-5,000"
    return "Above Rs 5,000"
spend_bracket = np.array([bucket_spend(v) for v in monthly_spend_inr])

WILLING_EXTRA_PROBS = {"NRI_Quality": {"Yes": 0.55, "No": 0.20, "Maybe": 0.25},
                        "Traveler_Budget": {"Yes": 0.30, "No": 0.40, "Maybe": 0.30},
                        "Convenience_Family": {"Yes": 0.35, "No": 0.35, "Maybe": 0.30},
                        "Skeptical_LowIntent": {"Yes": 0.10, "No": 0.65, "Maybe": 0.25}}
willing_pay_extra_speed = np.array([rng.choice(list(WILLING_EXTRA_PROBS[p].keys()), p=list(WILLING_EXTRA_PROBS[p].values())) for p in true_persona])

RECOMMEND_PROBS = {"NRI_Quality": {"Yes": 0.75, "No": 0.05, "Maybe": 0.20},
                    "Traveler_Budget": {"Yes": 0.55, "No": 0.10, "Maybe": 0.35},
                    "Convenience_Family": {"Yes": 0.45, "No": 0.15, "Maybe": 0.40},
                    "Skeptical_LowIntent": {"Yes": 0.10, "No": 0.50, "Maybe": 0.40}}
would_recommend = np.array([rng.choice(list(RECOMMEND_PROBS[p].keys()), p=list(RECOMMEND_PROBS[p].values())) for p in true_persona])

# -----------------------------------------------------------------------------
# 7. ASSEMBLE DATAFRAME
# -----------------------------------------------------------------------------
df = pd.DataFrame({
    "respondent_id": [f"R{i+1:04d}" for i in range(N)],
    "true_persona": true_persona,            # hidden ground truth - see note above
    "age": age.astype(int),
    "age_group": age_group,
    "gender": gender,
    "locality": locality,
    "occupation": occupation,
    "monthly_income_inr": monthly_income_inr,
    "income_bracket": income_bracket,
    "household_size_bracket": household_size_bracket,
    "ever_sent_homecooked_food": ever_sent_homecooked_food,
    "segment": segment,
    "heard_of_freeze_dry": heard_of_freeze_dry,
    "used_freeze_dry_before": used_freeze_dry_before,
    "brand_used": brand_used,
    "frequency_sending_food": frequency_sending_food,
})
for opt in METHOD_OPTIONS:
    df[opt] = method_cols[opt]
for opt in DISH_OPTIONS:
    df[opt] = dish_cols[opt]
for opt in OCCASION_OPTIONS:
    df[opt] = occasion_cols[opt]
for opt in CHANNEL_OPTIONS:
    df[opt] = channel_cols[opt]
df["packaging_pref"] = packaging_pref
for trait, vals in likert_cols.items():
    df[trait] = vals
df["likelihood_to_use"] = likelihood_to_use
df["likely_to_use"] = likely_to_use
df["max_price_per_kg_inr"] = max_price_per_kg_inr
df["price_bracket"] = price_bracket
df["min_order_qty_kg"] = min_order_qty_kg
df["qty_bracket"] = qty_bracket
df["monthly_spend_inr"] = monthly_spend_inr
df["spend_bracket"] = spend_bracket
df["willing_pay_extra_speed"] = willing_pay_extra_speed
df["would_recommend"] = would_recommend

# -----------------------------------------------------------------------------
# 8. INJECT OUTLIERS (~3% on key numeric columns)
# -----------------------------------------------------------------------------
def inject_outliers(series, frac=0.03, low_mult=0.2, high_mult=3.0):
    s = series.copy()
    idx = rng.choice(len(s), size=int(len(s) * frac), replace=False)
    for i in idx:
        if rng.random() < 0.5:
            s.iloc[i] = s.iloc[i] * rng.uniform(low_mult, low_mult + 0.1)
        else:
            s.iloc[i] = s.iloc[i] * rng.uniform(high_mult - 0.3, high_mult)
    return s

for col, lo, hi in [("monthly_income_inr", 0.15, 4.0), ("max_price_per_kg_inr", 0.3, 1.8),
                     ("monthly_spend_inr", 0.2, 3.0), ("min_order_qty_kg", 0.3, 2.5)]:
    df[col] = inject_outliers(df[col], frac=0.03, low_mult=lo, high_mult=hi)

df["monthly_income_inr"] = df["monthly_income_inr"].round(-2).clip(10000, 1500000)
df["max_price_per_kg_inr"] = df["max_price_per_kg_inr"].round(0).clip(150, 1200)
df["monthly_spend_inr"] = df["monthly_spend_inr"].round(-1).clip(100, 15000)
df["min_order_qty_kg"] = df["min_order_qty_kg"].round(1).clip(0.5, 30)

# Re-bucket brackets after outlier injection so they stay internally consistent
df["price_bracket"] = df["max_price_per_kg_inr"].apply(bucket_price)
df["qty_bracket"] = df["min_order_qty_kg"].apply(bucket_qty)
df["spend_bracket"] = df["monthly_spend_inr"].apply(bucket_spend)
df["income_bracket"] = df["monthly_income_inr"].apply(bucket_income)

# -----------------------------------------------------------------------------
# 9. INJECT MISSING VALUES (realistic survey non-response, soft fields only)
# -----------------------------------------------------------------------------
def inject_missing(col, frac):
    idx = rng.choice(len(df), size=int(len(df) * frac), replace=False)
    df.loc[df.index[idx], col] = np.nan

inject_missing("locality", 0.02)
inject_missing("household_size_bracket", 0.015)
inject_missing("willing_pay_extra_speed", 0.01)
# brand_used already mostly NaN by design (only used_freeze_dry_before == Yes)

# -----------------------------------------------------------------------------
# 10. SAVE
# -----------------------------------------------------------------------------
df.to_csv("freeze_dry_survey_synthetic.csv", index=False)
print(f"Saved {len(df)} rows, {df.shape[1]} columns -> freeze_dry_survey_synthetic.csv")
print(df["true_persona"].value_counts())
print(df["segment"].value_counts())
print(df["likely_to_use"].value_counts(normalize=True))

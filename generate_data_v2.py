# =============================================================================
# Synthetic Business Dataset Generator (v2) - with noise, outliers, skewness
# Freeze-Dried Home-Cooked Meal Business (Hyderabad) - Sales Pipeline
# =============================================================================
# Produces: leads.csv, orders.csv, batches.csv
# Designed for EDA in Excel/SAS JMP (correlation graphs, descriptive stats).
#
# Deliberate realism built in:
#   - Lognormal (right-skewed) distributions for order weight, price, revenue
#   - ~3% outlier injection on weight, price, wait time, packaging cost
#   - A genuine volume-discount effect: price/kg drops slightly as order size
#     grows -> a clear, explainable negative correlation for the report
#   - Segment-level separation (NRI_Family pays a premium baseline vs
#     Traveler_Student) -> clean visual split for segment charts
#
# Run directly in Google Colab - only needs numpy, pandas, matplotlib
# (all pre-installed in Colab).
# =============================================================================

import numpy as np
import pandas as pd
from datetime import timedelta

RNG_SEED = 7
rng = np.random.default_rng(RNG_SEED)

# -----------------------------------------------------------------------------
# 1. TIME WINDOW & SEASONALITY (12-month launch year)
# -----------------------------------------------------------------------------
months = pd.date_range("2026-08-01", periods=12, freq="MS")
seasonality = {
    "2026-08": 1.5, "2026-09": 1.35, "2026-10": 1.15, "2026-11": 1.25,
    "2026-12": 1.45, "2027-01": 1.30, "2027-02": 0.75, "2027-03": 0.70,
    "2027-04": 0.70, "2027-05": 0.80, "2027-06": 0.90, "2027-07": 1.20,
}
BASE_LEADS_PER_MONTH = 38
SEGMENTS = ["NRI_Family", "Traveler_Student"]
SEGMENT_SHARE = {"NRI_Family": 0.55, "Traveler_Student": 0.45}
LEAD_SOURCES = ["Instagram_Ad", "Referral_WordOfMouth", "Google_Search", "Walk_In", "Facebook_Group"]
SOURCE_PROB = [0.30, 0.35, 0.15, 0.10, 0.10]
LOST_REASONS = ["Price_Too_High", "Min_Order_Not_Met", "Slow_Response", "Chose_Competitor", "Timing_Mismatch"]
LOST_PROB = [0.30, 0.25, 0.15, 0.15, 0.15]
DISH_CATEGORIES = ["Dal_Rice_Combo", "Rotis_Sabzi", "Biryani_PulaoVariety", "Sweets_Snacks", "Mixed_Thali"]

# -----------------------------------------------------------------------------
# 2. LEADS (skewed order-size distributions per segment, via lognormal)
# -----------------------------------------------------------------------------
QTY_LOGPARAMS = {"NRI_Family": (1.15, 0.42), "Traveler_Student": (1.55, 0.48)}  # log-space (mean, sd)
BASE_PRICE = {"NRI_Family": 500, "Traveler_Student": 450}   # baseline Rs/kg before volume discount
VOLUME_DISCOUNT_RATE = 6.0   # Rs/kg discount per extra kg above 3kg baseline

leads = []
lead_counter = 1
for m in months:
    key = m.strftime("%Y-%m")
    n_leads = max(1, int(rng.poisson(BASE_LEADS_PER_MONTH * seasonality[key])))
    for _ in range(n_leads):
        seg = rng.choice(SEGMENTS, p=[SEGMENT_SHARE[s] for s in SEGMENTS])
        day = rng.integers(1, 29)
        lead_date = m + timedelta(days=int(day) - 1)
        source = rng.choice(LEAD_SOURCES, p=SOURCE_PROB)

        # Skewed (lognormal) order size -> right-skewed, realistic long tail of bulk orders
        quoted_kg = round(float(rng.lognormal(*QTY_LOGPARAMS[seg])), 1)
        quoted_kg = float(np.clip(quoted_kg, 1.5, 25))

        # Volume discount: larger orders get a lower price/kg, plus noise
        discount = VOLUME_DISCOUNT_RATE * max(0, quoted_kg - 3)
        quoted_price = BASE_PRICE[seg] - discount + rng.normal(0, 25)
        quoted_price = round(float(np.clip(quoted_price, 280, 650)), 0)

        conv_prob = 0.27 if seg == "NRI_Family" else 0.19
        if quoted_kg < 3:
            conv_prob *= 0.6
        converted = rng.random() < conv_prob
        lost_reason = "" if converted else rng.choice(LOST_REASONS, p=LOST_PROB)

        leads.append({
            "lead_id": f"L{lead_counter:05d}", "lead_date": lead_date.date(), "month": key,
            "segment": seg, "lead_source": source, "quoted_kg": quoted_kg,
            "quoted_price_per_kg": quoted_price, "converted": converted, "lost_reason": lost_reason,
        })
        lead_counter += 1

leads_df = pd.DataFrame(leads)

# -----------------------------------------------------------------------------
# 3. OUTLIER INJECTION on leads (bulk-order outliers, price-shock outliers)
# -----------------------------------------------------------------------------
def inject_outliers(series, frac, low_mult, high_mult, rng_local):
    s = series.copy()
    idx = rng_local.choice(len(s), size=int(len(s) * frac), replace=False)
    for i in idx:
        if rng_local.random() < 0.5:
            s.iloc[i] = s.iloc[i] * rng_local.uniform(*low_mult)
        else:
            s.iloc[i] = s.iloc[i] * rng_local.uniform(*high_mult)
    return s

leads_df["quoted_kg"] = inject_outliers(leads_df["quoted_kg"], 0.03, (0.4, 0.6), (1.8, 2.6), rng).round(1).clip(1, 30)
leads_df["quoted_price_per_kg"] = inject_outliers(leads_df["quoted_price_per_kg"], 0.025, (0.55, 0.7), (1.3, 1.6), rng).round(0).clip(250, 900)

# -----------------------------------------------------------------------------
# 4. CUSTOMER / REPEAT LOGIC for converted leads
# -----------------------------------------------------------------------------
converted_df = leads_df[leads_df["converted"]].copy().sort_values("lead_date").reset_index(drop=True)
customer_pool = {s: [] for s in SEGMENTS}
customer_counter = 1
REPEAT_PROB = {"NRI_Family": 0.32, "Traveler_Student": 0.10}
last_order_date = {}
customer_ids, repeat_flag, days_since_last = [], [], []

for _, row in converted_df.iterrows():
    seg = row["segment"]
    pool = customer_pool[seg]
    is_repeat, cust_id = False, None
    if pool and rng.random() < REPEAT_PROB[seg]:
        cust_id, is_repeat = rng.choice(pool), True
    if cust_id is None:
        cust_id = f"C{customer_counter:05d}"
        customer_counter += 1
        customer_pool[seg].append(cust_id)
    customer_ids.append(cust_id)
    repeat_flag.append(is_repeat)
    days_since_last.append((row["lead_date"] - last_order_date[cust_id]).days if is_repeat and cust_id in last_order_date else np.nan)
    last_order_date[cust_id] = row["lead_date"]

converted_df["customer_id"] = customer_ids
converted_df["repeat_customer"] = repeat_flag
converted_df["days_since_last_order"] = days_since_last

# -----------------------------------------------------------------------------
# 5. ORDER INTAKE -> BATCH ASSIGNMENT (capacity-constrained, single small unit)
# -----------------------------------------------------------------------------
BATCH_CAPACITY_MIN, BATCH_CAPACITY_MAX = 5, 10
CYCLE_DAYS = 4

converted_df = converted_df.sort_values("lead_date").reset_index(drop=True)
converted_df["order_id"] = [f"O{i+1:05d}" for i in range(len(converted_df))]
converted_df["dish_category"] = rng.choice(DISH_CATEGORIES, size=len(converted_df))

start_date = converted_df["lead_date"].min()
batch_records = []
batch_counter = 1
next_batch_available_date = start_date
order_batch_map, order_wait_days, order_batch_date = {}, {}, {}

queue = [(r["order_id"], r["lead_date"], r["quoted_kg"]) for _, r in converted_df.iterrows()]
while queue:
    batch_cap = round(rng.uniform(BATCH_CAPACITY_MIN, BATCH_CAPACITY_MAX), 1)
    batch_load, batch_orders, remaining_queue = 0.0, [], []
    batch_open_date = next_batch_available_date
    for (oid, ldate, wt) in queue:
        if ldate <= batch_open_date + timedelta(days=3) and batch_load + wt <= batch_cap:
            batch_load += wt
            batch_orders.append((oid, ldate, wt))
        else:
            remaining_queue.append((oid, ldate, wt))
    if not batch_orders:
        oid, ldate, wt = remaining_queue.pop(0)
        batch_orders, batch_load, batch_open_date = [(oid, ldate, wt)], wt, ldate

    batch_close_date = batch_open_date + timedelta(days=CYCLE_DAYS)
    for (oid, ldate, wt) in batch_orders:
        order_batch_map[oid] = f"B{batch_counter:04d}"
        order_wait_days[oid] = max(0, (batch_open_date - ldate).days)
        order_batch_date[oid] = batch_close_date

    batch_records.append({
        "batch_id": f"B{batch_counter:04d}", "batch_open_date": batch_open_date,
        "batch_close_date": batch_close_date, "batch_capacity_kg": batch_cap,
        "batch_load_kg": round(batch_load, 1), "utilization_pct": round(100 * batch_load / batch_cap, 1),
        "n_orders": len(batch_orders),
    })
    batch_counter += 1
    next_batch_available_date = batch_close_date
    queue = remaining_queue

batches_df = pd.DataFrame(batch_records)

# Outlier batches: a few rushed near-empty batches, a few squeezed-in overloaded ones
batches_df["utilization_pct"] = inject_outliers(batches_df["utilization_pct"], 0.04, (0.25, 0.45), (1.05, 1.15), rng).clip(15, 100).round(1)

converted_df["batch_id"] = converted_df["order_id"].map(order_batch_map)
converted_df["wait_days_before_batch"] = converted_df["order_id"].map(order_wait_days)
converted_df["batch_close_date"] = pd.to_datetime(converted_df["order_id"].map(order_batch_date))

# Outlier wait times: occasional long supply-chain delays
converted_df["wait_days_before_batch"] = inject_outliers(
    converted_df["wait_days_before_batch"].astype(float), 0.03, (1.0, 1.0), (2.5, 4.0), rng
).round(0).clip(0, 25).astype(int)

# -----------------------------------------------------------------------------
# 6. OUTPUT WEIGHT, DELIVERY, REVENUE (with noise + outliers)
# -----------------------------------------------------------------------------
reduction_ratio = rng.uniform(0.22, 0.30, size=len(converted_df))
converted_df["output_weight_kg"] = (converted_df["quoted_kg"] * reduction_ratio).round(2)
# A few batches with poor yield (spoilage / quality issue) or unusually high retention
converted_df["output_weight_kg"] = inject_outliers(converted_df["output_weight_kg"], 0.03, (0.5, 0.7), (1.3, 1.5), rng).round(2).clip(0.2, 10)

converted_df["delivery_date"] = converted_df["batch_close_date"] + pd.to_timedelta(rng.integers(1, 4, size=len(converted_df)), unit="D")
converted_df["invoice_amount"] = (converted_df["quoted_kg"] * converted_df["quoted_price_per_kg"]).round(0)

converted_df["packaging_cost"] = (converted_df["output_weight_kg"] * rng.uniform(40, 60, size=len(converted_df))).round(0)
converted_df["packaging_cost"] = inject_outliers(converted_df["packaging_cost"], 0.025, (0.6, 0.8), (1.6, 2.2), rng).round(0).clip(10, 500)

converted_df["payment_status"] = rng.choice(["Paid", "Pending", "Refunded"], size=len(converted_df), p=[0.93, 0.05, 0.02])

# -----------------------------------------------------------------------------
# 7. SAVE
# -----------------------------------------------------------------------------
orders_df = converted_df[[
    "order_id", "lead_id", "customer_id", "segment", "lead_date", "lead_source",
    "dish_category", "quoted_kg", "quoted_price_per_kg", "invoice_amount",
    "batch_id", "batch_close_date", "wait_days_before_batch",
    "output_weight_kg", "packaging_cost", "delivery_date", "payment_status",
    "repeat_customer", "days_since_last_order"
]].rename(columns={"batch_close_date": "batch_processed_date"})

leads_df.to_csv("leads.csv", index=False)
orders_df.to_csv("orders.csv", index=False)
batches_df.to_csv("batches.csv", index=False)

print(f"leads.csv   -> {leads_df.shape}")
print(f"orders.csv  -> {orders_df.shape}  (conversion rate: {leads_df['converted'].mean():.1%})")
print(f"batches.csv -> {batches_df.shape}  (avg utilization: {batches_df['utilization_pct'].mean():.1f}%)")
print()
print("Key correlations (orders.csv):")
print(orders_df[["quoted_kg", "quoted_price_per_kg", "invoice_amount", "wait_days_before_batch", "output_weight_kg", "packaging_cost"]].corr().round(2))
print()
print("Skewness check (orders.csv):")
print(orders_df[["quoted_kg", "invoice_amount", "packaging_cost"]].skew().round(2))

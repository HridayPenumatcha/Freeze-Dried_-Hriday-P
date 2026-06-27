import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("freeze_dry_survey_synthetic.csv")
print("Shape:", df.shape)

# =========================================================
# 1. CLASSIFICATION: predict likely_to_use from attitudes + demographics
# =========================================================
likert_feats = ["taste_importance", "price_importance", "shelf_life_importance",
                 "hygiene_importance", "convenience_importance", "brand_trust_importance",
                 "variety_importance", "speed_importance"]
clf_features = likert_feats + ["age", "monthly_income_inr", "min_order_qty_kg"]
X = df[clf_features].fillna(df[clf_features].median())
y = LabelEncoder().fit_transform(df["likely_to_use"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("\n=== CLASSIFICATION (RandomForest: likely_to_use) ===")
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, target_names=["No", "Yes"]))

# =========================================================
# 2. REGRESSION: predict max_price_per_kg_inr
# =========================================================
reg_features = likert_feats + ["age", "monthly_income_inr"]
Xr = df[reg_features].fillna(df[reg_features].median())
yr = df["max_price_per_kg_inr"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.25, random_state=42)
reg = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
reg.fit(Xr_train, yr_train)
pred_r = reg.predict(Xr_test)
print("\n=== REGRESSION (RandomForest: max_price_per_kg_inr) ===")
print("R2:", round(r2_score(yr_test, pred_r), 3))
print("MAE: Rs", round(mean_absolute_error(yr_test, pred_r), 1))

# =========================================================
# 3. CLUSTERING: KMeans on attitude scores, validate against true_persona
# =========================================================
Xc = StandardScaler().fit_transform(df[likert_feats].fillna(df[likert_feats].median()))
km = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = km.fit_predict(Xc)
sil = silhouette_score(Xc, cluster_labels)
print("\n=== CLUSTERING (KMeans, k=4 on attitude scores) ===")
print("Silhouette score:", round(sil, 3))
print(pd.crosstab(df["true_persona"], cluster_labels))

pca = PCA(n_components=2)
coords = pca.fit_transform(Xc)
plt.figure(figsize=(7, 6))
personas = df["true_persona"].unique()
colors = ["#1D9E75", "#D85A30", "#378ADD", "#888780"]
for persona, color in zip(personas, colors):
    mask = df["true_persona"] == persona
    plt.scatter(coords[mask, 0], coords[mask, 1], label=persona, alpha=0.6, s=25, color=color)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("Customer segments (PCA of attitude scores), colored by true persona")
plt.legend()
plt.tight_layout()
plt.savefig("segment_visualization.png", dpi=150)
print("Saved segment_visualization.png")

# =========================================================
# 4. ASSOCIATION RULE MINING: dish/occasion/channel baskets
# =========================================================
basket_cols = [c for c in df.columns if c.startswith(("dish_", "occasion_", "channel_"))]
basket = df[basket_cols].astype(bool)
freq_items = apriori(basket, min_support=0.08, use_colnames=True)
rules = association_rules(freq_items, metric="lift", min_threshold=1.2)
rules = rules.sort_values("lift", ascending=False)
print("\n=== ASSOCIATION RULE MINING (Apriori) ===")
print(f"Frequent itemsets found: {len(freq_items)} | Rules (lift > 1.2): {len(rules)}")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(8).to_string())

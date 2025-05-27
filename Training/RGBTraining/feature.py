import xgboost as xgb
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# === Human-readable feature descriptions ===
feature_descriptions = {
    'f0': 'Red channel intensity (% change from baseline)',
    'f1': 'Green channel intensity (% change from baseline)',
    'f2': 'Blue channel intensity (% change from baseline)',
    'f3': 'Mean of R, G, B % changes',
    'f4': 'Absolute mean of R, G, B % changes'
}

# === Load Your Trained XGBoost Model ===
model = xgb.XGBRegressor()
model.load_model(r"E:\smartdetectai\models\hg_xgboost.json")

# === Extract Feature Importances by 'gain' ===
booster = model.get_booster()
importance_dict = booster.get_score(importance_type='gain')
print("Raw importance keys from model:", importance_dict.keys())

# Fill in missing features (if some were dropped)
all_model_features = ['f0', 'f1', 'f2', 'f3', 'f4']
importance_dict_full = {f: importance_dict.get(f, 0.0) for f in all_model_features}

# Convert to DataFrame
importance_df = pd.DataFrame({
    'Feature': list(importance_dict_full.keys()),
    'Importance': list(importance_dict_full.values()),
    'Description': [feature_descriptions[f] for f in all_model_features]
})

# Normalize importance
importance_df['Importance'] = 100 * importance_df['Importance'] / importance_df['Importance'].sum()

# Sort for plotting
importance_df = importance_df.sort_values(by='Importance', ascending=True)

# === Print Feature Info ===
print("\n📊 Feature Descriptions and Importance (% Gain):\n")
for _, row in importance_df.iterrows():
    print(f"- {row['Feature']:4s} → {row['Description']}\n  Importance: {row['Importance']:.2f}%\n")

# === Plot
plt.figure(figsize=(9, 6))
sns.barplot(x='Importance', y='Description', data=importance_df, hue='Feature', palette='coolwarm', legend=False)
plt.title("Feature Importance (Gain) - XGBoost", fontsize=14)
plt.xlabel("Relative Importance (%)")
plt.ylabel("Feature Description")
plt.tight_layout()
plt.savefig("feature_importance.png")

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── Realistic synthetic dataset matching the Kaggle House Rent Dataset ────────
def generate_dataset(n=4746, seed=42):
    np.random.seed(seed)

    cities    = ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai", "Hyderabad"]
    city_base = {"Mumbai": 42000, "Delhi": 28000, "Bangalore": 22000,
                 "Kolkata": 12000, "Chennai": 14000, "Hyderabad": 16000}
    furnish_mult = {"Furnished": 1.35, "Semi-Furnished": 1.12, "Unfurnished": 1.0}
    area_mult    = {"Super Area": 1.0, "Carpet Area": 1.08, "Built Area": 0.97}
    tenant_mult  = {"Bachelors/Family": 1.0, "Family": 1.05, "Bachelors": 0.95}

    city_counts = [972, 868, 886, 605, 524, 891]
    cities_col, bhk_col, rent_col, size_col = [], [], [], []
    floor_col, area_type_col, furnish_col, tenant_col, bath_col = [], [], [], [], []

    for city, count in zip(cities, city_counts):
        base = city_base[city]
        for _ in range(count):
            bhk       = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.12, 0.42, 0.31, 0.10, 0.03, 0.02])
            size      = int(np.random.normal(200 + bhk * 250, 120 + bhk * 60))
            size      = max(150, min(size, 8000))
            bathroom  = min(bhk + np.random.choice([0, 1], p=[0.6, 0.4]), 10)
            floor     = int(np.random.exponential(3))
            floor     = min(floor, 30)
            area_type = np.random.choice(["Super Area", "Carpet Area", "Built Area"], p=[0.55, 0.35, 0.10])
            furnish   = np.random.choice(["Unfurnished", "Semi-Furnished", "Furnished"], p=[0.42, 0.38, 0.20])
            tenant    = np.random.choice(["Bachelors/Family", "Family", "Bachelors"], p=[0.55, 0.25, 0.20])

            rent = (base
                    * (0.6 + bhk * 0.22)
                    * (0.7 + size / 2000)
                    * furnish_mult[furnish]
                    * area_mult[area_type]
                    * tenant_mult[tenant]
                    * (1 + floor * 0.01)
                    * np.random.normal(1.0, 0.12))
            rent = max(1200, min(int(rent), 3500000))

            cities_col.append(city); bhk_col.append(bhk); rent_col.append(rent)
            size_col.append(size); floor_col.append(floor); area_type_col.append(area_type)
            furnish_col.append(furnish); tenant_col.append(tenant); bath_col.append(int(bathroom))

    df = pd.DataFrame({
        "BHK": bhk_col, "Rent": rent_col, "Size": size_col,
        "Floor": floor_col, "Area Type": area_type_col,
        "City": cities_col, "Furnishing Status": furnish_col,
        "Tenant Preferred": tenant_col, "Bathroom": bath_col
    })
    return df


def train_and_save_model(output_path="rent_model.pkl"):
    print("Generating dataset...")
    df = generate_dataset()

    # ── Encode categoricals ──────────────────────────────────────────────────
    le_city    = LabelEncoder().fit(df["City"])
    le_furnish = LabelEncoder().fit(df["Furnishing Status"])
    le_area    = LabelEncoder().fit(df["Area Type"])
    le_tenant  = LabelEncoder().fit(df["Tenant Preferred"])

    df["City_enc"]    = le_city.transform(df["City"])
    df["Furnish_enc"] = le_furnish.transform(df["Furnishing Status"])
    df["Area_enc"]    = le_area.transform(df["Area Type"])
    df["Tenant_enc"]  = le_tenant.transform(df["Tenant Preferred"])

    FEATURES = ["BHK", "Size", "Floor", "Area_enc", "City_enc", "Furnish_enc", "Tenant_enc", "Bathroom"]
    X = df[FEATURES].values
    y = df["Rent"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # ── Train Random Forest (tuned) ──────────────────────────────────────────
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R² = {r2:.4f}  |  RMSE = ₹{rmse:,.0f}")

    # ── City average stats for insights ─────────────────────────────────────
    city_avgs = df.groupby("City")["Rent"].mean().to_dict()
    stats = {f"city_avg_{city}": avg for city, avg in city_avgs.items()}
    stats["r2"]   = r2
    stats["rmse"] = rmse

    # ── Save ─────────────────────────────────────────────────────────────────
    payload = {
        "model":      model,
        "le_city":    le_city,
        "le_furnish": le_furnish,
        "le_area":    le_area,
        "le_tenant":  le_tenant,
        "stats":      stats,
        "features":   FEATURES,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Model saved → {output_path}")
    return payload


if __name__ == "__main__":
    train_and_save_model()

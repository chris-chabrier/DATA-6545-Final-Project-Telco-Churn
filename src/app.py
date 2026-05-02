from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model + feature structure
model = joblib.load("gradient_boosting_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

def feature_engineering(data):
    df = pd.DataFrame(data if isinstance(data, list) else [data])

    # ---- Basic type cleaning ----
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # ---- Engineered features ----
    df["AvgChargesPerTenure"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["NewCustomer"] = (df["tenure"] <= 12).astype(int)
    df["HighMonthlyCharges"] = (df["MonthlyCharges"] > 70.35).astype(int)
    if "Contract" in df.columns:
        df["MonthToMonthContract"] = (df["Contract"] == "Month-to-month").astype(int)
    else:
        df["MonthToMonthContract"] = 0

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["ServiceCount"] = 0
    for col in service_columns:
        if col in df.columns:
            df["ServiceCount"] += (df[col] == "Yes").astype(int)

    # Drop ID fields if present
    df = df.drop(columns=["customerID"], errors="ignore")

    # One-hot encoding
    df_encoded = pd.get_dummies(df)

    # Align with training columns
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    return df_encoded


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

required_fields = ["tenure", "MonthlyCharges", "TotalCharges"]
threshold = 0.35

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Transform input into model-ready format
        X = feature_engineering(data)

        if X.isnull().any().any():
            return jsonify({"error": "Invalid input: contains non-numeric values"}), 400

        # Predict probability + class
        prob = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]

        # Threshold logic
        risk_label = "High Risk" if prob >= threshold else "Low Risk"

        return jsonify({
            "prediction": int(pred),
            "churn_probability": float(round(prob, 4)),
            "risk_label": risk_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json()

        if not isinstance(data, list):
            return jsonify({"error": "Input must be a list of records"}), 400

        for i, record in enumerate(data):
            for field in required_fields:
                if field not in record:
                    return jsonify({"error": f"Record {i} missing required field: {field}"}), 400

        X = feature_engineering(data)

        if X.isnull().any().any():
            return jsonify({"error": "Invalid input: contains non-numeric values"}), 400

        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        results = []

        for i in range(len(preds)):
            prob = probs[i]
            risk_label = "High Risk" if prob >= threshold else "Low Risk"

            results.append({
                "prediction": int(preds[i]),
                "churn_probability": float(round(prob, 4)),
                "risk_label": risk_label
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

bundle = joblib.load("loan_scoring_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
le_employment = bundle["label_encoder_employment"]
le_decision = bundle["label_encoder_decision"]
feature_cols = bundle["feature_cols"]

EMPLOYMENT_TYPES = list(le_employment.classes_)
DECISION_LABELS = {
    "approved": {"text": "Зээл олгоно", "color": "success", "icon": "✅"},
    "manual_review": {"text": "Гараар шалгах шаардлагатай", "color": "warning", "icon": "⚠️"},
    "rejected": {"text": "Зээл олгохгүй", "color": "danger", "icon": "❌"},
}


def build_features(monthly_income, employment_years, requested_amount, employment_type_encoded):
    amount_to_income_ratio = requested_amount / monthly_income if monthly_income > 0 else 0
    annual_dti = requested_amount / (monthly_income * 12) if monthly_income > 0 else 0
    log_income = np.log1p(monthly_income)
    log_amount = np.log1p(requested_amount)
    return pd.DataFrame(
        [[monthly_income, employment_years, requested_amount,
          amount_to_income_ratio, annual_dti, log_income, log_amount, employment_type_encoded]],
        columns=feature_cols,
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", employment_types=EMPLOYMENT_TYPES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        monthly_income = float(request.form["monthly_income"])
        employment_years = float(request.form["employment_years"])
        requested_amount = float(request.form["requested_amount"])
        employment_type = request.form["employment_type"]

        if monthly_income <= 0 or requested_amount <= 0 or employment_years < 0:
            return render_template(
                "index.html",
                employment_types=EMPLOYMENT_TYPES,
                error="Оруулсан утгууд буруу байна. Бүх утгыг зөв оруулна уу.",
            )

        emp_encoded = le_employment.transform([employment_type])[0]
        X = build_features(monthly_income, employment_years, requested_amount, emp_encoded)
        X_scaled = scaler.transform(X)

        pred_encoded = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        pred_label = le_decision.inverse_transform([pred_encoded])[0]

        class_order = list(le_decision.classes_)
        proba_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(class_order, proba)}

        info = DECISION_LABELS.get(pred_label, {"text": pred_label, "color": "secondary", "icon": "ℹ️"})
        score = round(proba_dict.get("approved", 0), 1)

        return render_template(
            "result.html",
            decision=pred_label,
            decision_info=info,
            proba=proba_dict,
            score=score,
            monthly_income=monthly_income,
            employment_years=employment_years,
            requested_amount=requested_amount,
            employment_type=employment_type,
            amount_to_income=round(requested_amount / monthly_income, 2) if monthly_income > 0 else 0,
        )
    except ValueError as e:
        return render_template(
            "index.html",
            employment_types=EMPLOYMENT_TYPES,
            error=f"Оруулсан утга буруу байна: {e}",
        )
    except Exception as e:
        return render_template(
            "index.html",
            employment_types=EMPLOYMENT_TYPES,
            error=f"Алдаа гарлаа: {e}",
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        monthly_income = float(data["monthly_income"])
        employment_years = float(data["employment_years"])
        requested_amount = float(data["requested_amount"])
        employment_type = data["employment_type"]

        emp_encoded = le_employment.transform([employment_type])[0]
        X = scaler.transform(build_features(monthly_income, employment_years, requested_amount, emp_encoded))

        pred_encoded = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        pred_label = le_decision.inverse_transform([pred_encoded])[0]
        class_order = list(le_decision.classes_)
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(class_order, proba)}

        return jsonify({"decision": pred_label, "probabilities": proba_dict, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)

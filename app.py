from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import joblib
import os
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# =====================================================
# Create Flask App
# =====================================================

app = Flask(__name__)
app.secret_key = "brain_stroke_prediction_project"

# =====================================================
# Load Machine Learning Model
# =====================================================

model = joblib.load("stroke_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# =====================================================
# Login Page
# =====================================================

@app.route("/")
def login():
    return render_template("login.html")


# =====================================================
# Login Validation
# =====================================================

@app.route("/login", methods=["POST"])
def login_user():

    username = request.form["username"]
    password = request.form["password"]

    if username == "admin" and password == "admin123":
        session["user"] = username
        return redirect(url_for("dashboard"))

    return "Invalid Username or Password"


# =====================================================
# Dashboard
# =====================================================

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("dashboard.html")


# =====================================================
# Prediction Page
# =====================================================

@app.route("/predict")
def predict():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("predict.html")

# =====================================================
# Prediction Result
# =====================================================

@app.route("/result", methods=["POST"])
def result():

    if "user" not in session:
        return redirect(url_for("login"))

    try:

        # -------------------------
        # Read Form Data
        # -------------------------

        patient_name = request.form["name"]

        gender = request.form["gender"]
        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        ever_married = request.form["ever_married"]
        work_type = request.form["work_type"]
        residence = request.form["Residence_type"]
        glucose = float(request.form["avg_glucose_level"])
        bmi = float(request.form["bmi"])
        smoking = request.form["smoking_status"]

        # -------------------------
        # Create DataFrame
        # -------------------------

        input_data = {
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "avg_glucose_level": glucose,
            "bmi": bmi
        }

        df = pd.DataFrame([input_data])

        df["gender"] = gender
        df["ever_married"] = ever_married
        df["work_type"] = work_type
        df["Residence_type"] = residence
        df["smoking_status"] = smoking

        # -------------------------
        # Convert Categories
        # -------------------------

        df = pd.get_dummies(df)

        # Match training columns
        df = df.reindex(columns=model_columns, fill_value=0)

        # -------------------------
        # Prediction
        # -------------------------

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        risk = round(probability * 100, 2)

        if prediction == 0:
            level = "Low Risk"
            color = "green"

        else:
            if risk < 50:
                level = "Medium Risk"
                color = "orange"
            else:
                level = "High Risk"
                color = "red"

        # -------------------------
        # Save Report in Session
        # -------------------------

        session["report"] = {
            "patient": patient_name,
            "age": age,
            "gender": gender,
            "risk": risk,
            "level": level
        }

        return render_template(
            "result.html",
            patient=patient_name,
            age=age,
            gender=gender,
            risk=risk,
            level=level,
            color=color
        )

    except Exception as e:
        return f"<h2>Error:</h2><pre>{e}</pre>"
    
    # =====================================================
# Download PDF Report
# =====================================================

@app.route("/download")
def download():

    if "user" not in session:
        return redirect(url_for("login"))

    if "report" not in session:
        return redirect(url_for("predict"))

    report = session["report"]

    # Create reports folder if it doesn't exist
    if not os.path.exists("reports"):
        os.makedirs("reports")

    pdf_path = os.path.join("reports", "Brain_Stroke_Report.pdf")

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph("<b>Brain Stroke Prediction Report</b>", styles["Title"])
    )

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(
        Paragraph(f"<b>Patient Name :</b> {report['patient']}", styles["BodyText"])
    )

    story.append(
        Paragraph(f"<b>Age :</b> {report['age']}", styles["BodyText"])
    )

    story.append(
        Paragraph(f"<b>Gender :</b> {report['gender']}", styles["BodyText"])
    )

    story.append(
        Paragraph(f"<b>Risk Percentage :</b> {report['risk']}%", styles["BodyText"])
    )

    story.append(
        Paragraph(f"<b>Risk Level :</b> {report['level']}", styles["BodyText"])
    )

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(
        Paragraph("<b>Health Recommendations</b>", styles["Heading2"])
    )

    tips = [
        "Maintain a healthy diet.",
        "Exercise regularly.",
        "Avoid smoking.",
        "Limit alcohol consumption.",
        "Monitor blood pressure.",
        "Monitor blood sugar.",
        "Maintain healthy body weight.",
        "Visit your doctor regularly."
    ]

    for tip in tips:
        story.append(
            Paragraph("• " + tip, styles["BodyText"])
        )

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(
        Paragraph(
            "Generated On : " +
            datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            styles["BodyText"]
        )
    )

    doc.build(story)

    return send_file(pdf_path, as_attachment=True)


# =====================================================
# Logout
# =====================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# =====================================================
# Run Flask
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
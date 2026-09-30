import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict the final grade (G3).")

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.subheader("Student Details")

# Input sliders
Studytime = st.slider("Studytime (weekly)", 1, 4, 2,
    help="1 = <2 hours, 2 = 2-5 hours, 3 = 5-10 hours, 4 = >10 hours")

Failures = st.slider("Failures (past)", 0, 4, 0,
    help="Number of times failed in past classes")

Absences = st.slider("Absences", 0, 50, 5,
    help="Number of school days missed")

G1 = st.slider("G1 (First Test Score)", 0, 20, 10,
    help="Score in first period test (0-20)")

G2 = st.slider("G2 (Second Test Score)", 0, 20, 10,
    help="Score in second period test (0-20)")

if st.button("🔮 Predict Final Score"):
    input_array = np.array([[Studytime, Failures, Absences, G1, G2]])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    # Clamp prediction between 0 and 20
    prediction = max(0, min(20, prediction))
    percentage = (prediction / 20) * 100

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.metric("Predicted Final Score", f"{prediction:.2f} / 20", f"{percentage:.1f}%")

    # Clear result based on percentage
    if percentage >= 50:
        st.success(f"✅ **PASS** — Score: {prediction:.2f}/20 ({percentage:.1f}%)")
        st.info("📈 You need at least **50% (10/20)** to pass. You are above the threshold.")
    elif percentage >= 40:
        st.warning(f"⚠️ **BORDERLINE** — Score: {prediction:.2f}/20 ({percentage:.1f}%)")
        st.info("💡 You need at least **50% (10/20)** to pass. Extra effort needed.")
    else:
        st.error(f"❌ **FAIL** — Score: {prediction:.2f}/20 ({percentage:.1f}%)")
        st.info("📚 You need at least **50% (10/20)** to pass. Serious improvement needed.")

    # Suggestions
    st.markdown("---")
    st.subheader("💡 Improvement Suggestions")

    if G1 < 8 or G2 < 8:
        st.warning("📚 G1/G2 scores are low. Focus on upcoming tests.")
    if Studytime <= 1:
        st.warning("⏰ Study time is very low. Increase weekly study hours.")
    if Failures >= 2:
        st.warning("⚠️ Multiple past failures. Consider extra coaching.")
    if Absences >= 15:
        st.warning("🚫 High absences. Attend classes regularly.")
    if G1 >= 12 and G2 >= 12 and Studytime >= 3 and Failures == 0 and Absences < 5:
        st.success("🌟 Excellent profile! Keep up the good work.")

# ================= PASS/FAIL THRESHOLD TABLE =================
st.markdown("---")
st.subheader("📖 Pass / Fail Threshold Guide")

st.write("**You need at least 50% (10/20) to pass.** Based on prediction percentage:")

threshold_data = {
    "Category": ["🌟 Excellent", "✅ Pass", "⚠️ Borderline", "❌ Fail"],
    "Prediction %": ["80% and above", "50% - 79%", "40% - 49%", "Below 40%"],
    "Score (out of 20)": ["16 - 20", "10 - 15.9", "8 - 9.9", "0 - 7.9"],
    "Result": ["Top performance", "Pass ✅", "High risk", "Fail ❌"]
}

st.table(threshold_data)

# Minimum passing criteria
st.markdown("---")
st.subheader("🎯 Minimum Passing Criteria")

st.info("""
**To PASS, you need:**
- **Predicted Score:** At least **10 / 20**
- **Prediction Percentage:** At least **50%**

If your prediction is **50% or above**, you will **PASS**.  
If it is **below 50%**, there is a risk of **FAIL**.
""")

# ================= REFERENCE TABLE =================
st.markdown("---")
st.subheader("📚 G1 & G2 Reference Guide")

st.write("Approximate G1 & G2 score ranges to expect final result:")

reference_data = {
    "Category": ["Excellent 🌟", "Good ✅", "Average ⚠️", "At Risk 🚫"],
    "G1 Score": ["16-20", "12-15", "8-11", "0-7"],
    "G2 Score": ["16-20", "12-15", "8-11", "0-7"],
    "Expected G3": ["16-20 (Pass)", "12-15 (Pass)", "10-11 (Borderline)", "0-9 (Fail)"]
}

st.table(reference_data)
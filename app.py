import streamlit as st
from datetime import date
import db
import pandas as pd
import posture_expression




st.set_page_config(page_title="Hybrid AI Fitness Tracker", layout="wide")
db.init_db()

st.title("💪 Hybrid AI Fitness & Wellness Tracker")

if st.button("🧠 Start AI Posture & Emotion Detection"):
    st.write("Initializing camera... press 'q' in camera window to stop.")
    # Run detector and return a summary dict when user quits camera
    summary = posture_expression.detect_webcam_feed()
    # Show summary
    st.success("Session finished — summary saved.")
    st.write("Session Summary:")
    st.json(summary)
    # optionally show some quick metrics nicely
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration (s)", summary.get("duration_seconds", 0))
    col2.metric("Frames", summary.get("frames", 0))
    col3.metric("Dominant emotion", summary.get("dominant_emotion", "Unknown"))

# --- Log Form ---
st.sidebar.header("📝 Log Your Day")
with st.sidebar.form("log_form"):
    log_date = st.date_input("Date", value=date.today())
    workout_type = st.selectbox("Workout Type", ["None", "Cardio", "Strength", "Yoga", "Other"])
    workout_minutes = st.number_input("Workout Minutes", min_value=0, max_value=600, value=0)
    calories_intake = st.number_input("Calories Consumed", min_value=0, max_value=10000, value=2000)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    mood_score = st.slider("Mood Score (1 = worst, 10 = best)", 1, 10, 7)
    mood_text = st.text_input("Describe your mood (optional)")
    submitted = st.form_submit_button("Save Entry")

if submitted:
    db.insert_log(log_date.isoformat(), workout_type, workout_minutes, calories_intake, sleep_hours, mood_score, mood_text)
    st.success("✅ Log saved successfully!")

# --- Display Data ---
st.header("📊 Your Fitness Logs")

logs = db.fetch_logs()
if logs.empty:
    st.info("No logs yet. Add some entries from the sidebar.")
else:
    st.dataframe(logs)
    import analysis

st.markdown("---")
st.subheader("📈 Correlation Analysis")

corr_matrix, corr_details = analysis.compute_correlations(logs)
if corr_matrix is None:
    st.info("Not enough data yet to compute correlations. Log a few more days!")
else:
    st.write("### Correlation Matrix")
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

    st.write("### Detailed Relationships (Top Insights)")
    df_corr = pd.DataFrame(corr_details)
    df_corr = df_corr.sort_values(by="r_value", key=abs, ascending=False).head(5)
    st.dataframe(df_corr)

    top = df_corr.iloc[0]
    relation = "positive" if top.r_value > 0 else "negative"
    st.success(f"✅ Strong {relation} correlation found between {top.var1} and {top.var2} (r = {top.r_value})")

    # Basic summary stats
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Sleep", f"{logs['sleep_hours'].mean():.1f} hrs")
    col2.metric("Avg Workout", f"{logs['workout_minutes'].mean():.0f} mins")
    col3.metric("Avg Mood", f"{logs['mood_score'].mean():.1f}/10")

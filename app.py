import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import os
from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Student Performance | AI Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM UI
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #f4f7fb;
    color: #17243a;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] {
    background: #101d35;
}
[data-testid="stSidebar"] * {
    color: #edf3ff !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent;
    border-radius: 10px;
    padding: 8px 10px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #203454;
}
.hero {
    background: linear-gradient(115deg, #142746 0%, #244d79 58%, #287d91 100%);
    border-radius: 20px;
    padding: 30px 34px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 12px 30px rgba(23, 52, 86, .13);
}
.hero h1 {
    font-family: 'Manrope', sans-serif;
    color: white;
    font-size: 2.05rem;
    margin: 0 0 8px 0;
    font-weight: 800;
}
.hero p {
    color: #dce9f8;
    margin: 0;
    font-size: 1rem;
}
.eyebrow {
    color: #9edce2;
    font-size: .76rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    margin-bottom: 10px;
}
.section-title {
    font-family: 'Manrope', sans-serif;
    color: #17243a;
    font-weight: 800;
    font-size: 1.25rem;
    margin: 18px 0 12px 0;
}
.metric-card {
    background: white;
    border: 1px solid #e5ebf3;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(27, 48, 76, .045);
    min-height: 112px;
}
.metric-label {
    color: #718096;
    font-size: .82rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.metric-value {
    color: #17243a;
    font-family: 'Manrope', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
}
.metric-note {
    color: #8492a6;
    font-size: .76rem;
    margin-top: 4px;
}
.info-card {
    background: white;
    border: 1px solid #e5ebf3;
    border-radius: 16px;
    padding: 20px;
    height: 100%;
}
.info-card h3 {
    color: #1d3557;
    font-family: 'Manrope', sans-serif;
    font-size: 1rem;
}
.info-card p {
    color: #64748b;
    font-size: .9rem;
}
div.stButton > button {
    background: #176b87;
    color: white;
    border: 0;
    border-radius: 10px;
    padding: .65rem 1.2rem;
    font-weight: 700;
}
div.stButton > button:hover {
    background: #12566e;
    color: white;
    border: 0;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #e5ebf3;
    border-radius: 12px;
}
.stAlert {
    border-radius: 12px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("data/raw/student_performance.csv")

try:
    model = load_model()
    df = load_dataset()
except Exception as exc:
    st.error("The model or dataset could not be loaded.")
    st.code(str(exc))
    st.stop()

feature_names = [
    "attendance",
    "internal_marks",
    "study_hours",
    "previous_performance",
    "assignment_score"
]

missing_features = [col for col in feature_names if col not in df.columns]
if missing_features:
    st.error(f"Dataset is missing required columns: {', '.join(missing_features)}")
    st.stop()

# Feature importance (available for the saved Logistic Regression pipeline)
try:
    coefficients = model.named_steps["model"].coef_[0]
    if len(coefficients) == len(feature_names):
        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(coefficients)
        }).sort_values("Importance", ascending=False)
    else:
        feature_importance = pd.DataFrame({
            "Feature": feature_names, "Importance": np.zeros(len(feature_names))
        })
except Exception:
    feature_importance = pd.DataFrame({
        "Feature": feature_names, "Importance": np.zeros(len(feature_names))
    })

# ============================================================
# PDF REPORT GENERATOR
# ============================================================
def build_student_report(student_values, student_name, student_id, predicted_label,
                         pass_probability, fail_probability, confidence, recommendations):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=48
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        textColor=colors.HexColor("#173653"), fontSize=20, leading=25
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], alignment=TA_CENTER,
        textColor=colors.HexColor("#60758b"), fontSize=9, leading=13
    )
    section_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"],
        textColor=colors.HexColor("#176b87"), spaceBefore=14, spaceAfter=7
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], leading=15,
        textColor=colors.HexColor("#26364a")
    )

    story = []
    logo_path = "cutm_logo.jpg"
    if os.path.exists(logo_path):
        logo = ReportLabImage(logo_path, width=72, height=90)
        logo.hAlign = "CENTER"
        story.extend([logo, Spacer(1, 8)])

    story.extend([
        Paragraph("AI-BASED STUDENT ACADEMIC PERFORMANCE", title_style),
        Paragraph("PREDICTION AND ANALYSIS SYSTEM", title_style),
        Spacer(1, 6),
        Paragraph("Centurion University of Technology and Management (CUTM)", subtitle_style),
        Paragraph("Machine Learning Based Outcome Prediction", subtitle_style),
        Spacer(1, 10),
        Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", subtitle_style),
        Spacer(1, 22),
        Paragraph("Student Information", section_style),
    ])
    identity_table = Table([
        ["Student name", student_name or "Not provided"],
        ["Student ID", student_id or "Not provided"],
    ], colWidths=[245, 190])
    identity_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf3f8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7e0e9")),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.extend([identity_table, Spacer(1, 10), Paragraph("Prediction Summary", section_style)])

    outcome_color = "#16875B" if predicted_label.upper() == "PASS" else "#C2413A"
    summary_data = [
        ["Predicted outcome", predicted_label.upper()],
        ["Selected outcome confidence", f"{confidence:.2f}%"],
        ["Pass probability", f"{pass_probability:.2f}%"],
        ["Fail probability", f"{fail_probability:.2f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[245, 190])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf3f8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#26364a")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor(outcome_color)),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7e0e9")),
        ("PADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([summary_table, Spacer(1, 12), Paragraph("Student Input Details", section_style)])

    input_data = [["Indicator", "Entered value"]] + [
        ["Attendance", f"{student_values['attendance']:.1f}%"],
        ["Internal marks", f"{student_values['internal_marks']:.1f}"],
        ["Study hours per day", f"{student_values['study_hours']:.1f}"],
        ["Previous performance", f"{student_values['previous_performance']:.1f}"],
        ["Assignment score", f"{student_values['assignment_score']:.1f}"],
    ]
    input_table = Table(input_data, colWidths=[245, 190])
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173653")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7e0e9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f9fc")]),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.extend([input_table, Spacer(1, 12), Paragraph("Improvement Recommendations", section_style)])

    if recommendations:
        for item in recommendations:
            story.append(Paragraph("• " + item.replace("&", "&amp;"), body_style))
            story.append(Spacer(1, 5))
    else:
        story.append(Paragraph("No recommendation thresholds were triggered for the entered values.", body_style))

    story.extend([
        Spacer(1, 18),
        Paragraph(
            "Disclaimer: This report is generated for educational and demonstration purposes. "
            "Model predictions are estimates and are not guaranteed academic outcomes.",
            styles["Italic"]
        )
    ])
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# PREDICTION HISTORY STORAGE
# ============================================================
HISTORY_FILE = "prediction_history.csv"
HISTORY_COLUMNS = [
    "prediction_date", "student_name", "student_id", "predicted_outcome",
    "confidence", "pass_probability", "fail_probability", "attendance",
    "internal_marks", "study_hours", "previous_performance", "assignment_score"
]

def load_prediction_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        history = pd.read_csv(HISTORY_FILE)
        for column in HISTORY_COLUMNS:
            if column not in history.columns:
                history[column] = ""
        return history[HISTORY_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

def save_prediction_record(record):
    history = load_prediction_history()
    updated = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
    updated.to_csv(HISTORY_FILE, index=False)



# ============================================================
# PHASE 18 — LOGIN & ROLE-BASED ACCESS
# ============================================================
def get_auth_settings():
    settings = {}
    try:
        settings = dict(st.secrets.get("auth", {}))
    except Exception:
        settings = {}
    return {
        "admin_username": settings.get("admin_username", os.getenv("APP_ADMIN_USERNAME", "")),
        "admin_password": settings.get("admin_password", os.getenv("APP_ADMIN_PASSWORD", "")),
        "student_username": settings.get("student_username", os.getenv("APP_STUDENT_USERNAME", "")),
        "student_password": settings.get("student_password", os.getenv("APP_STUDENT_PASSWORD", ""))
    }

auth_settings = get_auth_settings()
auth_ready = all(auth_settings.values())

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_role" not in st.session_state:
    st.session_state["user_role"] = ""

if not auth_ready:
    st.title("🔐 Configure application login")
    st.warning(
        "Login credentials are not configured yet. Create `.streamlit/secrets.toml` "
        "in the project folder and add the four credentials shown below. Do not commit this file to GitHub."
    )
    st.code(
        '[auth]\\n'
        'admin_username = "admin"\\n'
        'admin_password = "CHANGE_TO_A_STRONG_ADMIN_PASSWORD"\\n'
        'student_username = "student"\\n'
        'student_password = "CHANGE_TO_A_STRONG_STUDENT_PASSWORD"',
        language="toml"
    )
    st.stop()

if not st.session_state["authenticated"]:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at 4% 30%, rgba(23,135,170,.10), transparent 25%),
          linear-gradient(135deg,#f4f8fd 0%,#eaf3fb 58%,#f7fbff 100%);
    }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { display:none; }
    .block-container { max-width:1450px; padding-top:1rem; padding-bottom:1.5rem; }
    .top-brand { display:flex; align-items:center; justify-content:space-between; gap:18px; padding:4px 8px 16px 8px; }
    .university-name { color:#142746; font-size:1.04rem; font-weight:800; line-height:1.2; }
    .university-sub { color:#526985; font-size:.82rem; margin-top:3px; }
    .project-name { color:#142746; font-size:1.12rem; font-weight:800; text-align:right; }
    .project-sub { color:#526985; font-size:.82rem; text-align:right; margin-top:3px; }
    .login-left-card {
        background:linear-gradient(145deg,#092b56 0%,#075173 58%,#078b99 100%);
        border-radius:25px; min-height:535px; padding:34px 34px 24px 34px;
        color:white; box-shadow:0 18px 45px rgba(15,54,91,.16);
        position:relative; overflow:hidden;
    }
    .login-pill { display:inline-block; padding:7px 12px; border-radius:30px;
        background:rgba(25,146,194,.22); border:1px solid rgba(131,218,242,.3);
        color:#e1f7ff; font-size:.8rem; font-weight:700; }
    .login-left-card h1 { color:white; font-family:'Manrope',sans-serif; font-size:2.15rem;
        font-weight:800; line-height:1.12; margin:20px 0 12px 0; }
    .login-left-card h1 span { color:#55e3e4; }
    .login-left-card p { color:#e0eff9; font-size:.94rem; line-height:1.55; }
    .login-feature-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:24px 0 20px 0; text-align:center; }
    .login-feature-item { color:#f1f8ff; font-size:.77rem; line-height:1.3; font-weight:600; }
    .login-feature-icon { width:48px; height:48px; border-radius:14px; margin:0 auto 8px auto;
        display:flex; align-items:center; justify-content:center; background:rgba(23,153,207,.28); font-size:1.4rem; }
    .login-stack { border-top:1px solid rgba(220,245,255,.3); padding-top:16px; margin-top:16px; color:#e4f6ff; font-size:.8rem; }
    .campus-strip { margin-top:20px; border-radius:14px; padding:12px 15px;
        background:linear-gradient(90deg,rgba(4,28,58,.25),rgba(25,191,190,.18));
        border:1px solid rgba(177,238,245,.2); color:#dffaff; font-size:.8rem; }
    .login-form-title { text-align:center; color:#142746; font-family:'Manrope',sans-serif;
        font-size:1.65rem; font-weight:800; margin:0 0 4px 0; }
    .login-form-sub { text-align:center; color:#60748e; margin-bottom:18px; font-size:.88rem; }
    div[data-testid="stForm"] {
        background:rgba(255,255,255,.92); border:1px solid #e2ebf4;
        border-radius:24px; padding:28px 30px 22px 30px;
        box-shadow:0 18px 45px rgba(22,55,90,.11);
    }
    div[data-testid="stTextInput"] label { color:#314966; font-weight:700; }
    div[data-testid="stTextInput"] input { border-radius:11px; min-height:46px; border-color:#d4e0ec; background:#fbfdff; }
    div[data-testid="stFormSubmitButton"] button {
        width:100%; min-height:50px; border-radius:12px;
        background:linear-gradient(100deg,#0d2e5b,#078d9c);
        color:white; font-size:1rem; font-weight:800; border:0;
    }
    div[data-testid="stFormSubmitButton"] button:hover { color:white; border:0; filter:brightness(1.05); }
    .role-divider { display:flex; align-items:center; gap:12px; color:#60748e; font-size:.84rem; margin:17px 0 10px 0; }
    .role-divider:before,.role-divider:after { content:""; height:1px; flex:1; background:#d5e0eb; }
    div[role="radiogroup"] { justify-content:center; gap:10px; }
    div[role="radiogroup"] label { background:#f0f6fc; border:1px solid #dce8f3; border-radius:11px; padding:9px 14px; }
    .login-footnote { color:#8999ad; font-size:.74rem; text-align:center; margin-top:12px; }
    @media(max-width:900px) {
        .top-brand { flex-direction:column; align-items:flex-start; }
        .project-name,.project-sub { text-align:left; }
        .login-left-card { min-height:auto; padding:26px 22px; }
        .login-left-card h1 { font-size:1.8rem; }
        .login-feature-grid { grid-template-columns:repeat(2,1fr); }
        div[data-testid="stForm"] { padding:22px 18px; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Compact university / project header
    header_left, header_right = st.columns([1.15, 1], vertical_alignment="center")
    with header_left:
        logo_col, uni_col = st.columns([0.22, 1], vertical_alignment="center")
        with logo_col:
            if os.path.exists("cutm_logo.jpg"):
                st.image("cutm_logo.jpg", width=66)
            else:
                st.markdown("🎓")
        with uni_col:
            st.markdown(
                '<div class="university-name">Centurion University</div>'
                '<div class="university-name">of Technology and Management</div>'
                '<div class="university-sub">Bhubaneswar Campus, Odisha</div>',
                unsafe_allow_html=True
            )
    with header_right:
        st.markdown(
            '<div class="project-name">🔷 Student Performance Prediction</div>'
            '<div class="project-sub">A Machine Learning Based Academic Analytics System</div>',
            unsafe_allow_html=True
        )

    left, right = st.columns([1.08, 1], gap="large", vertical_alignment="top")
    with left:
        st.markdown(
            '<div class="login-left-card">'
            '<span class="login-pill">🎓 &nbsp; Academic Analytics</span>'
            '<h1>Student Performance<br>Prediction <span>System</span></h1>'
            '<p>A machine learning based system to predict student performance, identify key factors, '
            'and provide personalized learning insights for academic improvement.</p>'
            '<div class="login-feature-grid">'
            '<div class="login-feature-item"><div class="login-feature-icon">▥</div>Performance<br>Prediction</div>'
            '<div class="login-feature-item"><div class="login-feature-icon">⌁</div>Explainable<br>AI (XAI)</div>'
            '<div class="login-feature-item"><div class="login-feature-icon">▤</div>Improvement<br>Advisor</div>'
            '<div class="login-feature-item"><div class="login-feature-icon">♟</div>Student<br>Registry</div>'
            '</div>'
            '<div class="login-stack"><b>◈ &nbsp; Tech Stack:</b> &nbsp; Python &nbsp; • &nbsp; Scikit-learn '
            '&nbsp; • &nbsp; Streamlit &nbsp; • &nbsp; Joblib</div>'
            '<div class="campus-strip">CENTURION UNIVERSITY · BHUBANESWAR<br>'
            '<b>Predict. Understand. Improve.</b></div>'
            '</div>',
            unsafe_allow_html=True
        )
    with right:
        st.markdown(
            '<div style="text-align:center;margin:5px 0 12px 0;font-size:2rem;">👥</div>'
            '<div class="login-form-title">Sign In to Continue</div>'
            '<div class="login-form-sub">Access your dashboard based on your role</div>',
            unsafe_allow_html=True
        )
        with st.form("application_login_form"):
            login_username = st.text_input("Username", placeholder="Enter your username")
            login_password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_submit = st.form_submit_button("➜   Sign In", use_container_width=True)
        st.markdown('<div class="role-divider">Choose your role</div>', unsafe_allow_html=True)
        login_role = st.radio(
            "Choose your role", ["👑 Admin", "🎓 Student"],
            horizontal=True, label_visibility="collapsed", key="login_role_choice"
        )
        st.markdown('<div class="login-footnote">Secure access for authorised users only.</div>', unsafe_allow_html=True)

    if login_submit:
        selected_role = "Admin" if login_role == "👑 Admin" else "Student"
        expected_username = auth_settings["admin_username"] if selected_role == "Admin" else auth_settings["student_username"]
        expected_password = auth_settings["admin_password"] if selected_role == "Admin" else auth_settings["student_password"]
        if login_username == expected_username and login_password == expected_password:
            st.session_state["authenticated"] = True
            st.session_state["user_role"] = selected_role
            st.session_state["login_username"] = login_username
            st.rerun()
        else:
            st.error(f"Incorrect {selected_role} credentials. Please check your username and password.")
    st.stop()


# ============================================================
# SHARED STUDENT REGISTRY HELPERS
# Keep these at module scope because multiple pages use them.
# ============================================================
REGISTRY_FILE = "student_registry.csv"
REGISTRY_COLUMNS = ["student_id", "student_name", "course", "semester", "section", "created_at"]

def load_student_registry():
    if not os.path.exists(REGISTRY_FILE):
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    try:
        registry_data = pd.read_csv(REGISTRY_FILE, dtype={"student_id": str})
        for column in REGISTRY_COLUMNS:
            if column not in registry_data.columns:
                registry_data[column] = ""
        registry_data["student_id"] = registry_data["student_id"].fillna("").astype(str).str.strip()
        return registry_data[REGISTRY_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    if os.path.exists("cutm_logo.jpg"):
        st.image("cutm_logo.jpg", width=150)
    st.markdown("## 🎓 Student Analytics")
    st.caption("MACHINE LEARNING SYSTEM")
    st.divider()
    st.caption(f"Signed in as: {st.session_state.get('login_username', '')}")
    st.caption(f"Role: {st.session_state.get('user_role', 'Student')}")
    all_pages = [
        "🏠  Dashboard", "🔮  Predict Result", "📊  Dataset Analysis",
        "🤖  Model Performance", "🔍  Student Insights", "🗂️  Prediction History",
        "📈  Prediction Analytics", "📥  Bulk Prediction", "📉  Model Comparison",
        "📄  Analytics Report", "👤  Student Profiles", "🗃️  Student Registry",
        "🎯  Improvement Advisor", "🧠  Explainable AI",
        "📅  Student Progress Tracker", "📊  Advanced Model Evaluation",
        "ℹ️  About Project"
    ]
    student_pages = [
        "🔮  Predict Result", "🎯  Improvement Advisor",
        "🧠  Explainable AI", "ℹ️  About Project"
    ]
    allowed_pages = all_pages if st.session_state["user_role"] == "Admin" else student_pages
    page = st.radio("NAVIGATION", allowed_pages, label_visibility="visible")
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        for key in ["authenticated", "user_role", "login_username"]:
            st.session_state.pop(key, None)
        st.rerun()
    st.caption("Academic decision-support prototype")
    st.caption("Built with Python · Scikit-learn · Streamlit")

def hero(kicker, title, description):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{kicker}</div>'
        f'<h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True
    )

def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

def prediction_frame(attendance, internal_marks, study_hours,
                     previous_performance, assignment_score):
    return pd.DataFrame([{
        "attendance": attendance,
        "internal_marks": internal_marks,
        "study_hours": study_hours,
        "previous_performance": previous_performance,
        "assignment_score": assignment_score
    }], columns=feature_names)

# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================
if page == "🏠  Dashboard":
    hero(
        "STUDENT SUCCESS • PREDICTIVE ANALYTICS",
        "AI-Based Student Academic Performance Prediction and Analysis System",
        "A machine-learning dashboard to explore academic indicators and estimate student outcomes."
    )

    total_students = len(df)
    missing_count = int(df.isnull().sum().sum())
    pass_count = int((df["result"].astype(str).str.lower() == "pass").sum()) if "result" in df.columns else None
    pass_rate = f"{pass_count / total_students * 100:.1f}%" if pass_count is not None and total_students else "—"

    cols = st.columns(4)
    with cols[0]: metric_card("STUDENT RECORDS", f"{total_students:,}", "Rows in loaded dataset")
    with cols[1]: metric_card("INPUT FEATURES", str(len(feature_names)), "Academic indicators")
    with cols[2]: metric_card("MISSING VALUES", f"{missing_count:,}", "Across all dataset columns")
    with cols[3]: metric_card("PASS RATE IN DATA", pass_rate, "Dataset distribution, not a prediction")

    st.markdown('<div class="section-title">Project overview</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("""
        <div class="info-card">
        <h3>🎯 What this system does</h3>
        <p>It uses attendance, internal marks, study hours, previous performance and assignment scores
        to estimate whether a student may Pass or Fail. It also highlights indicators that may need attention.</p>
        <h3>🔄 Prediction workflow</h3>
        <p>Student inputs → saved preprocessing/model pipeline → predicted outcome → probability and general guidance.</p>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="info-card">
        <h3>🧠 Technology stack</h3>
        <p><b>Python</b> for application logic</p>
        <p><b>Scikit-learn</b> for machine learning</p>
        <p><b>Pandas & NumPy</b> for data processing</p>
        <p><b>Streamlit</b> for the interactive dashboard</p>
        </div>
        """, unsafe_allow_html=True)

    if "result" in df.columns:
        st.markdown('<div class="section-title">Outcome distribution in dataset</div>', unsafe_allow_html=True)
        outcome_counts = df["result"].astype(str).value_counts()
        st.bar_chart(outcome_counts)

# ============================================================
# PAGE 2 — PREDICT RESULT
# ============================================================
elif page == "🔮  Predict Result":
    hero("PREDICTION WORKSPACE", "Predict student outcome", "Enter the academic indicators below to generate a model prediction.")

    with st.container(border=True):
        st.markdown("### Student information")
        # Optional integration with the local student registry.
        registry_path = "student_registry.csv"
        registered_students = pd.DataFrame()
        if os.path.exists(registry_path):
            try:
                registered_students = pd.read_csv(registry_path, dtype={"student_id": str})
                if {"student_id", "student_name"}.issubset(registered_students.columns):
                    registered_students = registered_students.fillna("")
                else:
                    registered_students = pd.DataFrame()
            except Exception:
                registered_students = pd.DataFrame()

        registry_choice = "-- Enter student details manually --"
        if not registered_students.empty:
            choices = [registry_choice] + [
                f'{row["student_id"]} — {row["student_name"]}'
                for _, row in registered_students.iterrows()
            ]
            selected_student = st.selectbox("Choose a registered student (optional)", choices)
        else:
            selected_student = registry_choice
            st.caption("Tip: Register students in Student Registry to select them here.")

        default_name, default_id = "", ""
        if selected_student != registry_choice and not registered_students.empty:
            selected_id = selected_student.split(" — ", 1)[0]
            match = registered_students[registered_students["student_id"].astype(str) == selected_id]
            if not match.empty:
                default_name = str(match.iloc[0]["student_name"])
                default_id = str(match.iloc[0]["student_id"])

        student_name = st.text_input("Student name", value=default_name, placeholder="Enter student name")
        student_id = st.text_input("Student ID / Roll number", value=default_id, placeholder="Enter student ID or roll number")
        st.markdown("### Academic profile")
        c1, c2 = st.columns(2)
        with c1:
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0, 1.0)
            internal_marks = st.number_input("Internal marks", 0.0, 100.0, 60.0, 1.0)
            study_hours = st.number_input("Study hours per day", 0.0, 24.0, 3.0, 0.5)
        with c2:
            previous_performance = st.number_input("Previous performance", 0.0, 100.0, 65.0, 1.0)
            assignment_score = st.number_input("Assignment score", 0.0, 100.0, 70.0, 1.0)
            st.info("Use values on the same scale as the training dataset.")

        predict_clicked = st.button("✨ Generate prediction", use_container_width=True)

    if predict_clicked:
        student = prediction_frame(attendance, internal_marks, study_hours,
                                   previous_performance, assignment_score)
        try:
            prediction = model.predict(student)[0]
            probabilities = model.predict_proba(student)[0]
            st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
            predicted_label = str(prediction).upper()
            confidence = float(np.max(probabilities) * 100)
            pass_probability = float(probabilities[list(model.classes_).index("Pass")] * 100) if "Pass" in list(model.classes_) else 0.0
            fail_probability = float(probabilities[list(model.classes_).index("Fail")] * 100) if "Fail" in list(model.classes_) else 0.0

            if predicted_label == "PASS":
                result_bg = "#e2f7ed"
                result_border = "#1b9b65"
                result_icon = "✓"
                result_text = "PASS"
            else:
                result_bg = "#fff0ed"
                result_border = "#d9534f"
                result_icon = "!"
                result_text = "FAIL"

            st.markdown(
                f"""
                <div style="background:{result_bg}; border-left:7px solid {result_border};
                            border-radius:16px; padding:24px 26px; margin:10px 0 20px 0;">
                    <div style="font-size:.82rem; letter-spacing:1.5px; font-weight:700;
                                color:#53657b; text-transform:uppercase;">Predicted student outcome</div>
                    <div style="display:flex; align-items:center; gap:16px; margin-top:8px;">
                        <div style="width:54px;height:54px;border-radius:50%;background:{result_border};
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:30px;font-weight:800;">{result_icon}</div>
                        <div style="font-size:2.1rem;font-weight:800;color:#17243a;">{result_text}</div>
                    </div>
                    <div style="margin-top:12px;color:#53657b;">Model confidence in the selected outcome:
                        <b>{confidence:.2f}%</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("#### Outcome probability")
            p1, p2 = st.columns(2)
            with p1:
                metric_card("PASS PROBABILITY", f"{pass_probability:.2f}%", "Model-estimated probability")
            with p2:
                metric_card("FAIL PROBABILITY", f"{fail_probability:.2f}%", "Model-estimated probability")

            prob_df = pd.DataFrame({
                "Outcome": model.classes_,
                "Probability (%)": (probabilities * 100).round(2)
            })
            st.progress(min(max(confidence / 100, 0.0), 1.0), text=f"Selected outcome confidence: {confidence:.2f}%")
            st.bar_chart(prob_df.set_index("Outcome"))

            st.markdown("#### Student input summary")
            summary = pd.DataFrame({
                "Indicator": ["Attendance", "Internal marks", "Study hours/day",
                              "Previous performance", "Assignment score"],
                "Entered value": [f"{attendance:.1f}%", f"{internal_marks:.1f}",
                                  f"{study_hours:.1f} hours", f"{previous_performance:.1f}",
                                  f"{assignment_score:.1f}"]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.markdown("#### Factors to review")
            recommendations = []
            if attendance < 75:
                recommendations.append("Attendance is below 75%; improving class attendance may help.")
            if study_hours < 2:
                recommendations.append("Study time is under 2 hours per day; consider a consistent study routine.")
            if internal_marks < 50:
                recommendations.append("Internal marks are below 50; review topics and prepare for assessments.")
            if previous_performance < 50:
                recommendations.append("Previous performance is below 50; identify and practise weaker topics.")
            if assignment_score < 50:
                recommendations.append("Assignment score is below 50; give more attention to coursework.")
            if recommendations:
                for item in recommendations:
                    st.warning(item)
            else:
                st.success("No recommendation thresholds were triggered for these inputs.")

            prediction_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_record = {
                "prediction_date": prediction_timestamp,
                "student_name": student_name.strip(),
                "student_id": student_id.strip(),
                "predicted_outcome": predicted_label,
                "confidence": round(confidence, 2),
                "pass_probability": round(pass_probability, 2),
                "fail_probability": round(fail_probability, 2),
                "attendance": attendance,
                "internal_marks": internal_marks,
                "study_hours": study_hours,
                "previous_performance": previous_performance,
                "assignment_score": assignment_score
            }
            save_prediction_record(history_record)
            st.success("Prediction saved to Prediction History.")
            st.caption("Predictions are estimates from the trained model, not guaranteed academic outcomes.")

            report_values = {
                "attendance": attendance,
                "internal_marks": internal_marks,
                "study_hours": study_hours,
                "previous_performance": previous_performance,
                "assignment_score": assignment_score
            }
            pdf_bytes = build_student_report(
                report_values, student_name, student_id, predicted_label,
                pass_probability, fail_probability, confidence, recommendations
            )
            st.download_button(
                label="📄 Download Student Report (PDF)",
                data=pdf_bytes,
                file_name=f"student_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as exc:
            st.error("Prediction could not be generated.")
            st.code(str(exc))

# ============================================================
# PAGE 3 — DATASET ANALYSIS
# ============================================================
elif page == "📊  Dataset Analysis":
    hero("DATA EXPLORATION", "Understand the dataset", "Inspect records, data quality and distributions used by this application.")

    a, b, c = st.columns(3)
    with a: metric_card("ROWS", f"{df.shape[0]:,}", "Dataset records")
    with b: metric_card("COLUMNS", str(df.shape[1]), "Available variables")
    with c: metric_card("NULL VALUES", f"{int(df.isnull().sum().sum()):,}", "All columns")

    st.markdown('<div class="section-title">Dataset preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(25), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Missing values by column</div>', unsafe_allow_html=True)
        missing_df = df.isnull().sum().rename("Missing values").to_frame()
        st.dataframe(missing_df, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Statistical summary</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        st.markdown('<div class="section-title">Feature distribution</div>', unsafe_allow_html=True)
        selected_column = st.selectbox("Choose a numeric feature", numeric_columns)
        st.bar_chart(df[selected_column].dropna().value_counts().sort_index())

# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================
elif page == "🤖  Model Performance":
    hero("MODEL TRANSPARENCY", "Model performance", "Review the selected model and the evaluation information documented for this project.")

    st.markdown('<div class="section-title">Selected algorithm</div>', unsafe_allow_html=True)
    st.info("Logistic Regression — loaded from models/best_model.pkl")

    st.markdown('<div class="section-title">Reported benchmark</div>', unsafe_allow_html=True)
    metric_card("TEST ACCURACY REPORTED IN PROJECT README", "79.6%", "Verify against the original held-out test evaluation before final submission.")

    st.warning(
        "Precision, recall, F1-score and confusion-matrix values are not available in the supplied app code. "
        "They are intentionally not invented here. Add them only after evaluating the model on a held-out test set."
    )

    st.markdown('<div class="section-title">Feature importance</div>', unsafe_allow_html=True)
    if feature_importance["Importance"].sum() > 0:
        st.bar_chart(feature_importance.set_index("Feature")["Importance"])
        st.caption("Absolute Logistic Regression coefficients; larger values indicate stronger model influence on its linear decision function.")
    else:
        st.info("Feature coefficients could not be read from the saved model pipeline.")

# ============================================================
# PAGE 5 — STUDENT INSIGHTS
# ============================================================
elif page == "🔍  Student Insights":
    hero("INDIVIDUAL PROFILE", "Student insights", "Adjust the indicators to review a student's academic profile and potential risk flags.")

    c1, c2 = st.columns(2)
    with c1:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        internal_marks = st.slider("Internal marks", 0, 100, 60)
        study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
    with c2:
        previous_performance = st.slider("Previous performance", 0, 100, 65)
        assignment_score = st.slider("Assignment score", 0, 100, 70)

    values = {
        "Attendance": attendance,
        "Internal marks": internal_marks,
        "Study hours": study_hours,
        "Previous performance": previous_performance,
        "Assignment score": assignment_score
    }
    st.markdown('<div class="section-title">Profile summary</div>', unsafe_allow_html=True)
    profile = pd.DataFrame({"Indicator": list(values.keys()), "Entered value": list(values.values())})
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Areas to monitor</div>', unsafe_allow_html=True)
    flags = []
    if attendance < 75: flags.append("Attendance is below the 75% review threshold.")
    if internal_marks < 50: flags.append("Internal marks are below 50.")
    if study_hours < 2: flags.append("Daily study time is below 2 hours.")
    if previous_performance < 50: flags.append("Previous performance is below 50.")
    if assignment_score < 50: flags.append("Assignment score is below 50.")
    if flags:
        for flag in flags:
            st.warning(flag)
    else:
        st.success("No major areas were flagged by the current rule-based thresholds.")
    st.caption("These flags are simple guidance rules and are not a diagnosis or guaranteed outcome.")

# ============================================================
# PAGE 6 — PREDICTION HISTORY
# ============================================================
elif page == "🗂️  Prediction History":
    hero("STUDENT RECORDS", "Prediction history", "Review and export prediction records generated in this application.")
    history = load_prediction_history()

    if history.empty:
        st.info("No prediction records yet. Generate a prediction first; it will appear here automatically.")
    else:
        total_records = len(history)
        pass_records = int((history["predicted_outcome"].astype(str).str.upper() == "PASS").sum())
        fail_records = int((history["predicted_outcome"].astype(str).str.upper() == "FAIL").sum())
        m1, m2, m3 = st.columns(3)
        with m1: metric_card("TOTAL PREDICTIONS", f"{total_records:,}", "Saved records")
        with m2: metric_card("PREDICTED PASS", f"{pass_records:,}", "Model outcomes")
        with m3: metric_card("PREDICTED FAIL", f"{fail_records:,}", "Model outcomes")

        search_text = st.text_input("Search by student name or ID", placeholder="Type a name or student ID")
        filtered = history.copy()
        if search_text.strip():
            query = search_text.strip().lower()
            filtered = filtered[
                filtered["student_name"].astype(str).str.lower().str.contains(query, na=False)
                | filtered["student_id"].astype(str).str.lower().str.contains(query, na=False)
            ]

        outcome_filter = st.selectbox("Filter by predicted outcome", ["All", "PASS", "FAIL"])
        if outcome_filter != "All":
            filtered = filtered[filtered["predicted_outcome"].astype(str).str.upper() == outcome_filter]

        display_columns = [
            "prediction_date", "student_name", "student_id", "predicted_outcome",
            "confidence", "pass_probability", "fail_probability"
        ]
        st.markdown('<div class="section-title">Saved student prediction records</div>', unsafe_allow_html=True)
        st.dataframe(filtered[display_columns].sort_values("prediction_date", ascending=False),
                     use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download records as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="student_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Records are stored locally in prediction_history.csv in the project folder.")
        with st.expander("Delete all saved prediction history"):
            st.warning("This permanently clears the local prediction history CSV file.")
            confirm_delete = st.checkbox("I understand and want to delete all prediction records.")
            if st.button("🗑️ Clear prediction history", disabled=not confirm_delete):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.success("Prediction history cleared.")
                st.rerun()

# ============================================================
# PAGE 7 — PREDICTION ANALYTICS
# ============================================================
elif page == "📈  Prediction Analytics":
    hero(
        "HISTORICAL INSIGHTS",
        "Prediction analytics",
        "Explore the predictions saved by this application. These charts describe model outputs, not verified student results."
    )

    history = load_prediction_history()
    if history.empty:
        st.info("No saved predictions yet. Generate predictions first; analytics will appear here.")
    else:
        analytics = history.copy()
        analytics["prediction_date"] = pd.to_datetime(
            analytics["prediction_date"], errors="coerce"
        )
        analytics["confidence"] = pd.to_numeric(analytics["confidence"], errors="coerce")
        analytics["pass_probability"] = pd.to_numeric(
            analytics["pass_probability"], errors="coerce"
        )
        analytics["fail_probability"] = pd.to_numeric(
            analytics["fail_probability"], errors="coerce"
        )
        analytics["predicted_outcome"] = (
            analytics["predicted_outcome"].astype(str).str.upper()
        )

        total = len(analytics)
        pass_total = int((analytics["predicted_outcome"] == "PASS").sum())
        fail_total = int((analytics["predicted_outcome"] == "FAIL").sum())
        avg_confidence = analytics["confidence"].mean()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("TOTAL PREDICTIONS", f"{total:,}", "Saved model outputs")
        with k2:
            metric_card("PREDICTED PASS", f"{pass_total:,}", "Predicted outcome")
        with k3:
            metric_card("PREDICTED FAIL", f"{fail_total:,}", "Predicted outcome")
        with k4:
            metric_card(
                "AVERAGE CONFIDENCE",
                f"{avg_confidence:.1f}%" if pd.notna(avg_confidence) else "—",
                "Across saved predictions"
            )

        st.markdown('<div class="section-title">Predicted outcome distribution</div>', unsafe_allow_html=True)
        outcome_counts = (
            analytics["predicted_outcome"]
            .value_counts()
            .reindex(["PASS", "FAIL"], fill_value=0)
        )
        left, right = st.columns(2)
        with left:
            st.bar_chart(outcome_counts)
        with right:
            outcome_summary = pd.DataFrame({
                "Predicted outcome": ["PASS", "FAIL"],
                "Records": [pass_total, fail_total],
                "Share (%)": [
                    round(pass_total / total * 100, 1) if total else 0,
                    round(fail_total / total * 100, 1) if total else 0
                ]
            })
            st.dataframe(outcome_summary, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Prediction activity over time</div>', unsafe_allow_html=True)
        dated = analytics.dropna(subset=["prediction_date"]).copy()
        if not dated.empty:
            dated["prediction_day"] = dated["prediction_date"].dt.date.astype(str)
            daily_counts = dated.groupby("prediction_day").size().rename("Predictions")
            st.line_chart(daily_counts)
        else:
            st.info("Date information is unavailable for the saved records.")

        st.markdown('<div class="section-title">Confidence distribution</div>', unsafe_allow_html=True)
        confidence_values = analytics["confidence"].dropna()
        if not confidence_values.empty:
            confidence_bins = pd.cut(
                confidence_values,
                bins=[0, 50, 60, 70, 80, 90, 100],
                include_lowest=True
            ).value_counts().sort_index()
            confidence_bins.index = confidence_bins.index.astype(str)
            st.bar_chart(confidence_bins.rename("Number of predictions"))
        else:
            st.info("Confidence values are unavailable for the saved records.")

        st.caption(
            "Important: PASS/FAIL counts and confidence values describe the model's predictions only. "
            "They do not measure actual pass rates, model accuracy, or confirmed student outcomes."
        )



# ============================================================
# PAGE 8 — BULK STUDENT PREDICTION
# ============================================================
elif page == "📥  Bulk Prediction":
    hero(
        "BATCH PREDICTION",
        "Bulk student prediction",
        "Upload a CSV or Excel sheet to generate Pass/Fail predictions for multiple students at once."
    )

    st.markdown("""
    <div class="info-card">
      <h3>Required columns</h3>
      <p>Your file must contain these five numeric columns, using the same scale as the model's training data:
      <b>attendance, internal_marks, study_hours, previous_performance, assignment_score</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    template = pd.DataFrame({
        "student_name": ["Student 1", "Student 2"],
        "student_id": ["STU001", "STU002"],
        "attendance": [85, 60],
        "internal_marks": [72, 45],
        "study_hours": [3.5, 1.0],
        "previous_performance": [75, 50],
        "assignment_score": [80, 48]
    })
    st.download_button(
        "⬇️ Download sample upload template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="bulk_prediction_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload student data",
        type=["csv", "xlsx"],
        help="CSV or Excel (.xlsx) file. Maximum file size depends on your Streamlit configuration."
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                bulk_df = pd.read_csv(uploaded_file)
            else:
                bulk_df = pd.read_excel(uploaded_file)

            st.markdown('<div class="section-title">Uploaded data preview</div>', unsafe_allow_html=True)
            st.caption(f"{len(bulk_df):,} student row(s) found.")
            st.dataframe(bulk_df.head(10), use_container_width=True, hide_index=True)

            missing = [col for col in feature_names if col not in bulk_df.columns]
            if missing:
                st.error("Missing required column(s): " + ", ".join(missing))
                st.info("Please use the sample template above and keep the column names unchanged.")
            elif bulk_df.empty:
                st.warning("The uploaded file has no student rows.")
            else:
                cleaned = bulk_df.copy()
                for col in feature_names:
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

                invalid_mask = cleaned[feature_names].isna().any(axis=1)
                range_errors = pd.Series(False, index=cleaned.index)
                for col in ["attendance", "internal_marks", "previous_performance", "assignment_score"]:
                    range_errors |= (cleaned[col] < 0) | (cleaned[col] > 100)
                range_errors |= (cleaned["study_hours"] < 0) | (cleaned["study_hours"] > 24)
                invalid_mask |= range_errors

                if invalid_mask.any():
                    st.error(
                        f"{int(invalid_mask.sum())} row(s) contain missing, non-numeric, or out-of-range values. "
                        "Correct these rows and upload the file again."
                    )
                    st.dataframe(
                        bulk_df.loc[invalid_mask].head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    if st.button("🚀 Generate predictions for all students", use_container_width=True):
                        try:
                            model_input = cleaned[feature_names].copy()
                            predictions = model.predict(model_input)
                            probabilities = model.predict_proba(model_input)
                            classes = [str(c).strip().upper() for c in model.classes_]

                            result_df = bulk_df.copy()
                            result_df["predicted_outcome"] = [str(x).upper() for x in predictions]
                            result_df["confidence_percent"] = (np.max(probabilities, axis=1) * 100).round(2)

                            if "PASS" in classes:
                                pass_idx = classes.index("PASS")
                                result_df["pass_probability_percent"] = (probabilities[:, pass_idx] * 100).round(2)
                            else:
                                result_df["pass_probability_percent"] = 0.0

                            if "FAIL" in classes:
                                fail_idx = classes.index("FAIL")
                                result_df["fail_probability_percent"] = (probabilities[:, fail_idx] * 100).round(2)
                            else:
                                result_df["fail_probability_percent"] = 0.0

                            st.session_state["bulk_prediction_results"] = result_df
                            st.success(f"Predictions generated successfully for {len(result_df):,} students.")
                        except Exception as exc:
                            st.error("Prediction could not be completed. Please check the uploaded data and model.")
                            st.code(str(exc))

            results = st.session_state.get("bulk_prediction_results")
            if results is not None:
                st.markdown('<div class="section-title">Bulk prediction summary</div>', unsafe_allow_html=True)
                total_rows = len(results)
                pass_count = int((results["predicted_outcome"] == "PASS").sum())
                fail_count = int((results["predicted_outcome"] == "FAIL").sum())
                a, b, c = st.columns(3)
                with a:
                    metric_card("TOTAL STUDENTS", f"{total_rows:,}", "Rows processed")
                with b:
                    metric_card("PREDICTED PASS", f"{pass_count:,}", "Model output")
                with c:
                    metric_card("PREDICTED FAIL", f"{fail_count:,}", "Model output")

                st.dataframe(results, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download prediction results (CSV)",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="student_bulk_prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(
                    "These are model-generated predictions, not confirmed academic results. "
                    "Use them as decision-support information and verify important decisions independently."
                )
        except Exception as exc:
            st.error("The uploaded file could not be read. Please check that it is a valid CSV or .xlsx file.")
            st.code(str(exc))



# ============================================================
# PAGE 10 — MODEL COMPARISON
# ============================================================
elif page == "📉  Model Comparison":
    hero(
        "MODEL EVALUATION",
        "Model accuracy comparison",
        "Compare the accuracy values recorded in the project's model_results.csv file."
    )

    results_path = os.path.join("reports", "model_results.csv")
    if os.path.exists(results_path):
        try:
            results_df = pd.read_csv(results_path)
            required_cols = {"Model", "Accuracy"}
            if not required_cols.issubset(results_df.columns):
                st.error("model_results.csv must contain the columns: Model, Accuracy.")
            elif results_df.empty:
                st.warning("The model results file is empty.")
            else:
                results_df["Accuracy"] = pd.to_numeric(results_df["Accuracy"], errors="coerce")
                results_df = results_df.dropna(subset=["Accuracy"])
                if results_df.empty:
                    st.warning("No valid accuracy values were found in model_results.csv.")
                else:
                    # Values in the supplied file are stored as fractions (e.g. 0.796).
                    results_df["Accuracy (%)"] = results_df["Accuracy"].apply(
                        lambda value: value * 100 if 0 <= value <= 1 else value
                    ).round(1)

                    st.markdown('<div class="section-title">Recorded model results</div>', unsafe_allow_html=True)
                    st.dataframe(
                        results_df[["Model", "Accuracy (%)"]],
                        use_container_width=True,
                        hide_index=True
                    )

                    best_row = results_df.loc[results_df["Accuracy (%)"].idxmax()]
                    cols = st.columns(min(3, len(results_df) + 1))
                    with cols[0]:
                        metric_card("BEST RECORDED MODEL", str(best_row["Model"]), "Highest accuracy in this file")
                    with cols[1]:
                        metric_card("HIGHEST ACCURACY", f'{best_row["Accuracy (%)"]:.1f}%', "Reported result")

                    chart_df = results_df.set_index("Model")[["Accuracy (%)"]]
                    st.markdown('<div class="section-title">Accuracy by model</div>', unsafe_allow_html=True)
                    st.bar_chart(chart_df, y="Accuracy (%)", use_container_width=True)

                    st.info(
                        "These values are read directly from reports/model_results.csv. "
                        "They show the accuracy figures recorded by the project; this file alone "
                        "does not contain test predictions or actual labels."
                    )
                    st.warning(
                        "Precision, recall, F1-score and a confusion matrix cannot be calculated "
                        "from accuracy values alone. We would need the held-out test labels and "
                        "the model's predictions."
                    )
        except Exception as exc:
            st.error("Could not read the model results file.")
            st.code(str(exc))
    else:
        st.error("Could not find reports/model_results.csv. Please check the reports folder.")



# ============================================================
# PAGE 12 — DOWNLOADABLE ANALYTICS REPORT
# ============================================================
elif page == "📄  Analytics Report":
    hero(
        "REPORT CENTER",
        "Download analytics report",
        "Generate a PDF summary from prediction records saved by this application."
    )

    report_history = load_prediction_history()
    if report_history.empty:
        st.info("No prediction records are saved yet. Make a prediction first, then return here to generate a report.")
    else:
        total = len(report_history)
        outcomes = report_history["predicted_outcome"].astype(str).str.upper()
        pass_total = int((outcomes == "PASS").sum())
        fail_total = int((outcomes == "FAIL").sum())
        confidence = pd.to_numeric(report_history["confidence"], errors="coerce")
        avg_confidence = float(confidence.mean()) if confidence.notna().any() else None

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("SAVED PREDICTIONS", f"{total:,}", "Records in prediction history")
        with c2:
            metric_card("PREDICTED PASS", f"{pass_total:,}", "Model outputs")
        with c3:
            metric_card("PREDICTED FAIL", f"{fail_total:,}", "Model outputs")

        st.markdown('<div class="section-title">Report preview</div>', unsafe_allow_html=True)
        preview_cols = ["prediction_date", "student_name", "student_id", "predicted_outcome", "confidence"]
        st.dataframe(report_history[preview_cols], use_container_width=True, hide_index=True)

        if st.button("🧾 Generate PDF analytics report", use_container_width=True):
            try:
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=40, bottomMargin=36)
                styles = getSampleStyleSheet()
                story = [
                    Paragraph("Student Performance Prediction – Analytics Report", styles["Title"]),
                    Spacer(1, 10),
                    Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["Normal"]),
                    Spacer(1, 14),
                    Paragraph(f"Total saved predictions: <b>{total}</b>", styles["Normal"]),
                    Paragraph(f"Predicted PASS: <b>{pass_total}</b>", styles["Normal"]),
                    Paragraph(f"Predicted FAIL: <b>{fail_total}</b>", styles["Normal"]),
                    Paragraph(
                        f"Average model confidence: <b>{avg_confidence:.1f}%</b>" if avg_confidence is not None
                        else "Average model confidence: Not available",
                        styles["Normal"]
                    ),
                    Spacer(1, 16),
                    Paragraph("Prediction records", styles["Heading2"]),
                    Spacer(1, 8)
                ]
                table_data = [["Date", "Student", "Student ID", "Prediction", "Confidence"]]
                for _, row in report_history.iterrows():
                    conf = pd.to_numeric(pd.Series([row["confidence"]]), errors="coerce").iloc[0]
                    conf_text = f"{conf:.1f}%" if pd.notna(conf) else "—"
                    table_data.append([
                        str(row["prediction_date"])[:16],
                        str(row["student_name"])[:22],
                        str(row["student_id"])[:16],
                        str(row["predicted_outcome"]),
                        conf_text
                    ])
                table = Table(table_data, repeatRows=1, colWidths=[92, 110, 90, 75, 75])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E1EC")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(table)
                story.extend([
                    Spacer(1, 14),
                    Paragraph(
                        "Note: PASS/FAIL counts are model-generated predictions from saved application records. "
                        "They are not verified academic outcomes and must not be interpreted as model accuracy.",
                        styles["Italic"]
                    )
                ])
                doc.build(story)
                pdf_buffer.seek(0)
                st.download_button(
                    "⬇️ Download PDF analytics report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"student_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("PDF report generated successfully.")
            except Exception as exc:
                st.error("Could not generate the PDF report.")
                st.code(str(exc))



# ============================================================
# PAGE 13 — STUDENT PROFILE MANAGEMENT
# ============================================================
elif page == "👤  Student Profiles":
    hero(
        "STUDENT DIRECTORY",
        "Student profile management",
        "Find a student by ID and review their saved prediction history and academic input details."
    )

    profile_data = load_prediction_history()
    if profile_data.empty:
        st.info("No student profiles are available yet. Make and save a prediction with a Student ID first.")
    else:
        profile_data["student_id"] = profile_data["student_id"].fillna("").astype(str).str.strip()
        profile_data["student_name"] = profile_data["student_name"].fillna("").astype(str).str.strip()
        profile_data = profile_data[profile_data["student_id"] != ""]

        if profile_data.empty:
            st.warning("No Student IDs were found in the saved prediction history.")
        else:
            student_options = sorted(profile_data["student_id"].unique().tolist())
            search_id = st.text_input("Search by Student ID", placeholder="Enter full or partial Student ID")
            if search_id.strip():
                filtered_options = [sid for sid in student_options if search_id.strip().lower() in sid.lower()]
            else:
                filtered_options = student_options

            if not filtered_options:
                st.warning("No matching Student ID found.")
            else:
                selected_id = st.selectbox("Select student", filtered_options)
                student_records = profile_data[profile_data["student_id"] == selected_id].copy()
                student_records["_sort_date"] = pd.to_datetime(
                    student_records["prediction_date"], errors="coerce"
                )
                student_records = student_records.sort_values(
                    "_sort_date", ascending=False, na_position="last"
                )
                latest = student_records.iloc[0]
                display_name = latest["student_name"] if latest["student_name"] else "Name not provided"

                st.markdown('<div class="section-title">Student overview</div>', unsafe_allow_html=True)
                st.subheader(display_name)
                st.caption(f"Student ID: {selected_id}")

                latest_outcome = str(latest["predicted_outcome"]).upper()
                latest_conf = pd.to_numeric(pd.Series([latest["confidence"]]), errors="coerce").iloc[0]
                latest_conf_text = f"{latest_conf:.1f}%" if pd.notna(latest_conf) else "Not available"

                p1, p2, p3 = st.columns(3)
                with p1:
                    metric_card("SAVED PREDICTIONS", f"{len(student_records):,}", "For this Student ID")
                with p2:
                    metric_card("LATEST PREDICTION", latest_outcome, "Most recent saved model output")
                with p3:
                    metric_card("LATEST CONFIDENCE", latest_conf_text, "Model confidence, not accuracy")

                academic_fields = [
                    "attendance", "internal_marks", "study_hours",
                    "previous_performance", "assignment_score"
                ]
                st.markdown('<div class="section-title">Latest academic inputs</div>', unsafe_allow_html=True)
                input_cols = st.columns(len(academic_fields))
                for col_box, field in zip(input_cols, academic_fields):
                    value = pd.to_numeric(pd.Series([latest[field]]), errors="coerce").iloc[0]
                    shown = f"{value:g}" if pd.notna(value) else "—"
                    with col_box:
                        metric_card(field.replace("_", " ").title(), shown, "Latest saved input")

                st.markdown('<div class="section-title">Prediction history</div>', unsafe_allow_html=True)
                history_cols = [
                    "prediction_date", "predicted_outcome", "confidence",
                    "pass_probability", "fail_probability"
                ]
                st.dataframe(
                    student_records[history_cols],
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown('<div class="section-title">Download individual student report</div>', unsafe_allow_html=True)
                try:
                    pdf_buffer = BytesIO()
                    doc = SimpleDocTemplate(
                        pdf_buffer, pagesize=A4, rightMargin=36, leftMargin=36,
                        topMargin=40, bottomMargin=36
                    )
                    styles = getSampleStyleSheet()
                    story = [
                        Paragraph("Individual Student Prediction Report", styles["Title"]),
                        Spacer(1, 10),
                        Paragraph(f"<b>Student name:</b> {display_name}", styles["Normal"]),
                        Paragraph(f"<b>Student ID:</b> {selected_id}", styles["Normal"]),
                        Paragraph(f"<b>Report generated:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["Normal"]),
                        Spacer(1, 14),
                        Paragraph("Latest saved academic inputs", styles["Heading2"]),
                    ]
                    input_table = [["Academic input", "Latest value"]]
                    for field in academic_fields:
                        value = pd.to_numeric(pd.Series([latest[field]]), errors="coerce").iloc[0]
                        input_table.append([field.replace("_", " ").title(), f"{value:g}" if pd.notna(value) else "Not available"])
                    t1 = Table(input_table, colWidths=[230, 150])
                    t1.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8E1EC")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                        ("PADDING", (0, 0), (-1, -1), 7),
                    ]))
                    story.extend([t1, Spacer(1, 14), Paragraph("Saved prediction history", styles["Heading2"])])
                    table_data = [["Date", "Prediction", "Confidence", "Pass probability", "Fail probability"]]
                    for _, record in student_records.iterrows():
                        def pct_text(raw):
                            number = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
                            return f"{number:.1f}%" if pd.notna(number) else "—"
                        table_data.append([
                            str(record["prediction_date"])[:16],
                            str(record["predicted_outcome"]),
                            pct_text(record["confidence"]),
                            pct_text(record["pass_probability"]),
                            pct_text(record["fail_probability"])
                        ])
                    t2 = Table(table_data, repeatRows=1, colWidths=[95, 70, 65, 85, 85])
                    t2.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E1EC")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]))
                    story.extend([
                        t2, Spacer(1, 12),
                        Paragraph(
                            "Note: This report summarizes saved model predictions and the input values "
                            "provided to the application. Predictions are not verified academic results.",
                            styles["Italic"]
                        )
                    ])
                    doc.build(story)
                    pdf_buffer.seek(0)
                    st.download_button(
                        "⬇️ Download student PDF report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"student_report_{selected_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as exc:
                    st.error("Could not prepare the individual student PDF report.")
                    st.code(str(exc))



# ============================================================
# PAGE 14 — STUDENT REGISTRY
# ============================================================
elif page == "🗃️  Student Registry":
    hero(
        "STUDENT RECORDS",
        "Student registry",
        "Register and maintain a basic student directory separately from prediction history."
    )

    REGISTRY_FILE = "student_registry.csv"
    REGISTRY_COLUMNS = ["student_id", "student_name", "course", "semester", "section", "created_at"]

    def load_student_registry():
        if not os.path.exists(REGISTRY_FILE):
            return pd.DataFrame(columns=REGISTRY_COLUMNS)
        try:
            registry = pd.read_csv(REGISTRY_FILE, dtype={"student_id": str})
            for col in REGISTRY_COLUMNS:
                if col not in registry.columns:
                    registry[col] = ""
            registry["student_id"] = registry["student_id"].fillna("").astype(str).str.strip()
            return registry[REGISTRY_COLUMNS]
        except Exception:
            return pd.DataFrame(columns=REGISTRY_COLUMNS)

    registry = load_student_registry()

    st.markdown('<div class="section-title">Register a student</div>', unsafe_allow_html=True)
    with st.form("student_registry_form", clear_on_submit=True):
        r1, r2 = st.columns(2)
        with r1:
            new_id = st.text_input("Student ID *", placeholder="e.g. CUTM2026001")
            new_name = st.text_input("Student name *", placeholder="Enter full name")
            course = st.text_input("Course / Programme", placeholder="e.g. B.Tech CSE (AIML)")
        with r2:
            semester = st.selectbox("Semester", ["Not specified", "1", "2", "3", "4", "5", "6", "7", "8"])
            section = st.text_input("Section / Batch", placeholder="Optional")
            st.caption("* Required fields")
        submitted = st.form_submit_button("➕ Register student", use_container_width=True)

    if submitted:
        clean_id = new_id.strip()
        clean_name = new_name.strip()
        if not clean_id or not clean_name:
            st.error("Student ID and student name are required.")
        elif clean_id in registry["student_id"].tolist():
            st.error("This Student ID is already registered. Student IDs must be unique.")
        else:
            new_record = {
                "student_id": clean_id,
                "student_name": clean_name,
                "course": course.strip(),
                "semester": semester,
                "section": section.strip(),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            registry = pd.concat([registry, pd.DataFrame([new_record])], ignore_index=True)
            registry.to_csv(REGISTRY_FILE, index=False)
            st.success(f"Student {clean_name} registered successfully.")
            st.rerun()

    st.markdown('<div class="section-title">Registered students</div>', unsafe_allow_html=True)
    registry = load_student_registry()
    if registry.empty:
        st.info("No students registered yet. Add the first student using the form above.")
    else:
        k1, k2 = st.columns(2)
        with k1:
            metric_card("REGISTERED STUDENTS", f"{len(registry):,}", "Unique Student IDs")
        with k2:
            linked_ids = set(load_prediction_history()["student_id"].fillna("").astype(str))
            linked_count = int(registry["student_id"].isin(linked_ids).sum())
            metric_card("WITH PREDICTION HISTORY", f"{linked_count:,}", "Registered students with saved predictions")

        search_registry = st.text_input("Search registered students", placeholder="Search by ID or name")
        view_registry = registry.copy()
        if search_registry.strip():
            query = search_registry.strip().lower()
            mask = (
                view_registry["student_id"].str.lower().str.contains(query, na=False) |
                view_registry["student_name"].fillna("").astype(str).str.lower().str.contains(query, na=False)
            )
            view_registry = view_registry[mask]

        st.dataframe(view_registry, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export student registry (CSV)",
            data=registry.to_csv(index=False).encode("utf-8"),
            file_name="student_registry.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown('<div class="section-title">Remove a student record</div>', unsafe_allow_html=True)
        st.caption("Removing a registry entry does not delete that student's saved prediction history.")
        remove_id = st.selectbox("Select Student ID to remove", registry["student_id"].tolist())
        if st.button("🗑️ Remove selected student", type="secondary"):
            registry = registry[registry["student_id"] != remove_id]
            registry.to_csv(REGISTRY_FILE, index=False)
            st.success("Student registry entry removed.")
            st.rerun()






# ============================================================
# PAGE — ADVANCED MODEL EVALUATION
# ============================================================
elif page == "📊  Advanced Model Evaluation":
    hero(
        "MACHINE LEARNING PERFORMANCE",
        "Advanced Model Evaluation",
        "Evaluate the saved model on a reproducible held-out test split and inspect classification metrics."
    )
    st.info(
        "Evaluation is calculated from the project's dataset using an 80:20 stratified split "
        "with random_state=42. Metrics shown here are computed when this page loads; they are "
        "not copied from the model_results.csv benchmark."
    )

    if "result" not in df.columns:
        st.error("Target column 'result' was not found in the dataset. Evaluation cannot run.")
    else:
        eval_df = df[feature_names + ["result"]].copy()
        eval_df["result"] = eval_df["result"].astype(str).str.strip().str.upper()
        eval_df = eval_df.dropna(subset=feature_names + ["result"])
        eval_df = eval_df[eval_df["result"].isin(["PASS", "FAIL"])]

        if eval_df.empty or eval_df["result"].nunique() < 2:
            st.error("The dataset must contain valid PASS and FAIL examples to calculate these metrics.")
        else:
            X_eval = eval_df[feature_names]
            y_eval = eval_df["result"]
            try:
                _, X_test_eval, _, y_test_eval = train_test_split(
                    X_eval, y_eval, test_size=0.20, random_state=42, stratify=y_eval
                )
                y_pred_eval = pd.Series(model.predict(X_test_eval), index=y_test_eval.index).astype(str).str.upper()
                classes_eval = [str(c).strip().upper() for c in model.classes_]
                if not set(y_test_eval.unique()).issubset(set(classes_eval)):
                    raise ValueError("Model classes do not match the PASS/FAIL labels in the dataset.")

                accuracy_eval = accuracy_score(y_test_eval, y_pred_eval)
                precision_eval = precision_score(y_test_eval, y_pred_eval, pos_label="PASS", zero_division=0)
                recall_eval = recall_score(y_test_eval, y_pred_eval, pos_label="PASS", zero_division=0)
                f1_eval = f1_score(y_test_eval, y_pred_eval, pos_label="PASS", zero_division=0)
                try:
                    prob_eval = model.predict_proba(X_test_eval)
                    pass_index = classes_eval.index("PASS")
                    pass_scores = prob_eval[:, pass_index]
                    auc_eval = roc_auc_score((y_test_eval == "PASS").astype(int), pass_scores)
                except Exception:
                    pass_scores = None
                    auc_eval = None

                st.markdown('<div class="section-title">Held-out test metrics</div>', unsafe_allow_html=True)
                metric_cols = st.columns(5)
                metric_items = [
                    ("ACCURACY", accuracy_eval),
                    ("PRECISION · PASS", precision_eval),
                    ("RECALL · PASS", recall_eval),
                    ("F1-SCORE · PASS", f1_eval),
                    ("ROC-AUC", auc_eval)
                ]
                for box, (label, value) in zip(metric_cols, metric_items):
                    with box:
                        metric_card(label, f"{value:.3f}" if value is not None else "N/A", "Held-out test split")

                st.caption(
                    f"Test samples: {len(y_test_eval)} | Dataset rows used: {len(eval_df)} | "
                    "Positive class for Precision/Recall/F1: PASS."
                )

                st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
                cm_eval = confusion_matrix(y_test_eval, y_pred_eval, labels=["FAIL", "PASS"])
                cm_df = pd.DataFrame(
                    cm_eval,
                    index=["Actual Fail", "Actual Pass"],
                    columns=["Predicted Fail", "Predicted Pass"]
                )
                st.dataframe(cm_df, use_container_width=True)
                st.caption("Rows represent actual classes; columns represent model predictions.")

                st.markdown('<div class="section-title">Classification report</div>', unsafe_allow_html=True)
                report_eval = classification_report(
                    y_test_eval, y_pred_eval, labels=["FAIL", "PASS"],
                    target_names=["Fail", "Pass"], output_dict=True, zero_division=0
                )
                report_df = pd.DataFrame(report_eval).transpose().round(3)
                st.dataframe(report_df, use_container_width=True)

                if pass_scores is not None and len(np.unique((y_test_eval == "PASS").astype(int))) == 2:
                    fpr_eval, tpr_eval, _ = roc_curve((y_test_eval == "PASS").astype(int), pass_scores)
                    roc_df = pd.DataFrame({"False Positive Rate": fpr_eval, "True Positive Rate": tpr_eval})
                    st.markdown('<div class="section-title">ROC curve</div>', unsafe_allow_html=True)
                    st.line_chart(roc_df.set_index("False Positive Rate"), use_container_width=True)
                    st.caption(f"ROC-AUC = {auc_eval:.3f}. AUC measures ranking separation, not overall accuracy.")

                report_csv = report_df.to_csv().encode("utf-8")
                st.download_button(
                    "⬇️ Download classification report (CSV)",
                    data=report_csv,
                    file_name="advanced_model_evaluation.csv",
                    mime="text/csv"
                )

                st.markdown('<div class="section-title">How to explain these metrics in viva</div>', unsafe_allow_html=True)
                st.write("- **Accuracy:** fraction of all test predictions that were correct.")
                st.write("- **Precision (Pass):** among students predicted Pass, how many were actually Pass.")
                st.write("- **Recall (Pass):** among actual Pass students, how many the model identified.")
                st.write("- **F1-score (Pass):** harmonic mean of Pass precision and recall.")
                st.write("- **ROC-AUC:** how well the model ranks Pass examples above Fail examples across thresholds.")
                st.warning(
                    "This is a fresh hold-out evaluation using the documented split settings. "
                    "If the original model was trained with different preprocessing, rows, or split settings, "
                    "the result may differ from the project's earlier reported accuracy. Do not describe "
                    "these values as an independent external validation."
                )
            except Exception as exc:
                st.error("Evaluation could not be completed. Check that the saved model and dataset use the same features and label encoding.")
                st.code(str(exc))

# ============================================================
# PAGE — STUDENT PROGRESS TRACKER
# ============================================================
elif page == "📅  Student Progress Tracker":
    hero(
        "LONGITUDINAL STUDENT MONITORING",
        "Student Progress Tracker",
        "Review saved prediction history and observe how a student's academic indicators change over time."
    )

    history = load_prediction_history()
    if history.empty:
        st.warning("No prediction history is available yet. Make and save predictions first.")
    else:
        history["prediction_date"] = pd.to_datetime(history["prediction_date"], errors="coerce")
        history["student_id"] = history["student_id"].fillna("").astype(str).str.strip()
        history["student_name"] = history["student_name"].fillna("").astype(str).str.strip()

        identified = history[
            (history["student_id"] != "") |
            (history["student_name"] != "")
        ].copy()

        if identified.empty:
            st.info("Saved history does not contain student names or IDs yet. Add student details when making predictions.")
        else:
            identified["_student_label"] = identified.apply(
                lambda r: (
                    (r["student_name"] if r["student_name"] else "Unnamed student")
                    + " — "
                    + (r["student_id"] if r["student_id"] else "No ID")
                ),
                axis=1
            )
            student_options = sorted(identified["_student_label"].unique().tolist())
            selected_student = st.selectbox("Select student", student_options)
            student_history = identified[
                identified["_student_label"] == selected_student
            ].copy().sort_values("prediction_date")

            numeric_cols = [
                "confidence", "pass_probability", "fail_probability",
                "attendance", "internal_marks", "study_hours",
                "previous_performance", "assignment_score"
            ]
            for col in numeric_cols:
                student_history[col] = pd.to_numeric(student_history[col], errors="coerce")

            st.markdown('<div class="section-title">Progress summary</div>', unsafe_allow_html=True)
            first = student_history.iloc[0]
            latest = student_history.iloc[-1]
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                metric_card("SAVED PREDICTIONS", str(len(student_history)), "Records for selected student")
            with k2:
                latest_outcome = str(latest["predicted_outcome"]) if pd.notna(latest["predicted_outcome"]) else "—"
                metric_card("LATEST PREDICTION", latest_outcome, "Most recent saved model output")
            with k3:
                delta_pass = (
                    float(latest["pass_probability"]) - float(first["pass_probability"])
                    if pd.notna(latest["pass_probability"]) and pd.notna(first["pass_probability"]) else None
                )
                metric_card(
                    "PASS PROBABILITY CHANGE",
                    f"{delta_pass:+.1f} pp" if delta_pass is not None else "—",
                    "Latest minus earliest saved record"
                )
            with k4:
                last_date = latest["prediction_date"].strftime("%d %b %Y") if pd.notna(latest["prediction_date"]) else "Date unavailable"
                metric_card("LATEST RECORD", last_date, "Saved prediction date")

            st.markdown('<div class="section-title">Prediction probability trend</div>', unsafe_allow_html=True)
            probability_trend = student_history.dropna(subset=["prediction_date"]).set_index("prediction_date")
            if len(probability_trend) >= 2 and probability_trend[["pass_probability", "fail_probability"]].notna().any().any():
                st.line_chart(probability_trend[["pass_probability", "fail_probability"]], use_container_width=True)
                st.caption("Trend shows saved model probabilities over time; it is not a measure of confirmed academic results.")
            else:
                st.info("At least two dated prediction records are needed to display a probability trend.")

            st.markdown('<div class="section-title">Academic indicator trends</div>', unsafe_allow_html=True)
            indicator_cols = [
                "attendance", "internal_marks", "study_hours",
                "previous_performance", "assignment_score"
            ]
            indicator_labels = {
                "attendance": "Attendance (%)",
                "internal_marks": "Internal marks",
                "study_hours": "Study hours/day",
                "previous_performance": "Previous performance",
                "assignment_score": "Assignment score"
            }
            selected_indicators = st.multiselect(
                "Choose indicators to compare",
                indicator_cols,
                default=["attendance", "internal_marks", "assignment_score"],
                format_func=lambda x: indicator_labels[x]
            )
            indicator_trend = student_history.dropna(subset=["prediction_date"]).set_index("prediction_date")
            if selected_indicators and len(indicator_trend) >= 2:
                chart = indicator_trend[selected_indicators].rename(columns=indicator_labels)
                st.line_chart(chart, use_container_width=True)
            elif len(student_history) < 2:
                st.info("Save more predictions for this student to build an indicator trend.")
            else:
                st.info("Select at least one indicator to display its trend.")

            st.markdown('<div class="section-title">Saved prediction records</div>', unsafe_allow_html=True)
            display_cols = [
                "prediction_date", "predicted_outcome", "confidence",
                "pass_probability", "fail_probability"
            ] + indicator_cols
            display_history = student_history[display_cols].copy()
            display_history["prediction_date"] = display_history["prediction_date"].dt.strftime("%d %b %Y %I:%M %p")
            st.dataframe(display_history, use_container_width=True, hide_index=True)

            csv_data = student_history.drop(columns=["_student_label"], errors="ignore").to_csv(index=False).encode("utf-8")
            safe_filename = "".join(ch for ch in selected_student if ch.isalnum() or ch in "-_")[:50] or "student"
            st.download_button(
                "⬇️ Export this student's progress (CSV)",
                data=csv_data,
                file_name=f"{safe_filename}_progress.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.caption(
                "Important: The tracker visualises saved model predictions and input values only. "
                "It does not independently verify attendance, marks, or final academic outcomes."
            )

# ============================================================
# PAGE — EXPLAINABLE AI DASHBOARD
# ============================================================
elif page == "🧠  Explainable AI":
    hero(
        "MODEL TRANSPARENCY",
        "Explainable AI Dashboard",
        "Understand the model prediction, inspect influential inputs, and explore what-if changes."
    )

    st.info(
        "This page explains the saved model's output. The what-if chart changes one input at a time "
        "while holding the other inputs constant. These are sensitivity checks, not causal explanations."
    )

    registered = load_student_registry()
    chosen_name, chosen_id = "", ""
    if not registered.empty:
        opts = ["Manual input"] + [
            f'{row["student_id"]} — {row["student_name"]}'
            for _, row in registered.iterrows()
        ]
        selected = st.selectbox("Select a registered student (optional)", opts, key="xai_student")
        if selected != "Manual input":
            sid = selected.split(" — ", 1)[0]
            row = registered[registered["student_id"].astype(str) == sid].iloc[0]
            chosen_name, chosen_id = str(row["student_name"]), str(row["student_id"])

    st.markdown('<div class="section-title">Student academic inputs</div>', unsafe_allow_html=True)
    with st.form("xai_prediction_form"):
        n1, n2, n3 = st.columns(3)
        with n1:
            xai_name = st.text_input("Student name", value=chosen_name, key="xai_name")
            xai_attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0, 1.0, key="xai_att")
            xai_internal = st.number_input("Internal marks", 0.0, 100.0, 60.0, 1.0, key="xai_int")
        with n2:
            xai_id = st.text_input("Student ID", value=chosen_id, key="xai_id")
            xai_study = st.number_input("Study hours/day", 0.0, 24.0, 3.0, 0.5, key="xai_study")
        with n3:
            xai_previous = st.number_input("Previous performance", 0.0, 100.0, 65.0, 1.0, key="xai_prev")
            xai_assignment = st.number_input("Assignment score", 0.0, 100.0, 70.0, 1.0, key="xai_assign")
        run_xai = st.form_submit_button("🧠 Explain this prediction", use_container_width=True)

    if run_xai:
        xai_values = {
            "attendance": xai_attendance,
            "internal_marks": xai_internal,
            "study_hours": xai_study,
            "previous_performance": xai_previous,
            "assignment_score": xai_assignment
        }
        xai_frame = pd.DataFrame([[xai_values[f] for f in feature_names]], columns=feature_names)
        try:
            xai_prediction = str(model.predict(xai_frame)[0]).strip().upper()
            xai_probabilities = model.predict_proba(xai_frame)[0]
            xai_classes = [str(c).strip().upper() for c in model.classes_]
            pass_prob = float(xai_probabilities[xai_classes.index("PASS")] * 100) if "PASS" in xai_classes else 0.0
            fail_prob = float(xai_probabilities[xai_classes.index("FAIL")] * 100) if "FAIL" in xai_classes else 0.0

            # Risk bands are demo interpretation bands, not institutional policy.
            if fail_prob >= 60:
                risk_label, risk_color, risk_note = "HIGH", "#c0392b", "The model assigns a relatively high probability to the Fail class."
            elif fail_prob >= 30:
                risk_label, risk_color, risk_note = "MEDIUM", "#d68910", "The model indicates a moderate Fail probability; review the input profile."
            else:
                risk_label, risk_color, risk_note = "LOW", "#16845b", "The model assigns a relatively low probability to the Fail class."

            st.session_state["xai_result"] = {
                "name": xai_name.strip() or "Not provided",
                "student_id": xai_id.strip() or "Not provided",
                "values": xai_values,
                "prediction": xai_prediction,
                "pass_probability": pass_prob,
                "fail_probability": fail_prob,
                "risk_label": risk_label,
                "risk_color": risk_color,
                "risk_note": risk_note
            }
        except Exception as exc:
            st.error("The model could not explain this input. Please verify the feature values.")
            st.code(str(exc))

    result = st.session_state.get("xai_result")
    if result:
        st.markdown('<div class="section-title">Prediction and risk overview</div>', unsafe_allow_html=True)
        a, b, c, d = st.columns(4)
        with a:
            metric_card("PREDICTED OUTCOME", result["prediction"], "Saved ML model output")
        with b:
            metric_card("PASS PROBABILITY", f'{result["pass_probability"]:.1f}%', "Model probability")
        with c:
            metric_card("FAIL PROBABILITY", f'{result["fail_probability"]:.1f}%', "Model probability")
        with d:
            metric_card("DEMO RISK BAND", result["risk_label"], "Based on fail probability")

        st.markdown(
            f'<div style="background:#fff;border-left:6px solid {result["risk_color"]};'
            f'border-radius:12px;padding:16px 20px;margin:12px 0;">'
            f'<b>Risk interpretation: {result["risk_label"]}</b><br>{result["risk_note"]}</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Risk bands used in this demonstration: Low <30% fail probability; "
            "Medium 30–59.9%; High ≥60%. These are illustrative thresholds, not official CUTM criteria."
        )

        st.markdown('<div class="section-title">What-if sensitivity analysis</div>', unsafe_allow_html=True)
        st.write(
            "For each input, the app replaces only that value with the dataset median and recalculates "
            "Pass probability. A larger change means the model output is more sensitive to that input "
            "for this particular student profile."
        )
        sensitivity_rows = []
        baseline_pass = result["pass_probability"]
        for feature in feature_names:
            changed = result["values"].copy()
            median_value = pd.to_numeric(df[feature], errors="coerce").median()
            if pd.isna(median_value):
                continue
            changed[feature] = float(median_value)
            changed_frame = pd.DataFrame([[changed[f] for f in feature_names]], columns=feature_names)
            try:
                changed_probs = model.predict_proba(changed_frame)[0]
                changed_classes = [str(c).strip().upper() for c in model.classes_]
                changed_pass = float(changed_probs[changed_classes.index("PASS")] * 100) if "PASS" in changed_classes else 0.0
                sensitivity_rows.append({
                    "Input changed": feature.replace("_", " ").title(),
                    "Current value": result["values"][feature],
                    "Dataset median": round(float(median_value), 2),
                    "Pass probability after change (%)": round(changed_pass, 2),
                    "Change vs current (percentage points)": round(changed_pass - baseline_pass, 2),
                    "Absolute sensitivity": round(abs(changed_pass - baseline_pass), 2)
                })
            except Exception:
                continue

        if sensitivity_rows:
            sensitivity_df = pd.DataFrame(sensitivity_rows).sort_values(
                "Absolute sensitivity", ascending=False
            )
            chart_df = sensitivity_df.set_index("Input changed")[["Absolute sensitivity"]]
            st.bar_chart(chart_df, use_container_width=True)
            st.dataframe(
                sensitivity_df.drop(columns=["Absolute sensitivity"]),
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "A positive change means the one-at-a-time median substitution increased Pass probability; "
                "a negative change means it decreased. This does not prove that changing the real-world factor causes an outcome."
            )

        st.markdown('<div class="section-title">Model-level feature importance</div>', unsafe_allow_html=True)
        if "feature_importance" in globals() and not feature_importance.empty and feature_importance["Importance"].sum() > 0:
            importance_view = feature_importance.copy()
            importance_view["Feature"] = importance_view["Feature"].str.replace("_", " ").str.title()
            st.bar_chart(importance_view.set_index("Feature")[["Importance"]], use_container_width=True)
            st.caption(
                "This view uses absolute Logistic Regression coefficient magnitudes from the saved pipeline. "
                "It indicates model association/weight magnitude, not causation; interpretation may depend on preprocessing and feature scale."
            )
        else:
            st.warning(
                "Coefficient-based feature importance is unavailable for the currently loaded model. "
                "Use the what-if sensitivity analysis above for this individual prediction."
            )

        st.markdown('<div class="section-title">Practical next steps</div>', unsafe_allow_html=True)
        advice = []
        if result["fail_probability"] >= 30:
            advice.append("Review the student's academic inputs with a faculty mentor and identify the specific topics needing support.")
        if result["values"]["attendance"] < 75:
            advice.append("Check attendance regularly and confirm the applicable attendance requirement with the department.")
        if result["values"]["internal_marks"] < 50 or result["values"]["assignment_score"] < 50:
            advice.append("Review assessment feedback and practise the topics/questions where marks were lost.")
        if result["values"]["study_hours"] < 2:
            advice.append("Build a realistic, consistent study routine and revise class topics frequently.")
        if not advice:
            advice.append("Maintain consistent study habits and review progress after the next assessment.")
        for item in advice:
            st.markdown("• " + item)

        st.warning(
            "Important: This is an educational decision-support demonstration. Predictions and risk bands "
            "are not guaranteed outcomes and must not be used as the sole basis for academic decisions."
        )

# ============================================================
# PAGE — AI-ASSISTED STUDENT IMPROVEMENT ADVISOR
# ============================================================
elif page == "🎯  Improvement Advisor":
    hero(
        "PERSONALISED ACADEMIC GUIDANCE",
        "Student Improvement Advisor",
        "Review academic indicators and generate a practical improvement plan."
    )

    st.info(
        "This is a transparent, rule-based advisory module using the five project inputs. "
        "Its guide thresholds are generic demo values—not official CUTM rules or a separate trained AI model."
    )

    registry_df = load_student_registry() if "load_student_registry" in globals() else pd.DataFrame()
    history_df = load_prediction_history()
    selected_name, selected_id = "", ""
    selected_values = None

    if not registry_df.empty:
        registered_options = ["Manual entry"] + [
            f"{row['student_name']} ({row['student_id']})"
            for _, row in registry_df.iterrows()
        ]
        chosen = st.selectbox("Choose a registered student (optional)", registered_options)
        if chosen != "Manual entry":
            chosen_row = registry_df[
                registry_df.apply(
                    lambda r: f"{r['student_name']} ({r['student_id']})" == chosen, axis=1
                )
            ].iloc[0]
            selected_name = str(chosen_row["student_name"])
            selected_id = str(chosen_row["student_id"])
            if not history_df.empty and "student_id" in history_df.columns:
                records = history_df[
                    history_df["student_id"].fillna("").astype(str) == selected_id
                ].copy()
                if not records.empty:
                    records["_date_sort"] = pd.to_datetime(
                        records["prediction_date"], errors="coerce"
                    )
                    latest_record = records.sort_values("_date_sort").iloc[-1]
                    selected_values = {}
                    for feature in feature_names:
                        selected_values[feature] = pd.to_numeric(
                            pd.Series([latest_record.get(feature)]), errors="coerce"
                        ).iloc[0]
                    st.success("Latest saved prediction inputs loaded for this student.")
                else:
                    st.warning("No saved prediction found for this student. Enter values below.")

    st.markdown('<div class="section-title">Academic indicators</div>', unsafe_allow_html=True)
    with st.form("improvement_advisor_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            advisor_name = st.text_input("Student name", value=selected_name)
            attendance = st.number_input(
                "Attendance (%)", min_value=0.0, max_value=100.0,
                value=float(selected_values["attendance"]) if selected_values is not None and pd.notna(selected_values["attendance"]) else 75.0,
                step=1.0
            )
        with c2:
            advisor_id = st.text_input("Student ID", value=selected_id)
            internal_marks = st.number_input(
                "Internal marks", min_value=0.0, max_value=100.0,
                value=float(selected_values["internal_marks"]) if selected_values is not None and pd.notna(selected_values["internal_marks"]) else 60.0,
                step=1.0
            )
        with c3:
            study_hours = st.number_input(
                "Study hours per day", min_value=0.0, max_value=24.0,
                value=float(selected_values["study_hours"]) if selected_values is not None and pd.notna(selected_values["study_hours"]) else 2.0,
                step=0.5
            )
            previous_performance = st.number_input(
                "Previous performance", min_value=0.0, max_value=100.0,
                value=float(selected_values["previous_performance"]) if selected_values is not None and pd.notna(selected_values["previous_performance"]) else 60.0,
                step=1.0
            )
        assignment_score = st.number_input(
            "Assignment score", min_value=0.0, max_value=100.0,
            value=float(selected_values["assignment_score"]) if selected_values is not None and pd.notna(selected_values["assignment_score"]) else 60.0,
            step=1.0
        )
        generate_advice = st.form_submit_button("✨ Generate improvement plan", use_container_width=True)

    if generate_advice:
        values = {
            "attendance": attendance,
            "internal_marks": internal_marks,
            "study_hours": study_hours,
            "previous_performance": previous_performance,
            "assignment_score": assignment_score
        }
        recommendations = []
        priorities = []

        if attendance < 75:
            recommendations.append(
                "Attendance is below the demo guide level of 75%. Aim to attend upcoming classes consistently and check the official attendance requirement with your department."
            )
            priorities.append(("High", "Improve attendance", "Plan class attendance and avoid preventable absences."))
        if study_hours < 2:
            recommendations.append(
                "Recorded study time is below the demo guide level of 2 hours/day. Start with a realistic daily study block and revise topics taught in class."
            )
            priorities.append(("Medium", "Build a daily study routine", "Begin with 1–2 focused sessions and review progress weekly."))
        if internal_marks < 50:
            recommendations.append(
                "Internal marks are below the demo guide level of 50/100. Review mistakes, clarify difficult concepts with faculty, and practise similar questions."
            )
            priorities.append(("High", "Strengthen internal assessment", "List weak topics and practise questions before the next assessment."))
        if assignment_score < 50:
            recommendations.append(
                "Assignment score is below the demo guide level of 50/100. Check feedback, understand the marking rubric, and submit complete work before deadlines."
            )
            priorities.append(("Medium", "Improve assignment work", "Use feedback as a checklist for the next submission."))
        if previous_performance < 50:
            recommendations.append(
                "Previous performance is below the demo guide level of 50/100. Identify recurring gaps and use short, regular revision sessions."
            )
            priorities.append(("High", "Revise foundational topics", "Prioritise concepts that caused difficulty in earlier assessments."))

        if not recommendations:
            recommendations.append(
                "None of the demo guidance thresholds were triggered. Maintain consistency and review your progress regularly."
            )
            priorities.append(("Maintain", "Continue current habits", "Keep a steady routine and seek feedback to identify the next learning goal."))

        st.session_state["advisor_result"] = {
            "name": advisor_name.strip() or "Not provided",
            "student_id": advisor_id.strip() or "Not provided",
            "values": values,
            "recommendations": recommendations,
            "priorities": priorities
        }

    result = st.session_state.get("advisor_result")
    if result:
        st.markdown('<div class="section-title">Your improvement overview</div>', unsafe_allow_html=True)
        metric_cols = st.columns(5)
        labels = [
            ("Attendance", "attendance", "%"),
            ("Internal", "internal_marks", "/100"),
            ("Study/day", "study_hours", " hrs"),
            ("Previous", "previous_performance", "/100"),
            ("Assignment", "assignment_score", "/100")
        ]
        for box, (label, key, suffix) in zip(metric_cols, labels):
            with box:
                metric_card(label.upper(), f"{result['values'][key]:g}{suffix}", "Entered indicator")

        st.markdown('<div class="section-title">Priority action plan</div>', unsafe_allow_html=True)
        plan_df = pd.DataFrame(
            result["priorities"], columns=["Priority", "Focus area", "Suggested action"]
        )
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Personalised recommendations</div>', unsafe_allow_html=True)
        for index, recommendation in enumerate(result["recommendations"], start=1):
            st.markdown(f"**{index}.** {recommendation}")

        pdf_buffer = BytesIO()
        pdf_doc = SimpleDocTemplate(
            pdf_buffer, pagesize=A4, rightMargin=42, leftMargin=42,
            topMargin=42, bottomMargin=42
        )
        pdf_styles = getSampleStyleSheet()
        pdf_story = [
            Paragraph("Student Improvement Plan", pdf_styles["Title"]),
            Spacer(1, 8),
            Paragraph(f"<b>Student:</b> {result['name']}", pdf_styles["Normal"]),
            Paragraph(f"<b>Student ID:</b> {result['student_id']}", pdf_styles["Normal"]),
            Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}", pdf_styles["Normal"]),
            Spacer(1, 14),
            Paragraph("Academic indicators", pdf_styles["Heading2"])
        ]
        pdf_rows = [["Indicator", "Value"]] + [
            [label, f"{result['values'][key]:g}{suffix}"]
            for label, key, suffix in labels
        ]
        pdf_table = Table(pdf_rows, colWidths=[250, 170])
        pdf_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173653")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7e0e9")),
            ("PADDING", (0, 0), (-1, -1), 8)
        ]))
        pdf_story.extend([pdf_table, Spacer(1, 14), Paragraph("Recommendations", pdf_styles["Heading2"])])
        for item in result["recommendations"]:
            safe_item = item.replace("&", "&amp;")
            pdf_story.extend([Paragraph("• " + safe_item, pdf_styles["BodyText"]), Spacer(1, 6)])
        pdf_story.extend([
            Spacer(1, 12),
            Paragraph(
                "Disclaimer: This is rule-based educational guidance using generic demo thresholds. "
                "It is not an official CUTM policy, a medical or psychological assessment, or a guaranteed outcome. "
                "Discuss academic concerns with the relevant faculty or advisor.",
                pdf_styles["Italic"]
            )
        ])
        pdf_doc.build(pdf_story)
        pdf_buffer.seek(0)
        safe_id = "".join(ch for ch in result["student_id"] if ch.isalnum() or ch in "-_") or "student"
        st.download_button(
            "⬇️ Download improvement plan (PDF)",
            data=pdf_buffer.getvalue(),
            file_name=f"improvement_plan_{safe_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ============================================================
# PAGE 14 — ABOUT PROJECT
# ============================================================
elif page == "ℹ️  About Project":
    hero("PROJECT INFORMATION", "AI-Based Student Academic Performance Prediction and Analysis System", "An academic demonstration of machine-learning-based student outcome prediction.")

    st.markdown("""
    <div class="info-card">
    <h3>Project objective</h3>
    <p>To estimate a student's Pass/Fail outcome using selected academic and study-related indicators,
    and present the result through an interactive web dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Input features</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Feature": ["Attendance", "Internal marks", "Study hours", "Previous performance", "Assignment score"],
        "Description": [
            "Class attendance percentage",
            "Internal assessment score",
            "Daily study time in hours",
            "Previous academic score",
            "Coursework assignment score"
        ]
    }), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Machine learning workflow</div>', unsafe_allow_html=True)
    steps = ["Dataset collection", "Data cleaning and exploration", "Preprocessing",
             "Model training and comparison", "Model evaluation", "Prediction and guidance"]
    for index, step in enumerate(steps, start=1):
        st.markdown(f"**{index:02d}.** {step}")

    st.markdown('<div class="section-title">Technology stack</div>', unsafe_allow_html=True)
    st.markdown("Python · Pandas · NumPy · Scikit-learn · Streamlit · Joblib · Jupyter Notebook")

    st.markdown("**Centurion University of Technology and Management (CUTM)**")
    st.info("This application is intended for educational and demonstration purposes. Predictions should not be treated as guaranteed academic outcomes.")

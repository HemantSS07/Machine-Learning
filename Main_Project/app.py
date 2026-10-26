
from pathlib import Path
from datetime import datetime, timezone
import json
import uuid
import ast

import numpy as np
import pandas as pd
import joblib
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "artifacts" / "career_guidance_model.joblib"
TRAIN_PATH = BASE / "career_guidance_train.csv"
QUESTIONS_PATH = BASE / "AI_Career_Guidance_Questions.xlsx"
RESULTS_PATH = BASE / "assessment_results.csv"
TARGET = "recommended_career_domain"

st.set_page_config(
    page_title="CareerCompass AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 CareerCompass AI")
st.subheader("Psychometric Assessment & ML-Based Career Matching")
st.write(
    "Answer the questionnaire honestly. Your mapped responses are converted "
    "into model features and used to rank career domains."
)

st.info(
    "Academic prototype: the model was trained on synthetic data. "
    "Recommendations are exploratory guidance, not a validated psychological "
    "assessment or a guarantee of career success."
)

if not all(p.exists() for p in [MODEL_PATH, TRAIN_PATH, QUESTIONS_PATH]):
    st.error("A required model, dataset, or questionnaire file is missing.")
    st.stop()

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    train = pd.read_csv(TRAIN_PATH)
    questions = pd.read_excel(QUESTIONS_PATH, sheet_name="Questions Dataset")
    return model, train, questions

model, train_df, questions_df = load_assets()
FEATURES = [c for c in train_df.columns if c != TARGET]
CATEGORICAL = ["age_group", "education_level", "experience_stage", "region_type"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

if len(FEATURES) != 25 or getattr(model, "n_features_in_", None) != 25:
    st.error("Model feature validation failed.")
    st.stop()

# Explicit, reviewable mappings: unmapped questions are stored but not
# arbitrarily converted into ML features.
QUESTION_FEATURES = {
    "Q001": ["interest_technology", "interest_problem_solving"],
    "Q002": ["interest_data"],
    "Q003": ["interest_technology", "interest_problem_solving"],
    "Q004": ["interest_creativity"],
    "Q005": ["interest_people"],
    "Q006": ["interest_helping"],
    "Q007": ["interest_people", "interest_helping"],
    "Q008": ["interest_creativity", "interest_data"],
    "Q009": ["interest_data", "interest_problem_solving"],
    "Q010": ["interest_technology", "interest_problem_solving"],
    "Q021": ["trait_extraversion", "trait_agreeableness"],
    "Q022": ["trait_conscientiousness"],
    "Q023": ["trait_extraversion"],
    "Q024": ["trait_conscientiousness"],
    "Q025": ["trait_openness"],
    "Q026": ["trait_emotional_stability"],
    "Q027": ["trait_conscientiousness"],
    "Q028": ["trait_extraversion"],
    "Q029": ["trait_conscientiousness"],
    "Q030": ["trait_openness"],
    "Q031": ["skill_programming"],
    "Q032": ["skill_data_analysis"],
    "Q036": ["skill_leadership"],
    "Q040": ["skill_research"],
    "Q123": ["skill_communication"],
}

APTITUDE_KEYS = {
    "Q011": "B", "Q012": "D", "Q013": "C", "Q014": "A",
    "Q015": "C", "Q016": "B", "Q017": "B", "Q018": "A",
    "Q019": "C", "Q020": "C", "Q121": "B", "Q122": "A"
}

APTITUDE_FEATURE_QUESTIONS = {
    "aptitude_logical": ["Q012", "Q014", "Q016", "Q019"],
    "aptitude_numerical": ["Q011", "Q013", "Q015", "Q017", "Q020"],
    "aptitude_critical_thinking": ["Q015", "Q016", "Q018", "Q019"],
    "aptitude_verbal": ["Q121"],
    "aptitude_spatial": ["Q122"],
}

MIN_ANSWERED = {
    "aptitude_logical": 3,
    "aptitude_numerical": 3,
    "aptitude_critical_thinking": 3,
    "aptitude_verbal": 1,
    "aptitude_spatial": 1,
}

CAREER_INFO = {
    "Business & Management": "Business operations, planning and management.",
    "Creative Arts & Media": "Creative production, media and visual communication.",
    "Cybersecurity": "Security operations, risk analysis and digital protection.",
    "Data Science & Analytics": "Data analysis, statistics and machine learning.",
    "Digital Marketing": "Digital campaigns, branding and marketing analytics.",
    "Education & Research": "Teaching, academic research and learning design.",
    "Engineering & Core Technology": "Engineering systems and technical problem-solving.",
    "Finance & Accounting": "Accounting, budgeting and financial analysis.",
    "Healthcare & Life Sciences": "Healthcare and biological or life-science fields.",
    "Law & Public Service": "Legal services, governance and public administration.",
    "Software Development": "Programming, software engineering and application development.",
    "UI/UX Design": "User research, interface design and usability."
}

def parse_options(raw):
    if raw is None or pd.isna(raw):
        return ["Not answered"]

    text = str(raw).strip()

    # Some questions store their options as a JSON object.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [f"{k}: {v}" for k, v in obj.items()]
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except (json.JSONDecodeError, TypeError):
        pass

    # Otherwise split the plain-text options.
    values = [x.strip() for x in text.split(",") if x.strip()]
    return values or [text]

def clean_answer(value):
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value

def score_assessment(answers):
    # Collect rating-derived feature values. A 1-5 response is mapped
    # linearly to the model's 2-10 scale.
    grouped = {feature: [] for feature in NUMERIC}

    for qid, mapped_features in QUESTION_FEATURES.items():
        answer = answers.get(qid)
        if answer is None or answer == "N/A":
            continue

        try:
            rating = float(answer)
        except (TypeError, ValueError):
            continue

        if 1 <= rating <= 5:
            score = rating * 2
            for feature in mapped_features:
                if feature in grouped:
                    grouped[feature].append(score)

    feature_scores = {}
    for feature, values in grouped.items():
        if values:
            feature_scores[feature] = round(float(np.mean(values)), 2)

    # Score aptitude MCQs using the stored answer key.
    for feature, qids in APTITUDE_FEATURE_QUESTIONS.items():
        attempted = [
            qid for qid in qids
            if answers.get(qid) in ["A", "B", "C", "D"]
        ]
        correct = sum(
            answers[qid] == APTITUDE_KEYS[qid] for qid in attempted
        )

        if len(attempted) >= MIN_ANSWERED[feature]:
            accuracy = correct / len(attempted)
            feature_scores[feature] = round(1 + accuracy * 9, 2)
        else:
            feature_scores[feature] = np.nan

    # Ensure every model numeric feature is present.
    for feature in NUMERIC:
        feature_scores.setdefault(feature, np.nan)

    return feature_scores

def append_result(record):
    results = pd.DataFrame([record])
    if RESULTS_PATH.exists():
        results.to_csv(
            RESULTS_PATH, mode="a", header=False,
            index=False, encoding="utf-8-sig"
        )
    else:
        results.to_csv(
            RESULTS_PATH, index=False, encoding="utf-8-sig"
        )

# Three additional demographic prompts required by the trained model.
st.sidebar.header("Assessment")
st.sidebar.write(f"Original questionnaire: {len(questions_df)} questions")
st.sidebar.write("Demographic prompts: 3")
st.sidebar.write(f"Total prompts: {len(questions_df) + 3}")
st.sidebar.caption("Your submitted answers are saved locally in a CSV file.")

with st.form("full_assessment"):
    st.header("Part A — Student profile")

    c1, c2 = st.columns(2)
    age_group = c1.selectbox(
        "Q124 — Your age group",
        ["15-18", "19-22", "23-30", "31+"]
    )
    region_type = c2.selectbox(
        "Q125 — Your residential area",
        ["Urban", "Semi-urban", "Rural", "Prefer not to say"]
    )

    education_options = sorted(
        train_df["education_level"].dropna().astype(str).unique().tolist()
    )
    education_options += ["Other / Not listed", "No formal qualification"]
    education_choice = st.selectbox(
        "Highest/current qualification",
        education_options
    )
    education_level = (
        education_choice
        if education_choice in set(train_df["education_level"].dropna())
        else None
    )

    experience_stage = st.selectbox(
        "Q126 — Your current education/work stage",
        [
            "Student",
            "Entry-level",
            "1-3 years",
            "4+ years",
            "Other / Prefer not to say"
        ]
    )
    if experience_stage == "Other / Prefer not to say":
        experience_stage = None

    st.header("Part B — Questionnaire")
    st.caption(
        "Questions are grouped by their original sections. "
        "Questions without a defined model mapping are still recorded."
    )

    answers = {}
    for section, section_df in questions_df.groupby("Section", sort=False):
        with st.expander(f"{section} ({len(section_df)} questions)", expanded=(section == "Interest")):
            for _, q in section_df.iterrows():
                qid = str(q["Question_ID"]).strip()
                question_text = str(q["Question_Text"]).strip()
                answer_type = str(q["Answer_Type"]).strip()
                options = parse_options(q.get("Answer_Options"))

                if answer_type == "MCQ":
                    # Display option labels, but store the option key when
                    # the workbook supplies a JSON A/B/C/D object.
                    raw = q.get("Answer_Options")
                    try:
                        parsed = json.loads(str(raw))
                    except (json.JSONDecodeError, TypeError):
                        parsed = None

                    if isinstance(parsed, dict):
                        labels = list(parsed.values())
                        keys = list(parsed.keys())
                        selected = st.radio(
                            f"{qid}. {question_text}",
                            ["Not answered"] + labels,
                            key=qid
                        )
                        if selected == "Not answered":
                            answers[qid] = None
                        else:
                            answers[qid] = keys[labels.index(selected)]
                    else:
                        selected = st.radio(
                            f"{qid}. {question_text}",
                            ["Not answered"] + options,
                            key=qid
                        )

                        if selected == "Not answered":
                            answers[qid] = None
                        elif qid in APTITUDE_KEYS:
                            # Convert the selected option's position to A/B/C/D.
                            # This allows the existing answer key to score the response.
                            option_index = options.index(selected)
                            answers[qid] = (
                                "ABCD"[option_index]
                                if option_index < 4
                                else selected
                            )
                        else:
                            answers[qid] = selected

                elif answer_type in [
                    "Rating scale (1-5)",
                    "Agreement Scale (1-5)",
                    "Skill Scale (1-5 + N/A)"
                ]:
                    choices = ["Not answered", "1", "2", "3", "4", "5"]
                    if "N/A" in answer_type:
                        choices.append("N/A")
                    selected = st.radio(
                        f"{qid}. {question_text}",
                        choices,
                        horizontal=True,
                        key=qid
                    )
                    if selected in ["Not answered", "N/A"]:
                        answers[qid] = None if selected == "Not answered" else "N/A"
                    else:
                        answers[qid] = int(selected)

                else:
                    selected = st.selectbox(
                        f"{qid}. {question_text}",
                        ["Not answered"] + options,
                        key=qid
                    )
                    answers[qid] = None if selected == "Not answered" else selected

    submitted = st.form_submit_button(
        "Calculate scores and recommend careers",
        type="primary",
        use_container_width=True
    )

if submitted:
    # Validate required demographic inputs and minimum MCQ coverage.
    answered_count = sum(v is not None for v in answers.values())
    if answered_count == 0:
        st.error("Please answer the questionnaire before submitting.")
        st.stop()

    scores = score_assessment(answers)

    # Map the complete response set into exactly the trained 25 features.
    profile = {
        "age_group": age_group,
        "education_level": education_level,
        "experience_stage": experience_stage,
        "region_type": None if region_type == "Prefer not to say" else region_type,
        **scores
    }

    input_df = pd.DataFrame([profile], columns=FEATURES)

    try:
        ranked = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        order = np.argsort(ranked)[::-1]
        top = [
            {
                "rank": rank,
                "career_domain": classes[i],
                "model_score_percent": round(float(ranked[i]) * 100, 2)
            }
            for rank, i in enumerate(order[:3], start=1)
        ]

        assessment_id = str(uuid.uuid4())
        saved_at = datetime.now(timezone.utc).isoformat()

        # Save every question response, derived features and recommendations.
        record = {
            "assessment_id": assessment_id,
            "submitted_at_utc": saved_at,
            "answered_question_count": answered_count,
            "total_original_questions": len(questions_df),
            "age_group": age_group,
            "education_level": education_level,
            "experience_stage": experience_stage,
            "region_type": region_type,
            "answers_json": json.dumps(
                {k: clean_answer(v) for k, v in answers.items()},
                ensure_ascii=False
            ),
            "feature_scores_json": json.dumps(
                {k: clean_answer(v) for k, v in scores.items()},
                ensure_ascii=False
            ),
            "recommendations_json": json.dumps(top, ensure_ascii=False)
        }
        append_result(record)

        st.success("Assessment calculated and saved successfully.")
        st.caption(f"Assessment ID: {assessment_id}")
        st.metric("Questions answered", f"{answered_count}/{len(questions_df)}")

        st.header("Your Top 3 Career Recommendations")
        columns = st.columns(3)
        for rank, item in enumerate(top):
            with columns[rank]:
                st.markdown(f"### #{item['rank']}")
                st.subheader(item["career_domain"])
                st.metric("Relative model score", f"{item['model_score_percent']:.2f}%")
                st.progress(float(np.clip(
                    item["model_score_percent"] / 100, 0, 1
                )))
                st.write(CAREER_INFO.get(item["career_domain"], ""))

        with st.expander("View calculated model features"):
            score_table = pd.DataFrame(
                [{"Feature": k, "Calculated score": clean_answer(v)}
                 for k, v in scores.items()]
            )
            st.markdown("| Feature | Calculated score |")
            st.markdown("|---|---:|")
            for _, row in score_table.iterrows():
                value = row["Calculated score"]
                display_value = "Not calculated" if pd.isna(value) else f"{float(value):.2f}"
                st.markdown(f"| {row['Feature']} | {display_value} |")

        with st.expander("View all career-domain rankings"):
            all_scores = pd.DataFrame({
                "Career domain": [classes[i] for i in order],
                "Relative model score (%)": [
                    round(float(ranked[i]) * 100, 2) for i in order
                ]
            })
            st.markdown("| Career domain | Relative model score (%) |")
            st.markdown("|---|---:|")
            for _, row in all_scores.iterrows():
                st.markdown(f"| {row['Career domain']} | {row['Relative model score (%)']:.2f} |")

        st.download_button(
            "Download this assessment result",
            data=json.dumps(record, ensure_ascii=False, indent=2),
            file_name=f"career_assessment_{assessment_id[:8]}.json",
            mime="application/json"
        )

        st.warning(
            "Scoring limitation: only explicitly mapped questions contribute "
            "to feature scores. Some features may be missing and are imputed by "
            "the saved model. The current aptitude MCQ key and score conversion "
            "are provisional and require subject-matter validation. Model scores "
            "are not validated probabilities of career success."
        )

    except Exception as exc:
        st.error(f"Assessment processing failed: {exc}")

st.divider()
st.caption(
    "Data storage: local CSV. This prototype is not a secure multi-user "
    "production system; avoid entering sensitive personal information."
)

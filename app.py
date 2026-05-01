from __future__ import annotations

import hashlib
import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

from src.audio_features import extract_audio_features
from src.question_planner import generate_adaptive_questions
from src.report_generator import generate_pdf_report
from src.scoring_engine import analyze_test
from src.speech_to_text import speech_to_text_status, transcribe_audio_file
from src.utils import INITIAL_STATEMENT_PROMPT, SAFETY_NOTE, reset_test_state, save_uploaded_file

APP_TITLE = "AI Based Voice Stress Analysis System"
FOLLOWUP_COUNT = 4
TOTAL_TEST_VOICE_SAMPLES = 1 + FOLLOWUP_COUNT

st.set_page_config(page_title=APP_TITLE, page_icon="🎙️", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1180px;
    }
    h1 {
        letter-spacing: -0.035em;
        line-height: 1.12;
        margin-bottom: 1rem;
    }
    h2, h3 {letter-spacing: -0.02em;}
    .app-kicker {
        color: var(--text-color, #f9fafb);
        opacity: 0.68;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .nav-shell {
        margin: 0.25rem 0 1.6rem 0;
        padding: 0.75rem;
        border: 1px solid rgba(128,128,128,0.24);
        border-radius: 18px;
        background: rgba(148, 163, 184, 0.06);
    }
    .nav-label {
        margin: 0 0 0.55rem 0.15rem;
        color: var(--text-color, #f9fafb);
        opacity: 0.72;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .hero-card, .info-card, .metric-card, .note-box, .question-card, .feature-card, .status-card {
        border: 1px solid rgba(128,128,128,0.28);
        border-radius: 18px;
        background: var(--secondary-background-color, #111827);
        color: var(--text-color, #f9fafb);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12);
    }
    .hero-card {
        padding: 1.05rem 1.1rem;
        margin: 0.65rem 0 1rem 0;
    }
    .hero-card p, .info-card p, .feature-card p, .status-card p {
        margin: 0 0 0.8rem 0;
        color: var(--text-color, #f9fafb);
        opacity: 0.88;
        line-height: 1.58;
    }
    .feature-card {
        padding: 1rem;
        min-height: 132px;
        margin-bottom: 0.75rem;
    }
    .feature-title, .status-title {
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 0.35rem;
        color: var(--text-color, #f9fafb);
    }
    .feature-card p, .status-card p {font-size: 0.92rem; opacity: 0.78;}
    .status-card {
        padding: 0.9rem 1rem;
        min-height: 116px;
        border-left: 4px solid rgba(148, 163, 184, 0.55);
    }
    .status-complete {border-left-color: #22c55e;}
    .status-active {border-left-color: #60a5fa;}
    .status-waiting {border-left-color: rgba(148, 163, 184, 0.55);}
    .status-badge {
        display: inline-block;
        padding: 0.18rem 0.5rem;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.14);
        color: var(--text-color, #f9fafb);
        font-size: 0.76rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .metric-card {
        min-height: 108px;
        margin-bottom: 0.85rem;
        padding: 1rem;
    }
    .metric-label {
        font-size: 0.86rem;
        color: var(--text-color, #f9fafb);
        opacity: 0.72;
        margin-bottom: 0.25rem;
        font-weight: 700;
    }
    .metric-value {
        font-size: 1.42rem;
        font-weight: 850;
        color: var(--text-color, #f9fafb);
        line-height: 1.2;
    }
    .note-box {
        border-left: 4px solid #60a5fa;
        padding: 1rem;
    }
    .small-muted {
        font-size: 0.9rem;
        color: var(--text-color, #f9fafb);
        opacity: 0.72;
    }
    .question-card {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .question-card h3 {
        margin: 0.35rem 0 0 0;
        line-height: 1.28;
    }
    .chart-intro {
        max-width: 860px;
        margin: 0.75rem auto 0.25rem auto;
    }
    div[data-testid="stPlotlyChart"] {
        max-width: 860px;
        margin-left: auto;
        margin-right: auto;
    }
    div[data-testid="stButton"] > button {
        border-radius: 12px;
        font-weight: 800;
        min-height: 2.6rem;
    }
    textarea, input {color: var(--text-color, #f9fafb) !important;}
    section[data-testid="stSidebar"] {display: none;}
    @media (max-width: 900px) {
        .main .block-container {
            padding-top: 1.25rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        h1 {font-size: 2rem !important;}
        .nav-shell {padding: 0.65rem; border-radius: 16px;}
        .hero-card, .feature-card, .status-card, .metric-card, .question-card, .note-box {
            border-radius: 15px;
            padding: 0.9rem;
        }
        div[data-testid="stPlotlyChart"] {max-width: 100%;}
    }
    @media (max-width: 520px) {
        .main .block-container {padding-left: 0.75rem; padding-right: 0.75rem;}
        h1 {font-size: 1.72rem !important;}
        .app-kicker, .nav-label {font-size: 0.75rem;}
        .metric-value {font-size: 1.22rem;}
        .hero-card p, .feature-card p, .status-card p {font-size: 0.9rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "current_page": "Home",
        "normal_audio_path": None,
        "normal_features": None,
        "initial_audio_path": None,
        "initial_features": None,
        "initial_transcript": "",
        "initial_manual_text": "",
        "generated_questions": [],
        "followup_answers": [],
        "question_index": 0,
        "questions_started": False,
        "analysis": None,
        "test_datetime": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    while len(st.session_state.followup_answers) < FOLLOWUP_COUNT:
        st.session_state.followup_answers.append(
            {
                "audio_path": None,
                "features": None,
                "transcript": "",
                "manual_text": "",
                "hash": "",
            }
        )


def set_page(page_name: str) -> None:
    st.session_state.current_page = page_name


def render_step_navigation() -> None:
    pages = ["Home", "Voice Test", "Analysis", "Final Result"]
    st.markdown("<div class='nav-shell'><div class='nav-label'>Test workflow</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for idx, (col, page) in enumerate(zip(cols, pages), start=1):
        active = st.session_state.current_page == page
        label = f"{idx}. {page}"
        if col.button(label, key=f"nav_step_{page}", type="primary" if active else "secondary", use_container_width=True):
            set_page(page)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _file_hash(file_obj: Any) -> str:
    try:
        data = bytes(file_obj.getbuffer())
        return hashlib.sha1(data).hexdigest()
    except Exception:
        return ""



def _html(value: Any) -> str:
    return html.escape(str(value or ""))


def render_feature_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class='feature-card'>
            <div class='feature-title'>{_html(title)}</div>
            <p>{_html(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_voice_test_progress_cards() -> None:
    normal_ready = bool(st.session_state.normal_features and st.session_state.normal_features.get("is_valid"))
    initial_ready = bool(st.session_state.initial_features and st.session_state.initial_features.get("is_valid") and st.session_state.initial_manual_text)
    followup_ready_count = sum(
        1
        for item in st.session_state.followup_answers[:FOLLOWUP_COUNT]
        if (item.get("features") or {}).get("is_valid") and _answer_text(item)
    )

    cards = [
        (
            "Normal Voice",
            "Add a calm sample so later answers can be compared with the usual voice pattern.",
            "Ready" if normal_ready else "Required",
            "complete" if normal_ready else "active",
        ),
        (
            "Initial Statement",
            "Describe the activity or situation. Follow-up questions are generated from this statement.",
            "Ready" if initial_ready else "Next" if normal_ready else "Locked",
            "complete" if initial_ready else "active" if normal_ready else "waiting",
        ),
        (
            "Follow-up Answers",
            "Answer the generated questions using English voice. Upload remains available as backup.",
            f"{followup_ready_count}/{FOLLOWUP_COUNT} Ready",
            "complete" if followup_ready_count == FOLLOWUP_COUNT else "active" if initial_ready else "waiting",
        ),
    ]
    cols = st.columns(3)
    for col, (title, body, badge, status) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class='status-card status-{status}'>
                    <div class='status-title'>{_html(title)}</div>
                    <p>{_html(body)}</p>
                    <span class='status-badge'>{_html(badge)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _record_audio(label: str, key: str):
    if hasattr(st, "audio_input"):
        return st.audio_input(label, key=key)
    st.caption("Microphone recording is not available in this Streamlit version. Use audio upload.")
    return None


def _process_audio_file(
    file_obj: Any,
    prefix: str,
    path_key: Optional[str] = None,
    features_key: Optional[str] = None,
    hash_key: Optional[str] = None,
    transcribe: bool = False,
) -> Dict[str, Any]:
    file_hash = _file_hash(file_obj)
    if hash_key and st.session_state.get(hash_key) == file_hash and features_key:
        return st.session_state.get(features_key) or {}

    audio_path = save_uploaded_file(file_obj, prefix=prefix)
    features = extract_audio_features(audio_path)

    if path_key:
        st.session_state[path_key] = audio_path
    if features_key:
        st.session_state[features_key] = features
    if hash_key:
        st.session_state[hash_key] = file_hash

    if transcribe and features.get("is_valid"):
        with st.spinner("Creating English transcript..."):
            transcript = transcribe_audio_file(audio_path)
        features["transcript"] = transcript or ""
    return features


def _show_audio_result(features: Dict[str, Any] | None) -> None:
    if not features:
        return
    if features.get("is_valid"):
        st.success("Audio sample is ready for analysis.")
    else:
        st.error(features.get("error_message", "Audio could not be analyzed."))


def _answer_text(answer_state: Dict[str, Any]) -> str:
    return str(answer_state.get("manual_text") or answer_state.get("transcript") or "").strip()


def _all_followup_audio_ready() -> bool:
    return all((item.get("features") or {}).get("is_valid") for item in st.session_state.followup_answers[:FOLLOWUP_COUNT])


def _all_followup_text_ready() -> bool:
    return all(_answer_text(item) for item in st.session_state.followup_answers[:FOLLOWUP_COUNT])


def _test_progress_count() -> int:
    count = 0
    if st.session_state.initial_features and st.session_state.initial_features.get("is_valid"):
        count += 1
    count += sum(1 for item in st.session_state.followup_answers[:FOLLOWUP_COUNT] if (item.get("features") or {}).get("is_valid"))
    return count


def render_home() -> None:
    st.markdown("<div class='app-kicker'>Final Year Project</div>", unsafe_allow_html=True)
    st.title("AI Based Voice Stress Analysis System")
    st.markdown(
        """
        <div class='hero-card'>
            <p>This system analyzes English voice answers and estimates lie possibility using voice stress, pauses, hesitation, and answer matching.</p>
            <p>The test starts with a normal voice sample. After that, the user gives an initial statement and the system asks follow-up questions based on that statement.</p>
            <p>The result is an estimate and should not be treated as final proof.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature_cols = st.columns(3)
    with feature_cols[0]:
        render_feature_card("Voice Pattern Check", "Compares answer voice samples with the normal voice sample.")
    with feature_cols[1]:
        render_feature_card("Adaptive Questions", "Creates follow-up questions from the first spoken statement.")
    with feature_cols[2]:
        render_feature_card("Clear Final Result", "Shows Lie Possibility as Low, Medium, or High with reasons.")

    st.write("")
    if st.button("Start Voice Test", type="primary"):
        set_page("Voice Test")
        st.rerun()


def render_normal_voice_section() -> None:
    st.subheader("Normal Voice Sample")
    st.markdown("Record your voice in a calm and natural way.")

    col_a, col_b = st.columns(2)
    with col_a:
        recorded_normal = _record_audio("Record Normal Voice", key="normal_voice_record")
    with col_b:
        uploaded_normal = st.file_uploader("Upload Normal Voice", type=["wav", "mp3", "m4a"], key="normal_voice_upload")

    normal_file = recorded_normal or uploaded_normal
    if normal_file is not None:
        features = _process_audio_file(
            normal_file,
            prefix="normal_voice",
            path_key="normal_audio_path",
            features_key="normal_features",
            hash_key="normal_audio_hash",
        )
        st.audio(normal_file)
        _show_audio_result(features)
    else:
        _show_audio_result(st.session_state.normal_features)


def render_initial_statement_section() -> None:
    st.subheader("Initial Statement")
    st.markdown("Describe one real activity or situation in clear English. The system will ask follow-up questions from this statement.")

    if not st.session_state.normal_features or not st.session_state.normal_features.get("is_valid"):
        st.info("Add a valid normal voice sample before the initial statement.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        recorded_statement = _record_audio("Record Initial Statement", key="initial_statement_record")
    with col_b:
        uploaded_statement = st.file_uploader("Upload Initial Statement", type=["wav", "mp3", "m4a"], key="initial_statement_upload")

    statement_file = recorded_statement or uploaded_statement
    if statement_file is not None:
        incoming_hash = _file_hash(statement_file)
        is_new_statement_audio = st.session_state.get("initial_audio_hash") != incoming_hash
        features = _process_audio_file(
            statement_file,
            prefix="initial_statement",
            path_key="initial_audio_path",
            features_key="initial_features",
            hash_key="initial_audio_hash",
            transcribe=True,
        )
        st.audio(statement_file)
        _show_audio_result(features)
        transcript = str(features.get("transcript") or "").strip()
        if transcript:
            st.session_state.initial_transcript = transcript
            if is_new_statement_audio or not st.session_state.get("initial_statement_text_area"):
                st.session_state.initial_manual_text = transcript
                st.session_state["initial_statement_text_area"] = transcript
        elif features.get("is_valid"):
            st.info("Transcript could not be created. Type the statement text below or record clearer English audio.")
    else:
        _show_audio_result(st.session_state.initial_features)

    statement_key = "initial_statement_text_area"
    if statement_key not in st.session_state:
        st.session_state[statement_key] = st.session_state.initial_manual_text or st.session_state.initial_transcript or ""
    typed_statement = st.text_area(
        "Statement Text",
        key=statement_key,
        height=90,
        placeholder=INITIAL_STATEMENT_PROMPT,
    )
    st.session_state.initial_manual_text = typed_statement.strip()

    can_generate = bool(st.session_state.initial_features and st.session_state.initial_features.get("is_valid") and st.session_state.initial_manual_text)
    button_label = "Generate Follow-up Questions" if not st.session_state.generated_questions else "Regenerate Follow-up Questions"
    if st.button(button_label, type="primary", disabled=not can_generate):
        st.session_state.generated_questions = generate_adaptive_questions(st.session_state.initial_manual_text, total_questions=FOLLOWUP_COUNT)
        st.session_state.questions_started = True
        st.session_state.question_index = 0
        for idx in range(FOLLOWUP_COUNT):
            st.session_state.followup_answers[idx] = {
                "audio_path": None,
                "features": None,
                "transcript": "",
                "manual_text": "",
                "hash": "",
            }
            key = f"followup_text_{idx}"
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if not can_generate:
        st.caption("Valid initial voice and statement text are required before follow-up questions.")


def render_followup_questions_section() -> None:
    if not st.session_state.questions_started or not st.session_state.generated_questions:
        return

    st.divider()
    st.subheader("Adaptive Questions")
    st.markdown("Answer each question using English voice.")

    questions: List[str] = st.session_state.generated_questions[:FOLLOWUP_COUNT]
    index = max(0, min(st.session_state.question_index, len(questions) - 1))
    st.session_state.question_index = index
    answer_state = st.session_state.followup_answers[index]

    st.markdown(
        f"""
        <div class='question-card'>
            <div class='small-muted'>Question {index + 1} of {len(questions)}</div>
            <h3>{_html(questions[index])}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(speech_to_text_status())

    col_a, col_b = st.columns(2)
    with col_a:
        recorded_answer = _record_audio("Record Answer", key=f"answer_record_{index}")
    with col_b:
        uploaded_answer = st.file_uploader("Upload Answer", type=["wav", "mp3", "m4a"], key=f"answer_upload_{index}")

    answer_file = recorded_answer or uploaded_answer
    if answer_file is not None:
        file_hash = _file_hash(answer_file)
        if answer_state.get("hash") != file_hash:
            audio_path = save_uploaded_file(answer_file, prefix=f"answer_{index + 1}")
            features = extract_audio_features(audio_path)
            transcript = ""
            if features.get("is_valid"):
                with st.spinner("Creating English transcript..."):
                    transcript = transcribe_audio_file(audio_path) or ""

            answer_state.update(
                {
                    "audio_path": audio_path,
                    "features": features,
                    "transcript": transcript,
                    "manual_text": transcript,
                    "hash": file_hash,
                }
            )
            if transcript:
                st.session_state[f"followup_text_{index}"] = transcript
        st.audio(answer_file)
        _show_audio_result(answer_state.get("features"))
        if (answer_state.get("features") or {}).get("is_valid") and not answer_state.get("transcript"):
            st.info("Transcript could not be created. Type the answer text below or record clearer English audio.")
    else:
        _show_audio_result(answer_state.get("features"))

    followup_text_key = f"followup_text_{index}"
    if followup_text_key not in st.session_state:
        st.session_state[followup_text_key] = answer_state.get("manual_text") or answer_state.get("transcript") or ""
    typed_text = st.text_area(
        "Answer Text",
        key=followup_text_key,
        height=90,
        placeholder="Transcript will appear here after speech-to-text.",
    )
    answer_state["manual_text"] = typed_text.strip()

    progress = _test_progress_count()
    st.progress(progress / TOTAL_TEST_VOICE_SAMPLES, text=f"{progress} of {TOTAL_TEST_VOICE_SAMPLES} voice samples ready")

    nav_left, nav_mid, nav_right = st.columns([1, 1, 2])
    with nav_left:
        if st.button("Previous Question", disabled=index == 0):
            st.session_state.question_index = max(0, index - 1)
            st.rerun()
    with nav_mid:
        if st.button("Next Question", disabled=index >= len(questions) - 1):
            st.session_state.question_index = min(len(questions) - 1, index + 1)
            st.rerun()
    with nav_right:
        normal_ready = bool(st.session_state.normal_features and st.session_state.normal_features.get("is_valid"))
        initial_audio_ready = bool(st.session_state.initial_features and st.session_state.initial_features.get("is_valid"))
        initial_text_ready = bool(st.session_state.initial_manual_text)
        audio_ready = normal_ready and initial_audio_ready and _all_followup_audio_ready()
        text_ready = initial_text_ready and _all_followup_text_ready()
        if not text_ready and audio_ready:
            st.warning("All answer text is required for answer matching. Check transcript fields before analysis.")
        if st.button("Analyze", type="primary", disabled=not (audio_ready and text_ready)):
            answer_features = [st.session_state.initial_features] + [item.get("features") or {} for item in st.session_state.followup_answers[:FOLLOWUP_COUNT]]
            answer_texts = [st.session_state.initial_manual_text] + [_answer_text(item) for item in st.session_state.followup_answers[:FOLLOWUP_COUNT]]
            analysis = analyze_test(st.session_state.normal_features or {}, answer_features, answer_texts)
            if analysis.get("ready"):
                analysis["question_labels"] = ["Statement"] + [f"Q{i + 1}" for i in range(FOLLOWUP_COUNT)]
                analysis["generated_questions"] = st.session_state.generated_questions
                analysis["initial_statement"] = st.session_state.initial_manual_text
                analysis["answer_texts"] = answer_texts
                analysis["question_answer_pairs"] = [
                    {"question": question, "answer": answer}
                    for question, answer in zip(st.session_state.generated_questions[:FOLLOWUP_COUNT], answer_texts[1:])
                ]
                st.session_state.analysis = analysis
                st.session_state.test_datetime = datetime.now()
                set_page("Analysis")
                st.rerun()
            else:
                st.error(str(analysis.get("error", "Analysis could not be completed.")))


def render_voice_test() -> None:
    st.markdown("<div class='app-kicker'>Voice Test</div>", unsafe_allow_html=True)
    st.title("Record and answer")
    st.markdown("Add the normal voice sample first, then give an initial statement and answer the generated follow-up questions.")
    st.caption("English-only voice test. Use a quiet place for better transcripts.")
    st.caption(speech_to_text_status())
    render_voice_test_progress_cards()
    st.write("")

    render_normal_voice_section()
    st.divider()
    render_initial_statement_section()
    render_followup_questions_section()


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis() -> None:
    st.markdown("<div class='app-kicker'>Analysis</div>", unsafe_allow_html=True)
    st.title("Analysis summary")
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Complete Voice Test first.")
        if st.button("Go to Voice Test"):
            set_page("Voice Test")
            st.rerun()
        return

    cards = analysis.get("cards", {})
    card_items = [
        ("Voice Stress", cards.get("Voice Stress", "Low")),
        ("Pause Level", cards.get("Pause Level", "Low")),
        ("Hesitation", cards.get("Hesitation", "Low")),
        ("Answer Matching", cards.get("Answer Matching", "Matched")),
        ("Speaking Speed", cards.get("Speaking Speed", "Stable")),
        ("Voice Change", cards.get("Voice Change", "Stable")),
    ]

    cols = st.columns(3)
    for idx, (label, value) in enumerate(card_items):
        with cols[idx % 3]:
            render_metric_card(label, str(value))

    st.write("")
    st.markdown(
        "<div class='chart-intro'><h3>Voice Stress Level per Answer</h3><div class='small-muted'>A compact view of stress score changes across the statement and follow-up answers.</div></div>",
        unsafe_allow_html=True,
    )
    scores = [float(value or 0) for value in analysis.get("voice_stress_scores", [])]
    label_count = max(1, len(scores))
    labels = ["Statement"] + [f"Q{i}" for i in range(1, label_count)]
    chart_scores: List[Optional[float]] = scores[:label_count]
    while len(chart_scores) < label_count:
        chart_scores.append(None)

    chart_df = pd.DataFrame(
        {
            "Answer": labels,
            "Voice Stress Level": chart_scores,
        }
    )
    fig = px.bar(
        chart_df,
        x="Answer",
        y="Voice Stress Level",
        range_y=[0, 100],
        color_discrete_sequence=["#7cc7ff"],
    )
    fig.update_layout(
        height=370,
        margin=dict(l=56, r=72, t=42, b=60),
        showlegend=False,
        bargap=0.52,
        dragmode="zoom",
        xaxis=dict(
            categoryorder="array",
            categoryarray=labels,
            tickmode="array",
            tickvals=labels,
            ticktext=labels,
            automargin=True,
        ),
        yaxis=dict(automargin=True),
        uirevision="voice_stress_chart",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9fafb", size=12),
    )
    fig.update_xaxes(
        title=None,
        fixedrange=False,
        tickangle=0,
        zeroline=False,
        linecolor="rgba(148, 163, 184, 0.30)",
        tickfont=dict(color="#f8fafc"),
        gridcolor="rgba(148, 163, 184, 0.12)",
    )
    fig.update_yaxes(
        title=dict(text="Level", font=dict(color="#f8fafc")),
        range=[0, 100],
        fixedrange=False,
        dtick=20,
        zeroline=False,
        linecolor="rgba(148, 163, 184, 0.30)",
        tickfont=dict(color="#f8fafc"),
        gridcolor="rgba(148, 163, 184, 0.24)",
    )
    fig.update_traces(
        marker_color="#7cc7ff",
        marker_line_color="#bae6fd",
        marker_line_width=1.1,
        opacity=0.95,
        hovertemplate="%{x}<br>Voice Stress Level: %{y:.1f}<extra></extra>",
    )
    chart_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
        "doubleClick": "reset",
        "responsive": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "voice_stress_level_chart",
            "height": 500,
            "width": 900,
            "scale": 2,
        },
    }
    chart_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config=chart_config,
        div_id="voice-stress-level-chart",
    )
    components.html(
        f"""
        <html>
        <head>
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    overflow: hidden;
                    color-scheme: dark;
                    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                }}
                #chart-root {{
                    width: 100%;
                    max-width: 900px;
                    margin: 0 auto;
                    position: relative;
                }}
                #voice-stress-level-chart {{
                    width: 100% !important;
                    min-height: 370px;
                }}
                .js-plotly-plot .plotly .modebar {{
                    opacity: 1 !important;
                    visibility: visible !important;
                    display: flex !important;
                    top: 4px !important;
                    right: 8px !important;
                    background: rgba(15, 23, 42, 0.84) !important;
                    border: 1px solid rgba(148, 163, 184, 0.38) !important;
                    border-radius: 10px !important;
                    padding: 3px 5px !important;
                }}
                .modebar-btn svg path {{
                    fill: #f8fafc !important;
                }}
                .modebar-btn:hover {{
                    background: rgba(96, 165, 250, 0.28) !important;
                }}
                @media (max-width: 700px) {{
                    #chart-root {{max-width: 100%;}}
                    .js-plotly-plot .plotly .modebar {{right: 2px !important; transform: scale(0.92); transform-origin: top right;}}
                }}
            </style>
        </head>
        <body>
            <div id="chart-root">{chart_html}</div>
            <script>
                const chart = document.getElementById('voice-stress-level-chart');
                function keepModebarVisible() {{
                    const modebar = document.querySelector('.modebar');
                    if (modebar) {{
                        modebar.style.opacity = '1';
                        modebar.style.visibility = 'visible';
                        modebar.style.display = 'flex';
                    }}
                }}
                window.addEventListener('load', () => {{
                    if (chart && window.Plotly) {{ window.Plotly.Plots.resize(chart); }}
                    keepModebarVisible();
                    setInterval(keepModebarVisible, 750);
                }});
                window.addEventListener('resize', () => {{
                    if (chart && window.Plotly) {{ window.Plotly.Plots.resize(chart); }}
                }});
            </script>
        </body>
        </html>
        """,
        height=430,
        scrolling=False,
    )

    if st.button("View Final Result", type="primary"):
        set_page("Final Result")
        st.rerun()


def render_final_result() -> None:
    st.markdown("<div class='app-kicker'>Final Result</div>", unsafe_allow_html=True)
    st.title("Final result")
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Complete Voice Test first.")
        if st.button("Go to Voice Test"):
            set_page("Voice Test")
            st.rerun()
        return

    score = analysis.get("final_score", 0)
    result_level = analysis.get("result_level", "Low")

    col1, col2 = st.columns(2)
    with col1:
        render_metric_card("Lie Possibility", f"{score}%")
    with col2:
        render_metric_card("Result Level", str(result_level))

    st.markdown(f"### Lie Possibility: {result_level}")
    st.subheader("Main Reasons")
    for reason in analysis.get("main_reasons", []):
        st.markdown(f"- {reason}")

    st.markdown(f"<div class='note-box'><strong>Note:</strong><br>{SAFETY_NOTE}</div>", unsafe_allow_html=True)
    st.write("")

    report_bytes = generate_pdf_report(analysis, st.session_state.test_datetime or datetime.now())
    st.download_button(
        "Download Report",
        data=report_bytes,
        file_name="lie_possibility_report.pdf",
        mime="application/pdf",
        type="primary",
    )

    if st.button("Start New Test"):
        reset_test_state(st.session_state)
        st.session_state.current_page = "Home"
        st.rerun()


def main() -> None:
    initialize_state()
    render_step_navigation()

    page = st.session_state.current_page
    if page == "Home":
        render_home()
    elif page == "Voice Test":
        render_voice_test()
    elif page == "Analysis":
        render_analysis()
    elif page == "Final Result":
        render_final_result()
    else:
        set_page("Home")
        st.rerun()


if __name__ == "__main__":
    main()

import json
import os
import random
from datetime import datetime
from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from utils_file import read_jsonl, append_jsonl, create_dir
from html import escape
from loguru import logger

for k, v in st.secrets.get("env", {}).items():
    os.environ.setdefault(k, str(v))

DATA_PATH = Path("data/human_eval_samples_qsaliency.jsonl")
OUTPUT_DIR = Path("annotations")
OUTPUT_FILE = OUTPUT_DIR / "annotations.jsonl"

# UI Colors
ANCHOR_BG = "#E8F4FF"  # light blue
QA_BG = "#E8F9E9"      # light green

# Global deterministic seed (same across annotators). Can be overridden via env var.
GLOBAL_ORDER_SEED = int(os.getenv("GLOBAL_ORDER_SEED", "20250901"))

# Shared instructions content (DRY)
INSTRUCTIONS_MD = (
    """
Please follow these steps for each task:

1. Read the learning material on the left (expand Prior sections if needed). The current section is always visible.
2. Review the three Questions in the middle (they relate to the current section).
3. Answer the Exam Problems on the right with help from the questions in the middle. You will not be graded on correctness, but please put in a reasonable effort to answer them. We are most interested in how you value the Questions. 
4. After all exam answers are filled, rate each Question on its usefulness (1-5) and interestingness (1-5) and choose a unique preference rank (1–3).
    1. Usefulness: how useful was the question for getting a deeper understanding of the content and answering the exam problems? 
        - [5 = directly useful,  = somewhat useful, indirectly useful, 1 = not useful at all]
    2. Interestingness: how interesting was the question, regardless of the exam problems? 
        - [5 = most interesting, 3 = somewhat interesting, 1 = not interesting]
    3. Ranking: provide a unique ranking (can't be tied) for each question based on its usefulness. Provide a brief explanation for your ranking. **Only need to fill in for one of the boxes**
        - [1 = most preferred, 2 = somewhat preferred, 3 = least preferred] 
5. Submit to move to the next task. A progress bar appears at the top.

You will complete one sample at a time. Progress bar will be shown at the top.
    """
)

DEFAULT_S3_BUCKET = "knic-quest"
DEFAULT_S3_PREFIX = "annotations/"
DEFAULT_AWS_REGION = "us-east-1"

# Optional: S3 healthcheck
def s3_healthcheck() -> dict:
    s3_bucket = os.getenv("S3_BUCKET", DEFAULT_S3_BUCKET)
    s3_prefix = os.getenv("S3_PREFIX", DEFAULT_S3_PREFIX)
    if not s3_bucket:
        return {"enabled": False, "ok": False, "message": "S3 not configured (S3_BUCKET missing)"}
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            region_name=os.getenv("AWS_REGION", DEFAULT_AWS_REGION),
        )
        s3.head_bucket(Bucket=s3_bucket)
        return {"enabled": True, "ok": True, "message": f"S3 bucket reachable: s3://{s3_bucket}/{s3_prefix}"}
    except Exception as e:
        return {"enabled": True, "ok": False, "message": f"S3 check failed: {e}"}


def load_samples(path: Path) -> list[dict]:
    samples = read_jsonl(path)
    return samples


def render_multiline_text(text: str | None):
    if not text:
        return
    # Preserve newlines and spaces exactly
    safe = escape(str(text))
    st.markdown(f"<div style='white-space: pre-wrap;'>{safe}</div>", unsafe_allow_html=True)


def render_multiline_colored(text: str | None, background_color: str):
    if not text:
        return
    safe = escape(str(text))
    st.markdown(
        f"""
<div style="white-space: pre-wrap; background-color: {background_color}; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">{safe}
</div>
""",
        unsafe_allow_html=True,
    )


def get_exam_questions(sample: dict) -> list[dict]:
    # relevant_questions can be a list of dicts or dict keyed by id
    rq = sample.get("relevant_questions")
    if rq is None:
        return []
    if isinstance(rq, dict):
        # assume mapping id -> {question, answer}
        items = []
        for k, v in rq.items():
            if isinstance(v, dict):
                items.append({"id": str(k), "question": v.get("question", ""), "answer": v.get("answer")})
        return items
    if isinstance(rq, list):
        # assume list of {id?, question, answer}
        items = []
        for idx, v in enumerate(rq):
            if isinstance(v, dict):
                items.append({"id": str(v.get("id", idx + 1)), "question": v.get("question", ""), "answer": v.get("answer")})
        return items
    return []


def get_generated_questions(sample: dict) -> list[dict]:
    # Collect high_utility_question, high_saliency_question, high_eig_question
    candidates = []
    mapping = {
        "utility": sample.get("high_utility_question"),
        "saliency": sample.get("high_saliency_question"),
        "eig": sample.get("high_eig_question"),
    }
    for label, q in mapping.items():
        if isinstance(q, dict) and q.get("question"):
            candidates.append({
                "label": label,
                "question": q.get("question"),
                "answer": q.get("answer"),
                "utility": q.get("utility"),
                "saliency": q.get("saliency"),
                "eig": q.get("eig"),
            })
    # Deterministic shuffle based on global seed and sample identifiers
    subject = str(sample.get("subject", ""))
    chapter = str(sample.get("chapter", ""))
    section_id = str(sample.get("section_id") or sample.get("section") or "")
    seed_str = f"{GLOBAL_ORDER_SEED}|{subject}|{chapter}|{section_id}"
    seed_int = int.from_bytes(hashlib.sha256(seed_str.encode("utf-8")).digest()[:8], byteorder="big")
    rnd = random.Random(seed_int)
    rnd.shuffle(candidates)
    return candidates


def save_annotation(record: dict) -> None:
    # If S3 is configured, upload a single-object JSON per record; else append locally
    s3_bucket = os.getenv("S3_BUCKET", DEFAULT_S3_BUCKET)
    s3_prefix = os.getenv("S3_PREFIX", DEFAULT_S3_PREFIX)
    if s3_bucket:
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                region_name=os.getenv("AWS_REGION", DEFAULT_AWS_REGION),
            )
            # Append JSONL to a single annotations object (like append_jsonl)
            key = f"{s3_prefix.rstrip('/')}/annotations.jsonl"
            line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            try:
                obj = s3.get_object(Bucket=s3_bucket, Key=key)
                existing = obj["Body"].read()
                new_body = existing + line
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                    new_body = line
                else:
                    raise
            s3.put_object(Bucket=s3_bucket, Key=key, Body=new_body, ContentType="text/plain; charset=utf-8")
            logger.info(f"Annotation saved to S3: {key}")
        except (BotoCoreError, ClientError, Exception) as e:
            # Fall back to local on any error
            logger.error(f"Error saving annotation to S3: {e}")
            # stacktrace
            import traceback
            traceback.print_exc()
            pass

    create_dir(OUTPUT_DIR)
    append_jsonl(record, OUTPUT_FILE)


def init_state(total_tasks: int):
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "completed" not in st.session_state:
        st.session_state.completed = 0
    if "order_seed" not in st.session_state:
        st.session_state.order_seed = GLOBAL_ORDER_SEED
    if "task_order" not in st.session_state:
        order = list(range(total_tasks))
        random.Random(st.session_state.order_seed).shuffle(order)
        st.session_state.task_order = order


def get_or_set_generated_questions(sample_key: str, sample: dict) -> list[dict]:
    if "gen_questions_map" not in st.session_state:
        st.session_state.gen_questions_map = {}
    if sample_key not in st.session_state.gen_questions_map:
        st.session_state.gen_questions_map[sample_key] = get_generated_questions(sample)
    return st.session_state.gen_questions_map[sample_key]


def render_instructions(total_tasks: int) -> bool:
    st.subheader("Instructions")
    st.markdown(INSTRUCTIONS_MD)
    
    # Annotator name
    st.text_input("Enter your name to begin", value=st.session_state.get("annotator_name", ""), key="annotator_name")
    st.info(f"Total tasks: {total_tasks}")
    
    st.markdown("---")
    st.subheader("Example preview")
    demo_left, demo_mid, demo_right = st.columns(3, gap="small")

    with demo_left:
        with st.expander("Prior sections (example, optional)", expanded=False):
            render_multiline_text("Consumers make choices based on preferences and budget constraints. Market demand aggregates individual demand across consumers. These concepts underpin demand curves used in microeconomics.")
        st.markdown("**Current section (example)**")
        render_multiline_colored("The law of demand states that, ceteris paribus, when the price of a good increases, the quantity demanded decreases.", ANCHOR_BG)


    with demo_mid:
        st.markdown("**Questions (example)**")
        for i in range(1, 4):
            with st.container(border=True):
                st.markdown(f"**Q{i}.**")
                render_multiline_colored("How would a price increase affect quantity demanded under the law of demand?", QA_BG)
                st.markdown("**Answer (shown by default in task)**")
                render_multiline_colored("Quantity demanded decreases when price increases, holding other factors constant.", QA_BG)


    with demo_right:
        st.markdown("**Exam Problems (example)**")
        with st.container(border=True):
            st.markdown("**Exam P1.** Define the law of demand in one sentence.")
            st.text_area("Your answer (disabled in preview)", disabled=True, key="preview_exam_q1")
        with st.container(border=True):
            st.markdown("**Exam P2.** Briefly explain a factor that can shift the demand curve.")
            st.text_area("Your answer (disabled in preview)", disabled=True, key="preview_exam_q2")
            
            st.caption("Survey (usefulness, interestingness, rank) appears after you answer all exam problems.")
    st.markdown("---")

    if st.button("Start annotation task", type="primary"):
        name = st.session_state.get("annotator_name", "").strip()
        if name:
            # Do not assign to st.session_state.annotator_name directly (widget key)
            st.session_state["_annotator"] = name
            st.session_state.started = True
            st.rerun()
        else:
            st.warning("Please enter your name before starting.")
            # Visually highlight and focus the name input
            st.markdown(
                """
<style>
input[aria-label="Enter your name to begin"] {
  border: 2px solid #ff4d4f !important;
  box-shadow: 0 0 0 1px rgba(255,77,79,0.2) !important;
}
</style>
""",
                unsafe_allow_html=True,
            )
            components.html(
                """
<script>
  const el = window.parent.document.querySelector('input[aria-label="Enter your name to begin"]');
  if (el) { el.focus(); }
</script>
""",
                height=0,
            )
    return False


def main():
    st.set_page_config(page_title="Question Quality Human Evaluation", layout="wide")
    
    # Hide GitHub icon and Streamlit UI elements
    st.markdown("""
    <style>
    #GithubIcon {
        visibility: hidden;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    header {
        visibility: hidden;
    }
    
    .stApp > div[data-testid="stToolbar"] {
        visibility: hidden;
    }
    
    .stApp > div[data-testid="stDecoration"] {
        visibility: hidden;
    }
    
    .stApp > div[data-testid="stStatusWidget"] {
        visibility: hidden;
    }
    
    /* Hide the hamburger menu */
    .stApp > div[data-testid="stSidebar"] > div[data-testid="stSidebarUserContent"] {
        visibility: hidden;
    }
    
    /* Hide footer */
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    st.title("Question Quality Human Evaluation")

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()

    samples = load_samples(DATA_PATH)
    total_tasks = len(samples)
    if total_tasks == 0:
        st.info("No samples available.")
        st.stop()

    # Show instructions before starting
    if not st.session_state.get("started"):
        # Surface S3 status for deploys
        status = s3_healthcheck()
        if status["enabled"]:
            if status["ok"]:
                logger.info(status["message"])
            else:
                logger.info(status["message"])
        else: 
            logger.info(f"S3 is not configured. Annotations will be saved locally+ {status['message']}")
        render_instructions(total_tasks)
        return

    init_state(total_tasks)

    current_pos = st.session_state.idx
    # Use randomized order to present tasks
    sample_idx = st.session_state.task_order[current_pos]
    sample = samples[sample_idx]

    # Collapsible instructions under title while task is active
    with st.expander("Instructions (click to expand)", expanded=False):
        st.markdown(INSTRUCTIONS_MD)

    # Progress
    st.progress((st.session_state.completed) / total_tasks)
    st.caption(f"Completed {st.session_state.completed} / {total_tasks}")

    # 3-column layout: context/anchor (1/3), generated (1/3), exam (1/3)
    col_context, col_generated, col_exam = st.columns(3, gap="small")

    # Prepare exam first to control survey rendering conditionally
    with col_exam:
        st.subheader("Exam Problems")
        exam_questions = get_exam_questions(sample)
        exam_answers = {}
        for i, q in enumerate(exam_questions, start=1):
            if i > 2: 
                continue 
            with st.container(border=True):
                st.markdown(f"**Exam P{i}.** {q['question']}")
                key_suffix = f"s{sample_idx}_exam_{i}"
                exam_answers[q["id"]] = st.text_area(f"Your answer to Exam P{i}", key=key_suffix)

        exam_complete = all(str(ans).strip() for ans in exam_answers.values()) if exam_answers else False

        if not exam_complete:
            st.info("Answer exam problems (right) to unlock survey.")

    with col_context:
        st.subheader("Learning Material")
        anchor = sample.get("anchor", "")
        context = sample.get("context", "")
        if anchor in context:
            context = context.replace(anchor, "")
        
        if context.strip(): 
            with st.expander("Prior sections (click to expand)", expanded=False):
                render_multiline_text(context)

        st.markdown("**Current section**")
        render_multiline_colored(sample.get("anchor", ""), ANCHOR_BG)


    with col_generated:
        st.subheader("Questions")
        sample_key = f"sample_{sample_idx}"
        gen_questions = get_or_set_generated_questions(sample_key, sample)
        if len(gen_questions) != 3:
            st.warning("This sample does not have exactly 3 generated questions.")

        survey_responses = []
        preferences = {}
        explanations = {}
        for idx, q in enumerate(gen_questions, start=1):
            with st.container(border=True):
                st.markdown(f"**Q{idx}.**")
                render_multiline_colored(q['question'], QA_BG)
                if q.get("answer"):
                    st.markdown("**Answer**")
                    render_multiline_colored(q["answer"], QA_BG) 

                key_root = f"s{sample_idx}_q{idx}"

                if exam_complete:
                    narrow_left, narrow_mid, narrow_right = st.columns([1, 2, 1])
                    with narrow_mid:
                        usefulness = st.slider("Usefulness (1-5); 5 = most useful", min_value=1, max_value=5, value=3, key=f"{key_root}_usefulness")
                        interestingness = st.slider("Interestingness (1-5; 5 = most interesting)", min_value=1, max_value=5, value=3, key=f"{key_root}_interestingness")
                        rank = st.selectbox(
                            "Preference rank (1 = most preferred)",
                            options=[None, 1, 2, 3],
                            format_func=lambda x: "Select rank" if x is None else str(x),
                            index=0,
                            key=f"{key_root}_rank",
                        )
                        
                        explanation = st.text_area("Brief explanation for your rank [only need to fill in for one of the boxes]", key=f"{key_root}_explanation")
                        
                    rank_value = int(rank) if isinstance(rank, int) else None
                    survey_responses.append({
                        "index": idx,
                        "label": q["label"],
                        "usefulness": usefulness,
                        "interestingness": interestingness,
                        "rank": rank_value,
                        "explanation": explanation,
                    })
                    preferences[idx] = rank_value
                    
                    explanations[f"{key_root}_explanation"] = explanation

    def validate_inputs():
        # Must answer all exam questions
        for _qid, ans in exam_answers.items():
            if not str(ans).strip():
                st.warning("Please answer all exam problems before submitting.")
                return False
        # All ranks must be selected and be a permutation of {1,2,3}
        if any(v is None for v in preferences.values()) or set(preferences.values()) != {1, 2, 3}:
            st.warning("Please provide a unique ranking 1, 2, 3 for the three questions.")
            return False
        
        # At least one explanation must be filled in
        if not any(str(explanation).strip() for explanation in explanations.values()):
            st.warning("Please provide a brief explanation for your rank in at least one of the text areas.")
            return False
                
        return True

    if st.button("Submit and Next", type="primary"):
        if validate_inputs():
            # Prepare payload
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "annotator": st.session_state.get("_annotator") or st.session_state.get("annotator_name", ""),
                "subject": sample.get("subject"),
                "chapter": sample.get("chapter"),
                "section_id": sample.get("section_id") or sample.get("section"),
                "task_index": int(sample_idx),
                "order_seed": st.session_state.order_seed,
                "generated_questions": gen_questions,  # include labels and metrics in stable order
                "survey": {
                    "per_question": survey_responses,
                    "preferences": preferences,
                },
                "exam": {
                    "questions": exam_questions,
                    "answers": exam_answers,
                },
            }
            save_annotation(record)

            # Move to next
            st.session_state.completed += 1
            st.session_state.idx += 1
            if st.session_state.idx >= total_tasks:
                st.success("All tasks completed. Thank you!")
                st.balloons()
            else:
                st.rerun()


if __name__ == "__main__":
    main()




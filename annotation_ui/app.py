"""
Poly-DriftBench Annotation UI — Flask Backend
===============================================
Human evaluation interface for synthetic conversation quality.

Features:
  - Annotator login (name-based, no password)
  - Per-annotator JSON files for persistence
  - Resume from last annotated conversation
  - Navigate back/forward to change annotations
  - Annotation guidelines page
  - Real-time progress tracking
"""

import json
import os
import random
import glob
import markdown as md
from markupsafe import Markup
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify, session

import re

app = Flask(__name__)
app.secret_key = "polydriftbench-annotation-2026-fixed-key"


@app.template_filter('markdown')
def markdown_filter(text):
    """Convert markdown-like text to HTML. Forgiving parser for LLM output."""
    import html as html_mod
    # Escape HTML first
    text = html_mod.escape(text)

    # Bold: **text** → <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Italic: *text* → <em>text</em>  (but not inside **)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)

    # Inline code: `text` → <code>text</code>
    text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)

    # [SYS_ACK: ACTIVE] and [Source: ...] → styled badges
    text = re.sub(
        r'\[SYS_ACK:\s*ACTIVE\]',
        '<span style="background:rgba(16,185,129,0.2);color:#10b981;padding:2px 8px;border-radius:4px;font-size:0.85em;font-weight:600">&#91;SYS_ACK: ACTIVE&#93;</span>',
        text
    )
    text = re.sub(
        r'\[Source:\s*([^\]]+)\]',
        r'<span style="background:rgba(102,126,234,0.15);color:#a5b4fc;padding:2px 8px;border-radius:4px;font-size:0.85em">&#91;Source: \1&#93;</span>',
        text
    )

    # Process lines for lists and paragraphs
    lines = text.split('\n')
    result = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        # Numbered list item: "1. ", "2. ", etc.
        list_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if list_match:
            if not in_list:
                result.append('<ol start="{}">'.format(list_match.group(1)))
                in_list = True
            result.append('<li>{}</li>'.format(list_match.group(2)))
        else:
            if in_list:
                result.append('</ol>')
                in_list = False
            if stripped:
                result.append('<p>{}</p>'.format(stripped))
            else:
                pass  # skip empty lines

    if in_list:
        result.append('</ol>')

    return Markup('\n'.join(result))

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "production"
ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
ANNOTATIONS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────
# Load & sample conversations
# ──────────────────────────────────────────────────────────

def load_sample():
    """Load the 30 sampled conversations (10 per tier)."""
    sample_file = Path(__file__).parent / "sample_manifest.json"

    if sample_file.exists():
        with open(sample_file) as f:
            return json.load(f)

    # First run: create the sample
    random.seed(42)
    samples = []
    idx = 0

    for tier in ["short", "medium", "long"]:
        files = sorted(glob.glob(str(DATA_DIR / tier / "generated" / "*.json")))
        sampled = random.sample(files, min(10, len(files)))

        for fpath in sampled:
            with open(fpath) as f:
                data = json.load(f)

            # Extract English conversation only (keep it lightweight)
            en_conv = data["conversations"]["en"]
            samples.append({
                "index": idx,
                "id": data["id"],
                "tier": tier,
                "domain": data["domain"],
                "num_turns": data["num_turns"],
                "num_messages": len(en_conv),
                "messages": en_conv,
                "plan": data.get("plan", {}),
                "quality_auto": data.get("quality", {}),
            })
            idx += 1

    with open(sample_file, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    return samples


CONVERSATIONS = load_sample()


# ──────────────────────────────────────────────────────────
# Annotator file management
# ──────────────────────────────────────────────────────────

def get_annotator_file(name: str) -> Path:
    safe_name = "".join(c for c in name.lower().replace(" ", "_") if c.isalnum() or c == "_")
    return ANNOTATIONS_DIR / f"{safe_name}.json"


def load_annotations(name: str) -> dict:
    fpath = get_annotator_file(name)
    if fpath.exists():
        with open(fpath) as f:
            return json.load(f)
    return {
        "annotator": name,
        "annotations": {},
        "last_index": 0,
    }


def save_annotations(name: str, data: dict):
    fpath = get_annotator_file(name)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "annotator" in session:
        return redirect(url_for("annotate"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if name:
            session["annotator"] = name
            return redirect(url_for("annotate"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("annotator", None)
    return redirect(url_for("login"))


@app.route("/guidelines")
def guidelines():
    return render_template("guidelines.html")


@app.route("/annotate")
@app.route("/annotate/<int:conv_idx>")
def annotate(conv_idx=None):
    if "annotator" not in session:
        return redirect(url_for("login"))

    name = session["annotator"]
    ann_data = load_annotations(name)

    # Determine which conversation to show
    if conv_idx is None:
        conv_idx = ann_data.get("last_index", 0)

    conv_idx = max(0, min(conv_idx, len(CONVERSATIONS) - 1))
    conv = CONVERSATIONS[conv_idx]

    # Get existing annotation for this conversation
    existing = ann_data["annotations"].get(str(conv_idx), {})

    # Progress stats
    total = len(CONVERSATIONS)
    done = len(ann_data["annotations"])

    return render_template(
        "annotate.html",
        conv=conv,
        conv_idx=conv_idx,
        total=total,
        done=done,
        existing=existing,
        annotator=name,
    )


@app.route("/api/save", methods=["POST"])
def save():
    if "annotator" not in session:
        return jsonify({"error": "Not logged in"}), 401

    name = session["annotator"]
    data = request.json

    ann_data = load_annotations(name)
    conv_idx = str(data["conv_idx"])

    ann_data["annotations"][conv_idx] = {
        "naturalness": data.get("naturalness"),
        "user_realism": data.get("user_realism"),
        "coherence": data.get("coherence"),
        "ddm_compliance": data.get("ddm_compliance"),
        "overall_quality": data.get("overall_quality"),
        "notes": data.get("notes", ""),
        "conversation_id": data.get("conversation_id", ""),
    }
    ann_data["last_index"] = int(conv_idx)
    save_annotations(name, ann_data)

    return jsonify({"status": "saved", "done": len(ann_data["annotations"])})


@app.route("/api/progress")
def progress():
    if "annotator" not in session:
        return jsonify({"error": "Not logged in"}), 401

    name = session["annotator"]
    ann_data = load_annotations(name)

    return jsonify({
        "total": len(CONVERSATIONS),
        "done": len(ann_data["annotations"]),
        "annotated_indices": list(ann_data["annotations"].keys()),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)

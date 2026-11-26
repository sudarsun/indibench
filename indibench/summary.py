"""One-page executive summary for business readers.

Turns ``results.json`` into a print-ready A4 HTML page (optionally PDF / Markdown):

* capability areas (Tamil, other Indian languages, Indian context, English, math & science,
  programming, vision, safety) with a 0-100 **skill score** and a plain-language readiness verdict
* headline KPIs, strengths, watch-outs and concrete recommendations
* optional comparison against a baseline run (deltas per area)

Skill score: multiple-choice accuracy is rescaled so that random guessing = 0 and perfect = 100
(``(acc - chance) / (100 - chance)``); error rates are inverted (100 - CER); other metrics
(chrF++, pass@1, ...) are used as-is. Area score = mean of its tasks' skill scores.
Everything is rule-based and reproducible; an LLM-written narrative is opt-in.
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import math
import os
import statistics
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------- areas
AREAS: "OrderedDict[str, Dict[str, str]]" = OrderedDict([
    ("tamil", {"label": "Tamil understanding",
               "what": "reading, knowledge, classification",
               "use": "Tamil customer queries, document understanding, search"}),
    ("tamil_gen", {"label": "Tamil writing & translation",
                   "what": "translation, summaries, Tamil answers",
                   "use": "customer-facing Tamil content, translation workflows"}),
    ("indic", {"label": "Other Indian languages",
               "what": "Hindi, Telugu, Kannada, Malayalam …",
               "use": "pan-India multilingual deployment"}),
    ("india", {"label": "Indian context (English)",
               "what": "Indian knowledge, culture, exams",
               "use": "Indian-market assistants, education, healthcare & exam prep"}),
    ("english", {"label": "English knowledge & reasoning",
                 "what": "knowledge, commonsense, reasoning",
                 "use": "general-purpose assistant work"}),
    ("stem", {"label": "Math & science",
              "what": "maths, science, medicine",
              "use": "analytics, tutoring, technical support"}),
    ("code", {"label": "Programming",
              "what": "Python that passes unit tests",
              "use": "developer productivity, automation"}),
    ("vision", {"label": "Vision & documents",
                "what": "charts, documents, OCR, images",
                "use": "document processing, forms, visual QA"}),
    ("safety", {"label": "Safety & reliability",
                "what": "truthfulness, hallucination, instructions",
                "use": "any customer-facing deployment"}),
])

_INDIC = {"hi", "te", "kn", "ml", "bn", "mr", "gu", "pa", "or", "as", "ur", "ne", "sa", "sat", "mni"}
_SAFETY_TAGS = {"safety", "truthfulness", "hallucination", "instruction_following", "moderation"}
_STEM_TAGS = {"math", "science", "medical"}
_INDIA_TAGS = {"indian_english", "culture", "exams"}


def area_of(res: Dict[str, Any]) -> str:
    tags = set(res.get("tags") or [])
    lang = res.get("lang")
    if "code" in tags or res.get("type") == "code":
        return "code"
    if res.get("modality") == "image":
        return "vision"
    if tags & _SAFETY_TAGS:
        return "safety"
    if lang == "ta":
        return "tamil_gen" if res.get("type") == "generation" else "tamil"
    if lang in _INDIC:
        return "indic"
    if tags & _STEM_TAGS:
        return "stem"
    if tags & _INDIA_TAGS:
        return "india"
    return "english"


def skill_score(res: Dict[str, Any]) -> Optional[float]:
    pm = res.get("primary_metric")
    m = (res.get("metrics") or {}).get(pm or "")
    if not m or m.get("value") is None or (isinstance(m["value"], float) and math.isnan(m["value"])):
        return None
    v = float(m["value"])
    if m.get("higher_is_better") is False:
        v = 100.0 - v
    chance = res.get("chance")
    if chance is not None and 0 <= chance < 100:
        v = (v - chance) / (100.0 - chance) * 100.0
    return max(0.0, min(100.0, v))


BANDS = [  # (min score, label, css class, verdict)
    (70, "Strong", "good", "Ready for production use"),
    (50, "Good", "ok", "Usable with human review"),
    (30, "Limited", "warn", "Pilot only / needs tuning"),
    (0, "Weak", "bad", "Not recommended yet"),
]


def band(score: Optional[float]) -> Tuple[str, str, str]:
    if score is None:
        return "Not tested", "na", "Not evaluated"
    for lo, label, css, verdict in BANDS:
        if score >= lo:
            return label, css, verdict
    return BANDS[-1][1:]  # pragma: no cover


FRIENDLY = {
    "belebele": "Reading comprehension (Belebele)", "milu": "India-centric knowledge (MILU)",
    "include": "Regional exam questions (INCLUDE)", "indicsentiment": "Review sentiment",
    "indicxnli": "Logical inference (XNLI)", "indicqa": "Question answering (IndicQA)",
    "xquad_in": "Question answering (XQuAD-IN)", "flores_en2xx": "Translation from English (FLORES)",
    "flores_xx2en": "Translation to English (FLORES)", "flores_in_en2xx": "Translation from English (Flores-IN)",
    "in22conv_en2xx": "Conversational translation (IN22)", "xlsum": "News summarisation (XL-Sum)",
    "crosssum_in": "Cross-lingual summarisation", "synth_ocr": "Reading text in images (OCR)",
    "synth_ocr_flores": "Reading text in images (OCR)", "pope_xl": "Image grounding, native-language answer",
    "tanglish_sentiment": "Tanglish (code-mixed) sentiment", "offensive_codemix": "Offensive-content detection",
    "mmlu_indic": "General knowledge (MMLU-Indic)", "indicmmlu_pro": "Hard knowledge (IndicMMLU-Pro)",
    "arc_indic": "Science reasoning (ARC-Indic)", "triviaqa_indic_mcq": "World knowledge (TriviaQA)",
    "boolq_indic": "Yes/no reading (BoolQ)", "global_mmlu": "General knowledge (Global-MMLU)",
    "drishtikon": "Indian culture from images", "medmcqa": "Medical entrance exams (NEET-PG)",
    "jeebench": "IIT-JEE Advanced problems", "mmlu_pro": "Hard general knowledge (MMLU-Pro)",
    "mmlu_en": "General knowledge (MMLU)", "gpqa_diamond": "Graduate-level science (GPQA)",
    "arc_challenge": "Science reasoning (ARC)", "hellaswag": "Commonsense (HellaSwag)",
    "winogrande": "Commonsense (WinoGrande)", "truthfulqa_mc1": "Truthfulness (TruthfulQA)",
    "gsm8k": "Math word problems (GSM8K)", "math500": "Competition math (MATH-500)", "aime24": "Olympiad math (AIME)",
    "mgsm": "Math word problems (MGSM)", "humaneval": "Python coding (HumanEval)", "mbpp": "Python coding (MBPP)",
    "ifeval": "Following formatting instructions", "mmmu_val": "College-level image questions (MMMU)",
    "mathvista_testmini": "Visual math (MathVista)", "chartqa": "Reading charts", "docvqa_val": "Reading documents",
    "textvqa_val": "Reading text in photos", "ocrbench": "OCR (OCRBench)", "ai2d": "Science diagrams",
    "scienceqa_img": "School science with images", "mmbench_en_dev": "General visual understanding",
    "pope": "Object hallucination (POPE)", "mme": "Perception (MME)", "cvqa": "Culturally-grounded VQA",
}
_LANG_NAME = {"ta": "Tamil", "hi": "Hindi", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali",
              "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ur": "Urdu",
              "ne": "Nepali", "en": "English"}


def _friendly(name: str, res: Dict[str, Any]) -> str:
    fam = res.get("family") or name
    lang = res.get("lang")
    if fam in FRIENDLY:
        base = FRIENDLY[fam]
        return f"{base} — {_LANG_NAME.get(lang, lang)}" if lang else base
    desc = (res.get("description") or "").strip()
    head = desc.split(" — ")[0].split(" - ")[0].split("(")[0].strip() if desc else ""
    head = head if 2 < len(head) <= 40 else (res.get("family") or name)
    lang = res.get("lang")
    return f"{head} ({lang})" if lang and lang not in head else head


# --------------------------------------------------------------------------------- analysis
def analyse(payload: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    results = payload.get("results", {})
    tasks = []
    for name, res in results.items():
        if "error" in res or "skipped" in res or res.get("predict_only"):
            continue
        sc = skill_score(res)
        if sc is None:
            continue
        pm = res.get("primary_metric")
        m = res["metrics"][pm]
        ci = (m.get("ci_high") - m.get("ci_low")) / 2 if m.get("ci_low") is not None else None
        tasks.append({"name": name, "label": _friendly(name, res), "area": area_of(res), "score": sc,
                      "raw": m["value"], "metric": pm, "n": res.get("n", 0), "ci": ci, "lang": res.get("lang"),
                      "family": res.get("family")})
    areas: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for key, meta in AREAS.items():
        ts = [t for t in tasks if t["area"] == key]
        score = statistics.fmean(t["score"] for t in ts) if ts else None
        label, css, verdict = band(score)
        areas[key] = dict(meta, key=key, score=score, band=label, css=css, verdict=verdict, tasks=ts,
                          n=sum(t["n"] for t in ts))

    base_areas = analyse(baseline)["areas"] if baseline else None
    if base_areas:
        for k, a in areas.items():
            b = base_areas.get(k, {}).get("score")
            a["delta"] = (a["score"] - b) if a["score"] is not None and b is not None else None

    measured = [a for a in areas.values() if a["score"] is not None]
    overall = statistics.fmean(a["score"] for a in measured) if measured else None

    # Tamil vs English parity on benchmarks run in both languages (same questions or same design)
    by_fam: Dict[str, Dict[str, float]] = {}
    for t in tasks:
        if t["lang"] in ("ta", "en") and t["family"]:
            by_fam.setdefault(t["family"], {})[t["lang"]] = t["score"]
    pairs = [(v["ta"], v["en"]) for v in by_fam.values() if "ta" in v and "en" in v and v["en"] > 0]
    parity = 100.0 * statistics.fmean(ta / en for ta, en in pairs) if pairs else None

    fid = [r["metrics"]["script_fidelity"]["value"] for r in results.values()
           if r.get("lang") == "ta" and "script_fidelity" in (r.get("metrics") or {})]
    fidelity = statistics.fmean(fid) if fid else None

    ns = [t["n"] for t in tasks]
    cis = [t["ci"] for t in tasks if t["ci"] is not None]
    errors = [k for k, v in results.items() if "error" in v]
    skipped = [k for k, v in results.items() if "skipped" in v]
    return {"tasks": tasks, "areas": areas, "overall": overall, "parity": parity, "parity_pairs": len(pairs),
            "fidelity": fidelity, "n_tasks": len(tasks), "n_samples": sum(ns),
            "median_n": statistics.median(ns) if ns else 0, "typical_ci": statistics.median(cis) if cis else None,
            "errors": errors, "skipped": skipped, "has_baseline": bool(base_areas),
            "baseline_name": (baseline or {}).get("model", {}).get("name")}


_ADVICE = {
    "tamil": "keep a human check on Tamil customer interactions and add Tamil examples to prompts or fine-tuning",
    "tamil_gen": "route customer-facing Tamil text and translations through human review",
    "indic": "pilot one Indian language at a time; do not assume Tamil results carry over",
    "india": "ground answers about Indian policy, culture or exams in retrieved documents (RAG)",
    "english": "prefer a larger model for open-ended English reasoning tasks",
    "stem": "verify calculations with a tool/calculator step; do not let the model work unsupervised on numbers",
    "code": "use for suggestions only, with tests and code review before merging",
    "vision": "keep humans on document/OCR extraction and validate key fields",
    "safety": "add guardrails (output filters, refusal policy) and monitor production traffic",
}


def recommendations(a: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    weak = sorted([x for x in a["areas"].values() if x["score"] is not None and x["score"] < 50],
                  key=lambda x: x["score"])
    for x in weak[:3]:
        recs.append(f"<b>{x['label']}</b> is {x['band'].lower()} ({x['score']:.0f}/100): {_ADVICE[x['key']]}.")
    if a["fidelity"] is not None and a["fidelity"] < 90:
        recs.append(f"When asked for Tamil, only <b>{a['fidelity']:.0f}%</b> of the answer text is in Tamil script — "
                    "state the output language explicitly in the system prompt and check outputs.")
    if a["parity"] is not None and a["parity"] < 80:
        recs.append(f"Tamil performance is <b>{a['parity']:.0f}%</b> of English on the same tests — budget for Tamil "
                    "fine-tuning or a stronger Indic model before a Tamil-first launch.")
    missing = [x["label"] for x in a["areas"].values() if x["score"] is None]
    if missing:
        recs.append("Not evaluated in this run: " + ", ".join(missing) + ". Add these suites before a go/no-go "
                    "decision if they matter for your use case.")
    if a["median_n"] and a["median_n"] < 200:
        ci = f" (typical margin ±{a['typical_ci']:.0f} pts)" if a["typical_ci"] else ""
        recs.append(f"Scores use a median of {a['median_n']:.0f} questions per test{ci}; small gaps between "
                    "models are not meaningful. Re-run with <code>--limit all</code> for final decisions.")
    strong = [x for x in a["areas"].values() if x["score"] is not None and x["score"] >= 70]
    if strong and len(recs) < 5:
        recs.append("Good fit today for: " + "; ".join(f"{x['use']}" for x in strong[:2]) + ".")
    return recs[:5]


def narrative(a: Dict[str, Any], model: str) -> str:
    if a["overall"] is None:
        return "No scored results were found in this run."
    ranked = sorted([x for x in a["areas"].values() if x["score"] is not None], key=lambda x: -x["score"])
    best = " and ".join(x["label"] for x in ranked[:2])
    worst = ranked[-1]["label"] if len(ranked) > 2 else None
    s = (f"<b>{html.escape(model)}</b> scores <b>{a['overall']:.0f}/100</b> overall "
         f"({band(a['overall'])[0].lower()}) across {a['n_tasks']} tests and {a['n_samples']:,} questions. "
         f"It is strongest at {best}")
    s += f" and weakest at {worst}." if worst else "."
    if a["parity"] is not None:
        s += f" In Tamil it keeps <b>{a['parity']:.0f}%</b> of its English performance."
    return s


def llm_narrative(a: Dict[str, Any], model_name: str, api_base: str, api_model: str, api_key_env: str) -> Optional[str]:
    """Optional: ask an OpenAI-compatible model for a 3-sentence summary of the computed facts only."""
    from indibench.models.openai_api import chat_completion

    facts = {"model": model_name, "overall": a["overall"], "tamil_vs_english_pct": a["parity"],
             "tamil_script_fidelity_pct": a["fidelity"],
             "areas": {x["label"]: (round(x["score"], 1) if x["score"] is not None else None) for x in a["areas"].values()}}
    prompt = ("Write a 3-sentence executive summary for business stakeholders about an AI model evaluation. "
              "Use ONLY these facts, do not invent numbers, avoid jargon, mention one strength, one risk and a "
              "recommendation.\nFacts (scores 0-100, chance-corrected):\n" + json.dumps(facts))
    resp = chat_completion(api_base, os.environ.get(api_key_env, "EMPTY"),
                           {"model": api_model, "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 220, "temperature": 0.2}, 120, 2)
    try:
        return html.escape(resp["choices"][0]["message"]["content"].strip())
    except (KeyError, IndexError):
        return None


# --------------------------------------------------------------------------------- rendering
_CSS = """
@page { size: A4; margin: 12mm 13mm; }
* { box-sizing: border-box; }
body { font-family: "Inter", "Segoe UI", "Noto Sans", "Noto Sans Tamil", Arial, sans-serif; color: #1d2330;
       font-size: 9.6pt; line-height: 1.33; margin: 0; }
.page { max-width: 186mm; margin: 0 auto; }
header { display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 2px solid #1d2330;
         padding-bottom: 6px; margin-bottom: 10px; }
h1 { font-size: 16pt; margin: 0; letter-spacing: -0.01em; }
.sub { color: #5b6475; font-size: 9pt; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0 10px; }
.kpi { border: 1px solid #dfe3ea; border-radius: 6px; padding: 7px 9px; }
.kpi .v { font-size: 18pt; font-weight: 700; line-height: 1.1; }
.kpi .l { color: #5b6475; font-size: 8.4pt; text-transform: uppercase; letter-spacing: .04em; }
.kpi .s { font-size: 8.6pt; color: #5b6475; }
.lead { margin: 0 0 9px; font-size: 10.4pt; }
table { width: 100%; border-collapse: collapse; margin-bottom: 8px; table-layout: fixed; }
th { text-align: left; font-size: 8.3pt; text-transform: uppercase; letter-spacing: .04em; color: #5b6475;
     border-bottom: 1px solid #c9ced8; padding: 3px 5px; }
td { border-bottom: 1px solid #eef0f4; padding: 3px 5px; vertical-align: middle; }
td.area { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
td.area b { font-weight: 650; } td.area span { color: #6b7385; font-size: 8pt; margin-left: 6px; }
.bar { background: #eef0f4; border-radius: 3px; height: 9px; width: 100%; min-width: 70px; }
.bar i { display: block; height: 9px; border-radius: 3px; }
.num { font-weight: 700; text-align: right; white-space: nowrap; }
.badge { display: inline-block; border-radius: 10px; padding: 1px 8px; font-size: 8.4pt; font-weight: 600; white-space: nowrap; }
.good { background: #dff3e6; color: #17663a; } .bar i.good { background: #2f9e5b; }
.ok   { background: #e3efff; color: #1f4f99; } .bar i.ok   { background: #3b7be0; }
.warn { background: #fff1d6; color: #8a5a00; } .bar i.warn { background: #e0a020; }
.bad  { background: #fde2e1; color: #9b1c1c; } .bar i.bad  { background: #d64545; }
.na   { background: #f1f2f5; color: #7b8293; }
.delta-up { color: #17663a; font-weight: 600; } .delta-down { color: #9b1c1c; font-weight: 600; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
h2 { font-size: 10.5pt; margin: 4px 0 4px; text-transform: uppercase; letter-spacing: .05em; color: #39404f; }
ul { margin: 0 0 8px 16px; padding: 0; } li { margin-bottom: 3px; }
footer { border-top: 1px solid #dfe3ea; margin-top: 6px; padding-top: 5px; color: #6b7385; font-size: 7.8pt; }
code { font-size: 8.6pt; background: #f1f2f5; padding: 0 3px; border-radius: 3px; }
@media print { .page { max-width: none; } }
"""


def _fmt(v: Optional[float], d: int = 0) -> str:
    return "–" if v is None else f"{v:.{d}f}"


def render_html(payload: Dict[str, Any], a: Dict[str, Any], title: Optional[str] = None,
                story: Optional[str] = None) -> str:
    model = payload.get("model", {}).get("name", "model")
    env = payload.get("env", {})
    gpu = (env.get("gpu") or {}).get("name", "")
    date = (env.get("time") or _dt.datetime.now().isoformat())[:10]
    title = title or "AI model evaluation — executive summary"
    ob, oc, ov = band(a["overall"])
    tamil = a["areas"]["tamil"]["score"]
    tamil_gen = a["areas"]["tamil_gen"]["score"]
    tamil_all = [x for x in (tamil, tamil_gen) if x is not None]
    tamil_score = statistics.fmean(tamil_all) if tamil_all else None
    tb, tc, _ = band(tamil_score)

    rows = []
    for x in a["areas"].values():
        w = 0 if x["score"] is None else max(2, x["score"])
        delta = ""
        if a["has_baseline"]:
            d = x.get("delta")
            delta = "<td class='num'>–</td>" if d is None else \
                f"<td class='num {'delta-up' if d >= 0 else 'delta-down'}'>{'+' if d >= 0 else ''}{d:.0f}</td>"
        rows.append(
            f"<tr><td class='area'><b>{x['label']}</b><span>{x['what']}</span></td>"
            f"<td><div class='bar'><i class='{x['css']}' style='width:{w:.0f}%'></i></div></td>"
            f"<td class='num'>{_fmt(x['score'])}</td>"
            f"<td><span class='badge {x['css']}'>{x['band']}</span></td>"
            f"<td>{x['verdict']}</td>{delta}</tr>")
    delta_head = "<th style='width:7%;text-align:right'>Δ</th>" if a["has_baseline"] else ""

    ranked = sorted(a["tasks"], key=lambda t: -t["score"])
    strengths = [f"{html.escape(t['label'])} — <b>{t['score']:.0f}</b>" for t in ranked[:4] if t["score"] >= 50]
    watch = [f"{html.escape(t['label'])} — <b>{t['score']:.0f}</b>" for t in reversed(ranked[-4:]) if t["score"] < 60]
    if a["errors"]:
        watch.append(f"{len(a['errors'])} test(s) could not run (see report.md)")
    recs = recommendations(a)

    kpi_parity = (f"<div class='kpi'><div class='l'>Tamil vs English</div><div class='v'>{_fmt(a['parity'])}%</div>"
                  f"<div class='s'>of English performance on {a['parity_pairs']} matched test(s)</div></div>"
                  if a["parity"] is not None else
                  f"<div class='kpi'><div class='l'>Tamil script fidelity</div><div class='v'>{_fmt(a['fidelity'])}"
                  f"{'%' if a['fidelity'] is not None else ''}</div><div class='s'>of answer text written in Tamil"
                  f"</div></div>")
    base_note = (f" Δ column: change versus <b>{html.escape(a['baseline_name'] or 'baseline')}</b> "
                 "(positive = this model is better)." if a["has_baseline"] else "")
    gen = payload.get("config", {}).get("run", {})
    limit = gen.get("limit") or "all"
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)} — {html.escape(model)}</title><style>{_CSS}</style></head><body><div class="page">
<header><div><h1>{html.escape(title)}</h1>
<div class="sub">Model: <b>{html.escape(model)}</b> · {date}{(' · ' + html.escape(gpu)) if gpu else ''}</div></div>
<div class="sub" style="text-align:right">indibench · {a['n_tasks']} tests · {a['n_samples']:,} questions</div></header>
<div class="kpis">
 <div class="kpi"><div class="l">Overall score</div><div class="v">{_fmt(a['overall'])}<span style="font-size:10pt">/100</span></div>
  <div class="s"><span class="badge {oc}">{ob}</span> {ov}</div></div>
 <div class="kpi"><div class="l">Tamil</div><div class="v">{_fmt(tamil_score)}<span style="font-size:10pt">/100</span></div>
  <div class="s"><span class="badge {tc}">{tb}</span> understanding + writing</div></div>
 {kpi_parity}
 <div class="kpi"><div class="l">Evidence</div><div class="v">{a['n_tasks']}</div>
  <div class="s">tests · median {a['median_n']:.0f} questions each{(' · ±' + _fmt(a['typical_ci']) + ' pts') if a['typical_ci'] else ''}</div></div>
</div>
<p class="lead">{story or narrative(a, model)}{base_note}</p>
<table><thead><tr><th style="width:40%">Capability</th><th style="width:15%"></th><th style="width:7%;text-align:right">Score</th>
<th style="width:11%">Rating</th><th style="width:19%">Readiness</th>{delta_head}</tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="cols">
 <div><h2>Strengths</h2><ul>{''.join(f'<li>{s}</li>' for s in strengths) or '<li>No test scored 50+ yet.</li>'}</ul>
      <h2>Watch-outs</h2><ul>{''.join(f'<li>{w}</li>' for w in watch) or '<li>No test scored below 60.</li>'}</ul></div>
 <div><h2>Recommendations</h2><ul>{''.join(f'<li>{r}</li>' for r in recs)}</ul></div>
</div>
<footer><b>How to read this:</b> scores are 0–100 "skill scores" — multiple-choice results are rescaled so random
guessing = 0 and perfect = 100; other measures (translation quality chrF++, code pass rate, OCR accuracy) are used
directly. Ratings: Strong ≥ 70 · Good 50–69 · Limited 30–49 · Weak &lt; 30. Based on {limit} questions per test
from public benchmarks (Δ: difference in points); real-world performance depends on your prompts, data and guardrails.
Full details: report.md / results.json in the run folder. Generated {_dt.datetime.now():%Y-%m-%d %H:%M} by indibench.</footer>
</div></body></html>"""


def render_markdown(payload: Dict[str, Any], a: Dict[str, Any]) -> str:
    model = payload.get("model", {}).get("name", "model")
    lines = [f"# Executive summary — {model}", "", narrative(a, model).replace("<b>", "**").replace("</b>", "**"), "",
             "| Capability | Score | Rating | Readiness |" + (" Δ |" if a["has_baseline"] else ""),
             "|---|---:|---|---|" + ("---:|" if a["has_baseline"] else "")]
    for x in a["areas"].values():
        d = ""
        if a["has_baseline"]:
            dv = x.get("delta")
            d = f" {'' if dv is None else ('+' if dv >= 0 else '') + f'{dv:.0f}'} |"
        lines.append(f"| {x['label']} | {_fmt(x['score'])} | {x['band']} | {x['verdict']} |{d}")
    lines += ["", "**Recommendations**", ""]
    lines += ["- " + r.replace("<b>", "**").replace("</b>", "**").replace("<code>", "`").replace("</code>", "`")
              for r in recommendations(a)]
    return "\n".join(lines) + "\n"


def html_to_pdf(html_path: str, pdf_path: str) -> Optional[str]:
    """Best effort: WeasyPrint, then headless Chrome/Chromium. Returns the PDF path or None."""
    try:
        from weasyprint import HTML  # type: ignore

        HTML(filename=html_path).write_pdf(pdf_path)
        return pdf_path
    except Exception:
        pass
    import shutil
    import subprocess

    for exe in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        if shutil.which(exe):
            try:
                subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                                f"--print-to-pdf={pdf_path}", "file://" + os.path.abspath(html_path)],
                               check=True, capture_output=True, timeout=120)
                return pdf_path
            except Exception:
                continue
    return None


def load_payload(path: str) -> Dict[str, Any]:
    f = os.path.join(path, "results.json") if os.path.isdir(path) else path
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


def write_executive_summary(payload: Dict[str, Any], out_dir: str, baseline: Optional[Dict[str, Any]] = None,
                            title: Optional[str] = None, fmt: str = "html", pdf: bool = False,
                            story: Optional[str] = None, filename: str = "executive_summary") -> str:
    a = analyse(payload, baseline)
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "md":
        path = os.path.join(out_dir, filename + ".md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(payload, a))
        return path
    path = os.path.join(out_dir, filename + ".html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(payload, a, title, story))
    if pdf:
        out = html_to_pdf(path, os.path.join(out_dir, filename + ".pdf"))
        return out or path
    return path

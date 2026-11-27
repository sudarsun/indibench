"""Reasoning-heavy task types: free-form math, code execution (HumanEval/MBPP) and JEEBench.

* ``math``      final-answer extraction (``\\boxed{}``, "The answer is", ``####``, last number) and
                equivalence checking (normalised LaTeX, numeric tolerance, optional SymPy).
* ``code``      runs generated Python against the benchmark's unit tests in a subprocess with a
                timeout and memory limits. **Opt-in** (``run.allow_code_execution``) because it executes
                model-written code; run it inside a container/VM.
* ``jeebench``  IIT-JEE Advanced physics/chemistry/maths with the paper's scoring: exact for
                single-correct/integer, ±0.01 for numeric, partial credit for multi-correct.
"""
from __future__ import annotations

import os
import math
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from indibench.metrics.extract import extract_number
from indibench.tasks.base import Doc, Task, _as_list, _get, register_type

# ============================================================================ math helpers
def last_boxed(text: str) -> Optional[str]:
    """Content of the last ``\\boxed{...}`` / ``\\fbox{...}`` (balanced braces)."""
    if not text:
        return None
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        m = re.match(r"\\boxed\s+(\S+)", text[idx:])   # "\boxed 5"
        return m.group(1) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


_ANSWER_PATTERNS = [
    r"final answer\s*(?:is|:)?\s*[:\-]?\s*(.+)",
    r"the answer is\s*[:\-]?\s*(.+)",
    r"answer\s*:\s*(.+)",
    r"####\s*(.+)",
]


def extract_math_answer(text: str) -> Optional[str]:
    if not text:
        return None
    b = last_boxed(text)
    if b is not None:
        return b.strip()
    for pat in _ANSWER_PATTERNS:
        hits = re.findall(pat, text, flags=re.IGNORECASE)
        if hits:
            ans = hits[-1].strip().split("\n")[0]
            return ans.strip().rstrip(".").strip("$ ").strip()
    n = extract_number(text)
    if n is None:
        return None
    return str(int(n)) if float(n).is_integer() else str(n)


def _strip_text_cmd(s: str) -> str:
    return re.sub(r"\\(?:text|textbf|mathrm|mbox|operatorname)\{([^{}]*)\}", r"\1", s)


def normalize_math(s: Any) -> str:
    """Hendrycks-MATH style normalisation, slightly extended."""
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\n", "").replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\displaystyle", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = _strip_text_cmd(s)
    s = re.sub(r"\\(?:quad|qquad)", "", s)
    s = s.strip().rstrip(".")
    # "x = 5" -> "5"  (single variable assignment only)
    m = re.fullmatch(r"\s*[a-zA-Z]\s*=\s*(.+)", s)
    if m:
        s = m.group(1)
    # \frac12 -> \frac{1}{2}, \sqrt2 -> \sqrt{2}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = s.replace(" ", "")
    if s.startswith("0.") is False and s.startswith("."):
        s = "0" + s
    # 5.0 -> 5 ; 1,000 -> 1000
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    return s


def _to_number(s: str) -> Optional[float]:
    s = normalize_math(s)
    m = re.fullmatch(r"(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    try:
        if m:
            v = float(m.group(2)) / float(m.group(3))
            return -v if m.group(1) else v
        m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", s)
        if m:
            return float(m.group(1)) / float(m.group(2))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def _sympy_equal(a: str, b: str) -> bool:
    try:
        import sympy
        from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                                standard_transformations)
    except ImportError:
        return False

    def conv(x: str) -> str:
        x = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", x)
        x = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", x)
        x = x.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
        x = x.replace("{", "(").replace("}", ")").replace("\\", "")
        return x

    try:
        tr = standard_transformations + (implicit_multiplication_application,)
        ea, eb = parse_expr(conv(a), transformations=tr), parse_expr(conv(b), transformations=tr)
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:
        return False


def math_equal(pred: Any, gold: Any, rel_tol: float = 1e-6) -> bool:
    if pred is None or gold is None:
        return False
    p, g = normalize_math(pred), normalize_math(gold)
    if not p:
        return False
    if p == g:
        return True
    pn, gn = _to_number(p), _to_number(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) <= max(rel_tol * abs(gn), 1e-9)
    # tuples / sets / intervals: compare element-wise with the same brackets
    sp, sg = re.split(r",(?![^{]*\})", p.strip("()[]{}")), re.split(r",(?![^{]*\})", g.strip("()[]{}"))
    if len(sp) == len(sg) > 1 and p[:1] == g[:1] and p[-1:] == g[-1:]:
        return all(math_equal(a, b, rel_tol) for a, b in zip(sp, sg))
    return _sympy_equal(p, g)


MATH_INSTRUCTION = ("Solve the following problem step by step. "
                    "Put your final answer within \\boxed{}.")


@register_type("math")
class MathTask(Task):
    """Free-form math word problems / competition math scored by final-answer equivalence.

    ``fields``: ``question`` (problem text), ``answer`` (gold). ``answer_after: "####"`` takes the
    text after a marker in the gold (GSM8K); ``answer_boxed: true`` extracts ``\\boxed{}`` from a
    gold solution.
    """
    default_max_new_tokens = 2048

    default_metrics = ["accuracy"]

    def default_template(self) -> str:
        return "{question}\n\n" + MATH_INSTRUCTION

    def _gold(self, row: Dict[str, Any]) -> Optional[str]:
        g = _get(row, self.fields.get("answer", "answer"))
        if g is None:
            return None
        g = str(g)
        marker = self.spec.get("answer_after")
        if marker and marker in g:
            g = g.split(marker)[-1]
        if self.spec.get("answer_boxed"):
            g = last_boxed(g) or g
        return g.strip()

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        q = _get(row, self.fields.get("question", "question"))
        gold = self._gold(row)
        if not q or (gold is None and not self.predict_only):
            return None
        v = {"question": str(q)}
        return Doc(id=self.doc_id(row, idx), vars=v, images=self._images_from_row(row), gold=gold,
                   group=self.doc_group(row))

    def gold_text(self, doc: Doc) -> str:
        return f"\\boxed{{{doc.gold}}}"

    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        pred = extract_math_answer(response)
        if self.predict_only:
            return pred, {}
        return pred, {"accuracy": float(math_equal(pred, doc.gold)), "extract_rate": float(pred is not None)}


# ============================================================================ code execution
_CODE_BLOCK = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    if not text:
        return ""
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        # the longest block is almost always the solution (others are usage examples)
        return max(blocks, key=len).strip("\n")
    if "```" in text:  # unterminated block (generation truncated)
        return text.split("```", 1)[1].split("\n", 1)[-1]
    return text


def _limits(mem_mb: int, cpu_s: int):
    def apply():
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
            os.setsid()
        except Exception:
            pass

    return apply


def run_python(program: str, timeout: float = 10.0, mem_mb: int = 2048) -> Tuple[bool, str]:
    """Execute ``program`` in a fresh, isolated-mode interpreter in a temp dir.

    Limits: wall-clock timeout, CPU seconds, address space, file size, empty environment.
    This is *damage limitation*, not a security sandbox: network and the filesystem remain
    reachable, so run code benchmarks inside a container or VM.
    """
    with tempfile.TemporaryDirectory(prefix="indibench_exec_") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(program)
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": d, "TMPDIR": d,
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        kw: Dict[str, Any] = {}
        if os.name == "posix":
            kw["preexec_fn"] = _limits(mem_mb, int(timeout) + 1)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, env=env, capture_output=True,
                               timeout=timeout, text=True, **kw)
            return r.returncode == 0, (r.stderr or "")[-600:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:  # pragma: no cover
            return False, f"{type(e).__name__}: {e}"


def code_execution_allowed(run) -> bool:
    return bool(getattr(run, "allow_code_execution", False)) or \
        os.environ.get("INDIBENCH_ALLOW_CODE_EXECUTION", "").lower() in ("1", "true", "yes")


@register_type("code")
class CodeTask(Task):
    """Function synthesis scored by unit tests (pass@1 with greedy decoding).

    ``subtype: humaneval`` -> fields ``prompt``, ``test``, ``entry_point``
    ``subtype: mbpp``      -> fields ``prompt`` (task text), ``tests`` (list of asserts), ``imports``
    """
    default_max_new_tokens = 1024

    default_metrics = ["pass@1"]
    parallel_scoring = True        # unit tests run in subprocesses -> score a batch concurrently

    def load_source(self, max_rows=None):
        if not code_execution_allowed(self.run):
            raise PermissionError(
                f"{self.name} executes model-generated code and is disabled by default. Re-run with "
                "--allow-code-execution (or run.allow_code_execution: true), preferably inside a container/VM.")
        return super().load_source(max_rows)

    def default_template(self) -> str:
        if self.spec.get("subtype") == "mbpp":
            return ("You are an expert Python programmer. {prompt}\n"
                    "Your code should pass these tests:\n\n{tests_block}\n\n"
                    "Write the complete solution in a single ```python code block.")
        return ("Complete the following Python function. Return the complete function "
                "(including the signature) in a single ```python code block.\n\n```python\n{prompt}```")

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        prompt = _get(row, self.fields.get("prompt", "prompt"))
        if not prompt:
            return None
        sub = self.spec.get("subtype", "humaneval")
        meta: Dict[str, Any] = {"subtype": sub}
        v: Dict[str, Any] = {"prompt": str(prompt)}
        if sub == "mbpp":
            tests = [str(t) for t in _as_list(_get(row, self.fields.get("tests", "test_list")))]
            if not tests:
                return None
            meta.update(tests=tests, imports=[str(i) for i in _as_list(_get(row, self.fields.get("imports", "test_imports")))])
            v["tests_block"] = "\n".join(tests)
        else:
            test, entry = _get(row, self.fields.get("test", "test")), _get(row, self.fields.get("entry_point", "entry_point"))
            if not test or not entry:
                return None
            meta.update(test=str(test), entry_point=str(entry))
        return Doc(id=self.doc_id(row, idx), vars=v, gold=None, meta=meta, group=self.doc_group(row))

    def gold_repr(self, doc: Doc) -> Any:
        return doc.meta.get("entry_point") or "unit tests"

    def gold_text(self, doc: Doc) -> str:
        return ""

    def build_program(self, doc: Doc, response: str) -> str:
        code = extract_code(response)
        m = doc.meta
        if m["subtype"] == "mbpp":
            return "\n".join(m["imports"]) + "\n" + code + "\n\n" + "\n".join(m["tests"]) + "\n"
        prompt = doc.vars["prompt"]
        if re.search(rf"def\s+{re.escape(m['entry_point'])}\s*\(", code):
            header = "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))
            program = header + "\n" + code
        else:  # model returned only the body
            program = prompt + code
        return program + "\n\n" + m["test"] + f"\n\ncheck({m['entry_point']})\n"

    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        program = self.build_program(doc, response)
        ok, err = run_python(program, float(self.spec.get("timeout", 10)), int(self.spec.get("memory_mb", 2048)))
        return ("passed" if ok else f"failed: {err.strip().splitlines()[-1] if err.strip() else 'error'}"), \
            {"pass@1": float(ok)}


# ============================================================================ JEEBench
JEE_INSTRUCTIONS = {
    "MCQ": "In this problem, only one option will be correct. Give a detailed solution and end the "
           "solution with the final answer as a single letter, in the form 'Final Answer: X'.",
    "MCQ(multiple)": "In this problem, multiple options can be correct. Give a detailed solution and end the "
                     "solution with the final answer as all correct option letters, in the form "
                     "'Final Answer: ACD'.",
    "Integer": "In this problem, the final answer will be a non-negative integer. Give a detailed solution "
               "and end the solution with 'Final Answer: N'.",
    "Numeric": "In this problem, the final answer is a numeric value. Give the numerical answer correct up "
               "to the 2nd decimal digit. Give a detailed solution and end the solution with "
               "'Final Answer: X'.",
}


def _final_segment(text: str) -> str:
    hits = re.findall(r"final\s*answer\s*[:\-]?\s*(.+)", text or "", flags=re.IGNORECASE)
    if hits:
        return hits[-1].strip()
    b = last_boxed(text or "")
    return b if b is not None else (text or "").strip().splitlines()[-1] if (text or "").strip() else ""


def jee_letters(text: str) -> str:
    """Option letters from the final answer: "BD", "A, C", "(A) and (D)", "\\boxed{ACD}" ..."""
    seg = _final_segment(text).replace("**", "")
    b = last_boxed(seg)
    seg = b if b is not None else seg
    seg = re.sub(r"\\(?:text|mathrm|textbf)\{([^{}]*)\}", r"\1", seg).replace("{", "").replace("}", "")
    m = re.match(r"^[\s(\[]*([A-D](?:[\s,;&/)(\]\[]*(?:and\s*)?[\s(\[]*[A-D])*)(?![a-z])", seg)
    if m:
        letters = re.findall(r"[A-D]", m.group(1))
    else:
        letters = re.findall(r"\(([A-D])\)", seg) or re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", seg)
    return "".join(sorted(set(letters)))


def jee_score(qtype: str, pred: Optional[str], gold: str) -> float:
    gold = str(gold).strip()
    if pred is None or pred == "":
        return 0.0
    if qtype == "MCQ":
        return float(pred == gold)
    if qtype == "MCQ(multiple)":
        p, g = set(pred), set(gold)
        if not p or p - g:
            return 0.0                      # any wrong option -> 0
        return 1.0 if p == g else 0.25 * len(p)
    try:
        pv, gv = float(pred), float(gold)
    except (ValueError, OverflowError):
        return 0.0
    if not (math.isfinite(pv) and math.isfinite(gv)):
        return 0.0                          # "inf"/"nan" answers; round() would raise
    if qtype == "Integer":
        return float(round(pv) == round(gv) and abs(pv - round(pv)) < 1e-6)
    return float(abs(pv - gv) <= 0.01)      # Numeric


@register_type("jeebench")
class JEEBenchTask(Task):
    default_metrics = ["score", "exact"]
    default_max_new_tokens = 4096

    def default_template(self) -> str:
        return "{instruction}\n\nProblem: {question}"

    def make_doc(self, row: Dict[str, Any], idx: int) -> Optional[Doc]:
        qtype = str(row.get("type", "")).strip()
        if qtype not in JEE_INSTRUCTIONS or not row.get("question"):
            return None
        rid = f"{row.get('description', '')}#{row.get('index', idx)}".replace(" ", "_")
        return Doc(id=rid, vars={"question": row["question"], "instruction": JEE_INSTRUCTIONS[qtype]},
                   gold=str(row.get("gold", "")).strip(), group=self.doc_group(row), meta={"type": qtype})

    def score(self, doc: Doc, response: str) -> Tuple[Any, Dict[str, float]]:
        qtype = doc.meta["type"]
        if qtype.startswith("MCQ"):
            pred: Optional[str] = jee_letters(response)
        else:
            n = extract_number(_final_segment(response))
            pred = None if n is None else (str(int(n)) if float(n).is_integer() else str(n))
        s = jee_score(qtype, pred, doc.gold)
        return pred, {"score": s, "exact": float(s == 1.0)}

"""
libero_plus_utils.py

LIBERO-Plus-specific helpers for the OpenVLA-OFT CycleVLA evaluation.

LIBERO-Plus (https://github.com/sylvestf/LIBERO-plus) is a robustness benchmark that
re-uses the original LIBERO suite names (`libero_spatial / libero_object / libero_goal /
libero_10`) but adds ~10k perturbed task *variants* across 7 perturbation categories
(Camera / Robot / Language / Light / Background / Noise / Layout). Every perturbation is
encoded in the task *name* (e.g. `..._view_..._initstate_..._noise_3`, `..._table_1`,
`..._language_2`), and the LIBERO-Plus env wrapper (`ControlEnv.__init__`) applies it
automatically from that name — so the only thing the eval scripts need on top of the
stock LIBERO eval is:

  1. mapping each variant to its perturbation category (for per-category reporting),
  2. recovering the *canonical* task instruction that CycleVLA's FSM subtask decomposition
     was authored against, and
  3. optionally sub-sampling variants for a cheaper-but-representative run.

This module centralises all of that so the eval scripts stay close mirrors of the
original LIBERO ones.

Why canonical-task recovery is needed
-------------------------------------
CycleVLA feeds the VLA *subtask* strings produced by an FSM that rigidly string-parses the
task instruction. For the 6 non-Language categories the instruction is unchanged, so the
canonical task is recovered for free from the task *name* (the perturbation suffix strips
off cleanly — `strip_perturbation_suffix` + `filename_to_instruction`). For the Language
category the instruction is an LLM-rewritten paraphrase that the FSM cannot parse, so we
map the rewrite back to one of the ~10 known canonical tasks with a GPT call (cached to
disk). GPT is therefore invoked *only* for Language-category tasks.
"""

import json
import os
import re
import random
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# Short CLI aliases -> the exact category strings used in LIBERO-Plus's
# `task_classification.json`.
CATEGORY_CANONICAL = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "noise": "Sensor Noise",
    "layout": "Objects Layout",
}
CATEGORY_FULL_NAMES = set(CATEGORY_CANONICAL.values())

# Full category string -> short, filesystem-friendly slug (used for output sub-dirs and
# per-category reporting headers).
CATEGORY_SLUG = {full: short.capitalize() for short, full in CATEGORY_CANONICAL.items()}


def category_slug(full_category):
    """Short filesystem-friendly name for a category (e.g. 'Camera Viewpoints' -> 'Camera')."""
    return CATEGORY_SLUG.get(full_category, full_category.replace(" ", "_"))

# Per-suite variant counts per category, taken directly from
# `LIBERO-plus/libero/libero/benchmark/task_classification.json` (total = 10,030).
# Used only to sanity-check that an `--eval_fraction 100` run covers every variant.
PLUS_CATEGORY_COUNTS = {
    "libero_spatial": {"Background Textures": 258, "Robot Initial States": 350,
                       "Camera Viewpoints": 376, "Language Instructions": 390,
                       "Sensor Noise": 351, "Objects Layout": 385, "Light Conditions": 292},
    "libero_object":  {"Background Textures": 248, "Robot Initial States": 398,
                       "Camera Viewpoints": 396, "Language Instructions": 354,
                       "Sensor Noise": 422, "Objects Layout": 403, "Light Conditions": 297},
    "libero_goal":    {"Background Textures": 281, "Robot Initial States": 409,
                       "Camera Viewpoints": 408, "Language Instructions": 410,
                       "Sensor Noise": 379, "Objects Layout": 425, "Light Conditions": 279},
    "libero_10":      {"Background Textures": 289, "Robot Initial States": 393,
                       "Camera Viewpoints": 419, "Language Instructions": 383,
                       "Sensor Noise": 449, "Objects Layout": 312, "Light Conditions": 274},
}

LANGUAGE_CATEGORY = "Language Instructions"
NOISE_CATEGORY = "Sensor Noise"

# LIBERO-Plus's sensor-noise corruptions (`fog` builds a fixed 256x256 `plasma_fractal`,
# and the other corruptions' severities are calibrated for 256x256) require the agentview
# to be rendered at 256x256 — see `env_render_resolution` below.
NOISE_RENDER_RESOLUTION = 256

# Paths resolved relative to this file (experiments/robot/libero_plus_utils.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../"))
_METAINFO_DIR = os.path.join(_THIS_DIR, "libero")
_TASK_CLASSIFICATION_PATH = os.path.join(
    _REPO_ROOT, "LIBERO-plus", "libero", "libero", "benchmark", "task_classification.json"
)

# The OpenAI model used for the Language-category instruction matcher — kept identical to
# the VLM detector used elsewhere in the eval so a single API key/model covers both.
_GPT_MATCH_MODEL = "gpt-5.5"


# ---------------------------------------------------------------------------
# String normalisation / name parsing
# ---------------------------------------------------------------------------

def _normalize(s):
    """Normalise an instruction string for robust exact comparison.

    Lowercases, strips, collapses internal whitespace and drops surrounding punctuation —
    so capitalisation / spacing / trailing punctuation never break a match.
    """
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,!?;:\"'()")
    return s


# A perturbation suffix always begins with one of these markers *followed by a digit*
# (e.g. `_table_1`, `_view_0_0_100_2_352`, `_language_2`, `_level1_sample1`). The digit
# requirement is essential: base task names legitimately contain words like
# `..._from_table_center_...`, where `_table_` is NOT a suffix because "center" follows.
_SUFFIX_RE = re.compile(r"_(?:table|tb|light|language|view|add|noise)_\d|_level\d")


def strip_perturbation_suffix(task_name):
    """Return the canonical base task name with every perturbation suffix removed.

    The LIBERO-Plus task name encodes the base task for *all 7* categories, so this is an
    exact, deterministic recovery — no model call needed.
    """
    m = _SUFFIX_RE.search(task_name)
    return task_name[:m.start()] if m else task_name


def filename_to_instruction(base_name):
    """Convert a base task name to the FSM-style instruction string.

    Replicates LIBERO-Plus's `grab_language_from_filename` for non-`_language_` tasks:
    underscores become spaces, and for LIBERO-100-style names (`libero_10`/`libero_90`,
    which start with an uppercase scene prefix like `KITCHEN_SCENE4_...`) the
    `<SCENE><n>_` prefix is dropped. The result matches exactly the instruction strings
    the CycleVLA FSM (`pick_place_states` / `complex_states`) was authored against.
    """
    x = base_name
    if x and x[0].isupper() and "SCENE" in x:
        idx = x.find("SCENE")
        # "SCENE10_" is 8 chars; "SCENE<1-digit>_" is 7 chars.
        offset = 8 if "SCENE10" in x else 7
        x = x[idx + offset:]
    return " ".join(x.split("_")).strip().lower()


# ---------------------------------------------------------------------------
# Canonical task registry + perturbation-category lookup
# ---------------------------------------------------------------------------

_canonical_cache = {}


def get_canonical_tasks(suite):
    """Return the ~10 canonical FSM-style instructions for a suite.

    Built from the keys of `experiments/robot/libero/libero_<suite>_metainfo.json`, which
    are exactly the original LIBERO task set the FSM decomposition was written for.
    """
    if suite not in _canonical_cache:
        suite_key = suite.replace("libero_", "")  # libero_10 -> "10"
        path = os.path.join(_METAINFO_DIR, f"libero_{suite_key}_metainfo.json")
        with open(path, "r") as f:
            meta = json.load(f)
        _canonical_cache[suite] = [_normalize(" ".join(k.split("_"))) for k in meta]
    return _canonical_cache[suite]


# Convenient module-level view of the registry.
CANONICAL_TASKS = {suite: get_canonical_tasks(suite) for suite in SUITES}


_classification_cache = None


def load_task_classification():
    """Load LIBERO-Plus's `task_classification.json` as {suite: {name: category}}."""
    global _classification_cache
    if _classification_cache is None:
        with open(_TASK_CLASSIFICATION_PATH, "r") as f:
            raw = json.load(f)
        _classification_cache = {
            suite: {entry["name"]: entry["category"] for entry in entries}
            for suite, entries in raw.items()
        }
    return _classification_cache


def get_category_map(suite):
    """Return {task_name: category} for a suite (category = exact classification string)."""
    return load_task_classification()[suite]


def normalize_category(category):
    """Resolve a CLI category value ('all', a short alias, or a full name) to its
    canonical form. Returns 'all' unchanged; raises on an unknown value."""
    if category is None:
        return "all"
    c = category.strip()
    if c.lower() == "all":
        return "all"
    if c.lower() in CATEGORY_CANONICAL:
        return CATEGORY_CANONICAL[c.lower()]
    # Allow passing the full classification string directly (case-insensitive).
    for full in CATEGORY_FULL_NAMES:
        if c.lower() == full.lower():
            return full
    raise ValueError(
        f"Unknown category '{category}'. Use 'all' or one of: "
        f"{sorted(CATEGORY_CANONICAL.keys())}"
    )


# ---------------------------------------------------------------------------
# Instruction resolution (incl. the Language-category GPT matcher)
# ---------------------------------------------------------------------------

def get_plus_clean_instruction(env):
    """Return the resolved bddl's `(:language ...)` instruction from a LIBERO-Plus env.

    `ControlEnv` stores the real (perturbation-resolved) instruction on
    `env.language_instruction`. The benchmark's `task.language` must NOT be used: for
    LIBERO-Plus it is derived from the raw filename and is dirty.
    """
    instr = getattr(env, "language_instruction", None)
    if instr is None:  # defensive: some wrappers nest the ControlEnv one level down
        instr = getattr(getattr(env, "env", None), "language_instruction", None)
    if instr is None:
        raise ValueError("Could not read `language_instruction` from the LIBERO-Plus env.")
    return instr


def _exact_match(instruction, canonical_list):
    """Return the canonical entry whose normalised form equals `instruction`, else None."""
    norm = _normalize(instruction)
    for cand in canonical_list:
        if _normalize(cand) == norm:
            return cand
    return None


def _cache_path(cache_dir, suite):
    """Path of the Language-category instruction->canonical-task cache for a suite.

    One file *per suite* (not a single shared file) so parallel runs of *different*
    suites never race on the same JSON. The file IS, however, shared by the transit and
    mbr scripts (they default `instruction_cache_dir` to the same directory) — this is
    intentional: Stage 2 (mbr) reuses Stage 1 (transit)'s mappings, so the GPT matcher
    fires once per unique rewritten instruction total, and both stages resolve each
    rewrite to the identical canonical task (keeping the FSM subtask decomposition
    consistent across stages, as the failed-episode rerun mechanism requires). Within one
    (suite, category) slice the launcher runs transit then mbr sequentially, so there is
    no concurrent write to this file.
    """
    return os.path.join(cache_dir, f"task_instruction_mapping_{suite}.json")


def _load_cache(cache_dir, suite):
    path = _cache_path(cache_dir, suite)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache_dir, suite, cache):
    os.makedirs(cache_dir, exist_ok=True)
    with open(_cache_path(cache_dir, suite), "w") as f:
        json.dump(cache, f, indent=2)


def _gpt_match(instruction, canonical_list):
    """Ask GPT which canonical task a paraphrased instruction denotes.

    Returns the matched canonical string. The model is asked for the *index* (not the
    text) so its answer is trivial to validate against `canonical_list`.
    """
    from openai import OpenAI

    # Source OPENAI_API_KEY from the repo-root `.env` (per SETUP.md). Done here so the
    # matcher works regardless of caller — the transit eval script does not itself call
    # load_dotenv() (only the mbr script does).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set — required for the Language-category matcher.")
    client = OpenAI(api_key=api_key)

    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(canonical_list))
    prompt = (
        "A robot is given a task instruction that has been paraphrased/reworded. "
        "Identify which canonical task it refers to.\n\n"
        f"Canonical tasks:\n{numbered}\n\n"
        f"Paraphrased instruction:\n\"{instruction}\"\n\n"
        f"Reply with ONLY the number (1-{len(canonical_list)}) of the canonical task "
        "that has the same goal. Output the number and nothing else."
    )
    completion = client.chat.completions.create(
        model=_GPT_MATCH_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
    )
    text = completion.choices[0].message.content.strip()
    m = re.search(r"\d+", text)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < len(canonical_list):
            return canonical_list[idx]
    raise ValueError(f"GPT matcher returned an unparseable answer '{text}' for: {instruction}")


def resolve_canonical_task(task_name, category, env, suite, cache_dir, log_fn=print):
    """Resolve the FSM-style canonical instruction for a LIBERO-Plus task.

    - 6 non-Language categories: recovered deterministically from the task name
      (`strip_perturbation_suffix` -> `filename_to_instruction`) and confirmed by an exact
      match against the canonical registry. No GPT call.
    - Language category: the (rewritten) `env.language_instruction` is matched to a
      canonical task — exact match first, then a disk cache, then a GPT call. The
      filename-derived base task is logged alongside as a free ground-truth correctness
      check on the matcher.
    """
    category = normalize_category(category)
    canonical_list = get_canonical_tasks(suite)

    # Filename-derived ground truth — exact for every category (the suffix strips cleanly).
    filename_instr = filename_to_instruction(strip_perturbation_suffix(task_name))
    ground_truth = _exact_match(filename_instr, canonical_list)

    if category != LANGUAGE_CATEGORY:
        if ground_truth is None:
            # Should not happen; fall back to the raw filename instruction so the run
            # continues, but make the anomaly loud.
            log_fn(f"[libero_plus] WARN: no canonical match for '{task_name}' "
                   f"(instr='{filename_instr}'); using raw instruction.")
            return filename_instr
        return ground_truth

    # --- Language category: match the rewritten instruction back to a canonical task ---
    raw = get_plus_clean_instruction(env)

    # 1) Some rewrites happen to equal a canonical instruction verbatim.
    match = _exact_match(raw, canonical_list)
    source = "exact"

    # 2) Disk cache (one GPT call per unique rewrite, ever).
    if match is None:
        cache = _load_cache(cache_dir, suite)
        if raw in cache:
            match, source = cache[raw], "cache"
        else:
            # 3) GPT call.
            match = _gpt_match(raw, canonical_list)
            source = "gpt"
            cache[raw] = match
            _save_cache(cache_dir, suite, cache)

    correct = (ground_truth is not None and _normalize(match) == _normalize(ground_truth))
    log_fn(f"[libero_plus] Language match ({source}): \"{raw}\" -> \"{match}\" "
           f"| filename_gt=\"{ground_truth}\" | correct={correct}")
    return match


def rollout_label(category, env, canonical_instruction):
    """Instruction string used to name / subtitle a saved rollout video.

    For the Language category this is the *original rewritten* instruction the policy was
    actually given (`env.language_instruction`), so each Language rollout is identifiable
    by its real (paraphrased) input. For the other 6 categories the instruction is
    unchanged from the task, so the canonical instruction is used. Note this only affects
    the saved-video label — the canonical instruction always drives the CycleVLA FSM.
    """
    if normalize_category(category) == LANGUAGE_CATEGORY:
        return get_plus_clean_instruction(env)
    return canonical_instruction


def env_render_resolution(category, default_res):
    """Camera render resolution (`camera_heights`/`camera_widths`) for a LIBERO-Plus variant.

    The Noise category MUST render at 256x256: LIBERO-Plus's sensor-noise corruptions in
    `env_wrapper.py` are written for that size — `fog` adds a hardcoded 256x256
    `plasma_fractal` (so a 1024x1024 agentview hits a broadcast error), and the other
    corruptions' severity params are calibrated for 256x256. The other 6 categories use
    `default_res` (the eval's `--env_img_res`, default 1024) for higher-quality video.
    This only affects the source render / saved-video size — the policy input is resized
    to 224 regardless.
    """
    if normalize_category(category) == NOISE_CATEGORY:
        return NOISE_RENDER_RESOLUTION
    return default_res


# ---------------------------------------------------------------------------
# Variant sub-sampling
# ---------------------------------------------------------------------------

def select_plus_tasks(task_suite, suite_name, category, eval_fraction, seed):
    """Return the (task_id, category) list to evaluate for a LIBERO-Plus suite.

    Enumerates every variant in the loaded benchmark suite, looks up its perturbation
    category from `task_classification.json`, filters to `category` (unless 'all'), and
    sub-samples to `eval_fraction%`. Both the transit and mbr scripts call this with the
    same args, so they evaluate the identical, identically-ordered task list.
    """
    category = normalize_category(category)
    cat_map = get_category_map(suite_name)
    items = []
    for task_id in range(task_suite.n_tasks):
        name = task_suite.get_task(task_id).name
        cat = cat_map.get(name)
        if cat is None:
            # Variant not present in task_classification.json — skip it rather than
            # guessing a category.
            continue
        if category != "all" and cat != category:
            continue
        items.append((task_id, cat))
    return subsample_tasks(items, eval_fraction, seed)


def subsample_tasks(items, eval_fraction, seed):
    """Deterministically sub-sample LIBERO-Plus variants for a cheaper representative run.

    `items` is a list of (task_id, category) pairs. Within each category `eval_fraction%`
    of the variants are kept by uniform-random sampling — valid because every variant is
    itself an independent random draw from the perturbation distribution. At
    `eval_fraction == 100` the full set is returned unchanged (the LIBERO-Plus paper
    protocol). The result is sorted by `task_id`, and the per-category RNG is seeded only
    from (`seed`, category) — so the transit and mbr scripts, given the same `seed` and
    `eval_fraction`, evaluate the identical task set in the identical order (keeping
    episode numbering aligned for the failed-episode rerun mechanism).
    """
    items = sorted(items, key=lambda x: x[0])
    if eval_fraction >= 100:
        return items

    frac = eval_fraction / 100.0
    by_cat = defaultdict(list)
    for task_id, category in items:
        by_cat[category].append((task_id, category))

    selected = []
    for category in sorted(by_cat):
        group = sorted(by_cat[category], key=lambda x: x[0])
        rng = random.Random(f"{seed}-{category}")
        rng.shuffle(group)
        keep = max(1, round(frac * len(group)))
        selected.extend(group[:keep])

    return sorted(selected, key=lambda x: x[0])

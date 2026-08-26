import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# -------------------------
# Config
# -------------------------
PROJECT_ROOT = Path("~/mack/yyin5/HUGSIM").expanduser()
BENCH_ROOT = PROJECT_ROOT / "outputs" / "benchmark_drivor_mpc"  # "benchmark_new_2"
MODEL_NAME = "dynamo_drivor_mpc_orig"  # base model name without private_ prefix

# If MODEL_NAME is passed with private_ prefix, normalize to a base name.
PRIVATE_PREFIX = "private_"
BASE_MODEL_NAME = MODEL_NAME[len(PRIVATE_PREFIX):] if MODEL_NAME.startswith(PRIVATE_PREFIX) else MODEL_NAME
PRIVATE_MODEL_NAME = f"{PRIVATE_PREFIX}{BASE_MODEL_NAME}"

DATASET_ORDER = ["kitti360", "nuscenes", "pandaset", "waymo"]
DIFF_ORDER = ["easy", "medium", "hard", "extreme"]
DIFF_RE = re.compile(r"(easy|medium|hard|extreme)", re.IGNORECASE)

DISPLAY_NAMES = {
    "kitti360": "KITTI360",
    "nuscenes": "nuScenes",
    "pandaset": "Pandaset",
    "waymo": "Waymo",
}

# Canonical score order and key mapping.
SCORE_KEYS = [
    ("nc", "NC"),
    ("dac", "DAC"),
    ("ttc", "TTC"),
    ("c", "COM"),
    ("rc", "RC"),
    # ("pdms", "PDMS"),
    ("hdscore", "HDSCORE"),
]


# -------------------------
# Helpers
# -------------------------
def avg_dict(acc):
    return {k: float(np.mean(v)) for k, v in acc.items() if len(v) > 0}


def ensure_floatable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def to_float_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt(v):
    return "" if v is None or np.isnan(v) else f"{v:.4f}"


def discover_dataset_roots(model_name):
    roots = {}
    for ds in DATASET_ORDER:
        candidate = BENCH_ROOT / f"{ds}_{model_name}"
        if candidate.is_dir():
            roots[ds] = [candidate]
    return roots


def combine_root_maps(*root_maps):
    combined = defaultdict(list)
    for root_map in root_maps:
        for ds, roots in root_map.items():
            combined[ds].extend(roots)

    out = {}
    for ds in DATASET_ORDER:
        if combined.get(ds):
            out[ds] = combined[ds]
    return out


def aggregate_from_roots(dataset_roots):
    # -------------------------
    # Accumulators
    # -------------------------
    global_totals = defaultdict(list)
    global_details = defaultdict(lambda: defaultdict(list))

    per_dataset_totals = defaultdict(lambda: defaultdict(list))
    per_dataset_details = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    per_diff_totals = defaultdict(lambda: defaultdict(list))
    per_diff_details = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    per_dataset_diff_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_dataset_diff_details = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    missing_per_dataset = defaultdict(list)
    present_per_dataset = defaultdict(list)
    all_eval_files = []

    # -------------------------
    # Aggregate data
    # -------------------------
    for ds, roots in dataset_roots.items():
        scene_dirs = []
        for ds_root in roots:
            scene_dirs.extend([p for p in ds_root.iterdir() if p.is_dir()])

        for scene_dir in sorted(scene_dirs, key=lambda p: str(p)):
            eval_fp = scene_dir / "eval.json"
            diff_match = DIFF_RE.search(scene_dir.name)
            diff = diff_match.group(1).lower() if diff_match else "unknown"

            if not eval_fp.exists():
                missing_per_dataset[ds].append(scene_dir)
                continue

            present_per_dataset[ds].append(scene_dir)
            all_eval_files.append(eval_fp)

            data = json.loads(eval_fp.read_text())
            top = ensure_floatable(data)

            # Global
            for k, v in top.items():
                global_totals[k].append(v)
            for th, subs in data.get("details", {}).items():
                for k, v in subs.items():
                    fv = to_float_or_none(v)
                    if fv is not None:
                        global_details[th][k].append(fv)

            # Per-dataset
            for k, v in top.items():
                per_dataset_totals[ds][k].append(v)
            for th, subs in data.get("details", {}).items():
                for k, v in subs.items():
                    fv = to_float_or_none(v)
                    if fv is not None:
                        per_dataset_details[ds][th][k].append(fv)

            # Per-difficulty
            for k, v in top.items():
                per_diff_totals[diff][k].append(v)
            for th, subs in data.get("details", {}).items():
                for k, v in subs.items():
                    fv = to_float_or_none(v)
                    if fv is not None:
                        per_diff_details[diff][th][k].append(fv)

            # Per (dataset, difficulty)
            for k, v in top.items():
                per_dataset_diff_totals[ds][diff][k].append(v)
            for th, subs in data.get("details", {}).items():
                for k, v in subs.items():
                    fv = to_float_or_none(v)
                    if fv is not None:
                        per_dataset_diff_details[ds][diff][th][k].append(fv)

    # -------------------------
    # Compute means
    # -------------------------
    global_means = avg_dict(global_totals)
    global_details_mean = {th: avg_dict(sub) for th, sub in global_details.items()}
    per_dataset_means = {ds: avg_dict(tot) for ds, tot in per_dataset_totals.items()}
    per_dataset_details_mean = {
        ds: {th: avg_dict(sub) for th, sub in details.items()} for ds, details in per_dataset_details.items()
    }
    per_diff_means = {diff: avg_dict(tot) for diff, tot in per_diff_totals.items()}
    per_diff_details_mean = {
        diff: {th: avg_dict(sub) for th, sub in details.items()} for diff, details in per_diff_details.items()
    }
    per_dataset_diff_means = {
        ds: {diff: avg_dict(tot) for diff, tot in diff_map.items()} for ds, diff_map in per_dataset_diff_totals.items()
    }
    per_dataset_diff_details_mean = {
        ds: {diff: {th: avg_dict(sub) for th, sub in th_map.items()} for diff, th_map in diff_map.items()}
        for ds, diff_map in per_dataset_diff_details.items()
    }

    return {
        "dataset_roots": dataset_roots,
        "eval_files": all_eval_files,
        "missing_per_dataset": missing_per_dataset,
        "present_per_dataset": present_per_dataset,
        "averages": {
            "global": global_means,
            "per_dataset": per_dataset_means,
            "per_difficulty": per_diff_means,
            "per_dataset_difficulty": per_dataset_diff_means,
        },
        "details": {
            "global": global_details_mean,
            "per_dataset": per_dataset_details_mean,
            "per_difficulty": per_diff_details_mean,
            "per_dataset_difficulty": per_dataset_diff_details_mean,
        },
    }


def get_dataset_diff_mean(stats, ds, diff, key):
    try:
        return float(stats["averages"]["per_dataset_difficulty"][ds][diff][key])
    except Exception:
        return np.nan


def get_dataset_mean(stats, ds, key):
    try:
        return float(stats["averages"]["per_dataset"][ds][key])
    except Exception:
        return np.nan


def get_diff_mean(stats, diff, key):
    try:
        return float(stats["averages"]["per_difficulty"][diff][key])
    except Exception:
        return np.nan


def get_global_mean(stats, key):
    try:
        return float(stats["averages"]["global"][key])
    except Exception:
        return np.nan


def print_tables(section_name, stats):
    print(f"\n\n========== {section_name} ==========")
    print("Found dataset roots:")
    for ds in DATASET_ORDER:
        roots = stats["dataset_roots"].get(ds, [])
        if roots:
            for p in roots:
                print(f"  {ds}: {p}")

    print("\n=== PER (DATASET, DIFFICULTY) MEANS ===")
    header_dataset_diff = [f"{DISPLAY_NAMES.get(ds, ds)} {d.capitalize()}" for ds in DATASET_ORDER for d in DIFF_ORDER]
    print("\t".join(header_dataset_diff))

    for key, _ in SCORE_KEYS:
        row = [fmt(get_dataset_diff_mean(stats, ds, d, key)) for ds in DATASET_ORDER for d in DIFF_ORDER]
        print("\t".join(row))

    print("\n=== PER-DATASET MEANS ===")
    header_dataset = [DISPLAY_NAMES.get(ds, ds) for ds in DATASET_ORDER]
    print("\t".join(header_dataset))

    for key, _ in SCORE_KEYS:
        row = [fmt(get_dataset_mean(stats, ds, key)) for ds in DATASET_ORDER]
        print("\t".join(row))

    print("\n=== PER-DIFFICULTY MEANS ===")
    header_diff = [d.capitalize() for d in DIFF_ORDER]
    print("\t".join(header_diff))

    for key, _ in SCORE_KEYS:
        row = [fmt(get_diff_mean(stats, d, key)) for d in DIFF_ORDER]
        print("\t".join(row))

    print("\n=== GLOBAL MEAN ===")
    row = [fmt(get_global_mean(stats, key)) for key, _ in SCORE_KEYS]
    print("\n".join(row))

    print("\n=== Folders missing eval.json (per dataset) ===")
    total_missing = 0
    for ds in DATASET_ORDER:
        missing = stats["missing_per_dataset"].get(ds, [])
        total_missing += len(missing)
        print(f"\n[{ds}]  missing: {len(missing)}")
        for d in missing:
            print(f"  {d}")

    print(f"\nTotal eval.json files found: {len(stats['eval_files'])}")
    print(f"Total missing folders: {total_missing}")


def save_outputs(section_slug, stats):
    # Save numeric means as TSV.
    tsv_path = BENCH_ROOT / f"aggregate_eval_{BASE_MODEL_NAME}_{section_slug}_means_for_figure.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        # Section 1
        header_dataset_diff = [
            f"{DISPLAY_NAMES.get(ds, ds)} {d.capitalize()}" for ds in DATASET_ORDER for d in DIFF_ORDER
        ]
        f.write("## Per (dataset, difficulty)\n")
        f.write("\t".join(header_dataset_diff) + "\n")
        for key, _ in SCORE_KEYS:
            row = [fmt(get_dataset_diff_mean(stats, ds, d, key)) for ds in DATASET_ORDER for d in DIFF_ORDER]
            f.write("\t".join(row) + "\n")

        # Section 2
        f.write("\n## Per-dataset means\n")
        f.write("\t".join(DATASET_ORDER) + "\n")
        for key, _ in SCORE_KEYS:
            row = [fmt(get_dataset_mean(stats, ds, key)) for ds in DATASET_ORDER]
            f.write("\t".join(row) + "\n")

        # Section 3
        f.write("\n## Per-difficulty means\n")
        f.write("\t".join(DIFF_ORDER) + "\n")
        for key, _ in SCORE_KEYS:
            row = [fmt(get_diff_mean(stats, d, key)) for d in DIFF_ORDER]
            f.write("\t".join(row) + "\n")

        # Section 4
        f.write("\n## Global mean\n")
        row = [fmt(get_global_mean(stats, key)) for key, _ in SCORE_KEYS]
        f.write("\n".join(row) + "\n")

    print(f"\nSaved all numeric means to: {tsv_path}")

    # Save full JSON report.
    report = {
        "section": section_slug,
        "base_model": BASE_MODEL_NAME,
        "dataset_roots": {ds: [str(p) for p in stats["dataset_roots"].get(ds, [])] for ds in DATASET_ORDER},
        "averages": stats["averages"],
        "details": stats["details"],
        "missing_folders": {
            ds: [str(p) for p in stats["missing_per_dataset"].get(ds, [])] for ds in DATASET_ORDER
        },
        "present_folders": {
            ds: [str(p) for p in stats["present_per_dataset"].get(ds, [])] for ds in DATASET_ORDER
        },
        "eval_file_count": len(stats["eval_files"]),
    }

    out_path = BENCH_ROOT / f"aggregate_eval_{BASE_MODEL_NAME}_{section_slug}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved aggregate report to: {out_path}")


# -------------------------
# Main
# -------------------------
normal_roots = discover_dataset_roots(BASE_MODEL_NAME)
private_roots = discover_dataset_roots(PRIVATE_MODEL_NAME)
combined_roots = combine_root_maps(normal_roots, private_roots)

if not combined_roots:
    raise SystemExit(
        f"No dataset folders found for '{BASE_MODEL_NAME}' or '{PRIVATE_MODEL_NAME}' under {BENCH_ROOT}"
    )

sections = [
    ("NORMAL", "normal", normal_roots),
    ("PRIVATE", "private", private_roots),
    ("COMBINED", "combined", combined_roots),
]

for section_name, section_slug, roots in sections:
    if not roots:
        print(f"\n\n========== {section_name} ==========")
        print("No dataset roots found. Skipping.")
        continue

    stats = aggregate_from_roots(roots)
    print_tables(section_name, stats)
    save_outputs(section_slug, stats)

"""
ARC Prize 2025 – Baseline Solver (PEAS Agent)
Author(s): [Your Name]
University of Missouri – AI Mini-Project

This baseline implements a simple color-frequency and mapping-based solver for the
ARC-AGI-2 dataset. It can evaluate both training and evaluation sets.
"""

import json, os
import numpy as np
from pathlib import Path


# --------------------------
# Visualization Utilities
# --------------------------
def visualize_grid(grid):
    """Pretty-print grid with simple ASCII colors."""
    color_map = {
        0: "  ", 1: "██", 2: "▒▒", 3: "░░", 4: "▓▓",
        5: "▤▤", 6: "▧▧", 7: "▦▦", 8: "▨▨", 9: "▩▩"
    }
    for row in grid:
        print("".join(color_map.get(c, "??") for c in row))
    print()


# --------------------------
# Step 1: Global Digest (Training Phase)
# --------------------------
def digest_training_data(folder_path):
    """
    Analyze all training JSONs to compute a global prior.
    Returns the dominant output color across all training examples.
    """
    folder = Path(folder_path)
    color_counts = {i: 0 for i in range(10)}

    for file in folder.glob("*.json"):
        data = json.load(open(file))
        for pair in data.get("train", []):
            out = np.array(pair["output"])
            unique, counts = np.unique(out, return_counts=True)
            for u, c in zip(unique, counts):
                color_counts[int(u)] += int(c)

    dominant_color = max(color_counts, key=color_counts.get)
    print(f"📊 Dominant training color: {dominant_color}")
    return dominant_color


# --------------------------
# Step 2: Solver Logic
# --------------------------
def infer_color_permutation_csp(train_pairs, enforce_injective=True):
    """
    CSP-style color mapping:
    - Variables: input colors observed in training inputs.
    - Domains: singleton choices implied by aligned input/output cells.
    - Constraints: functional; injectivity.
    If any conflict is found, return None so we fall back to the baseline.
    """
    mapping = {}
    used_targets = set()

    for pair in train_pairs:
        a = np.array(pair["input"])
        b = np.array(pair["output"])
        if a.shape != b.shape:
            continue

        for x, y in zip(a.ravel(), b.ravel()):
            x, y = int(x), int(y)

            # functional constraint
            if x in mapping and mapping[x] != y:
                return None  # conflict → not a pure color-permutation task

            # injectivity check
            if enforce_injective and x not in mapping and y in used_targets:
                return None  # two inputs trying to map to same output

            if x not in mapping:
                mapping[x] = y
                if enforce_injective:
                    used_targets.add(y)

    return mapping if mapping else None


def csp_solver(train_pairs, test_input, global_color=None):
    test = np.array(test_input)
    csp_map = infer_color_permutation_csp(train_pairs, enforce_injective=True)
    if csp_map:
        pred = np.vectorize(lambda c: csp_map.get(int(c), int(c)))(test)
        return pred.tolist()
    # fallback to your original baseline
    return solve_arc_task_baseline(train_pairs, test_input, global_color)

def solve_arc_task_baseline(train_pairs, test_input, global_color=None):
    """
    Learns a simple color mapping from training and applies it to test input.
    Falls back to global color fill if mapping fails.
    """
    input_grid = np.array(test_input)
    predicted = input_grid.copy()

    # Learn per-task color mapping (input→output)
    color_map = {}
    try:
        for pair in train_pairs:
            in_colors = np.unique(pair["input"])
            out_colors = np.unique(pair["output"])
            for i, o in zip(in_colors, out_colors):
                color_map[i] = o
    except Exception:
        pass

    # Apply learned mapping
    try:
        predicted = np.vectorize(lambda c: color_map.get(c, global_color if global_color is not None else c))(input_grid)
    except Exception:
        predicted = np.full_like(input_grid, global_color if global_color is not None else 0)

    return predicted.tolist()


# --------------------------
# Step 3: Evaluation Runner
# --------------------------
def run_on_folder(folder_path, global_color=None):
    """Run solver on all ARC JSON tasks in folder."""
    folder = Path(folder_path)
    files = sorted(folder.glob("*.json"))
    if not files:
        print(f"⚠️ No JSON files in {folder}")
        return

    correct, total = 0, 0
    for file in files:
        data = json.load(open(file))
        task_id = file.stem
        print(f"\n🧩 Task: {task_id}")

        for i, test in enumerate(data["test"]):
            prediction = csp_solver(data["train"], test["input"], global_color)
            expected = test.get("output")
            if expected is not None:
                is_correct = prediction == expected
                total += 1
                correct += int(is_correct)
                print(f"  ▶ Test {i+1}: {'✅' if is_correct else '❌'}")
            else:
                print(f"  ▶ Test {i+1}: (no ground truth in eval set)")
            visualize_grid(prediction)

    if total:
        print(f"\n📈 Accuracy: {correct}/{total} = {correct/total:.2%}")
    else:
        print("⚠️ No labeled test outputs available.")



# --------------------------
# Entry Point
# --------------------------
if __name__ == "__main__":
    training_path = "agi-2-eval-set/data/training"
    eval_path = "agi-2-eval-set/data/evaluation"

    print("🔍 Learning from training set...")
    global_color = digest_training_data(training_path)

    print("\n🧠 Running on evaluation set...")
    run_on_folder(eval_path, global_color)

import json
import os
import numpy as np
from pathlib import Path


# --------------------------
# Visualization
# --------------------------
def visualize_grid(grid):
    """Pretty-print grid with ASCII colors."""
    color_map = {
        0: "  ", 1: "██", 2: "▒▒", 3: "░░", 4: "▓▓",
        5: "▤▤", 6: "▧▧", 7: "▦▦", 8: "▨▨", 9: "▩▩"
    }
    for row in grid:
        print("".join(color_map.get(c, "??") for c in row))
    print()


# --------------------------
# Solver Logic
# --------------------------
def solve_arc_task(train_pairs, test_input):
    """Apply simple pattern rules to predict test output."""
    input_grid = np.array(test_input)
    predicted = input_grid.copy()  # baseline

    # Strategy 1: dominant color fill
    try:
        all_out_colors = []
        for pair in train_pairs:
            output = np.array(pair["output"])
            unique, counts = np.unique(output, return_counts=True)
            dominant = unique[np.argmax(counts)]
            all_out_colors.append(dominant)
        fill_color = max(set(all_out_colors), key=all_out_colors.count)
        predicted = np.full_like(input_grid, fill_color)
    except Exception:
        pass

    # Strategy 2: pixelwise diff (if same shape)
    try:
        diffs = []
        for pair in train_pairs:
            in_arr = np.array(pair["input"])
            out_arr = np.array(pair["output"])
            if in_arr.shape == out_arr.shape:
                diff = (out_arr - in_arr) % 10
                diffs.append(diff)
        if diffs:
            avg_diff = np.round(np.mean(diffs, axis=0)).astype(int) % 10
            if avg_diff.shape == input_grid.shape:
                predicted = (input_grid + avg_diff) % 10
    except Exception:
        pass

    return predicted.tolist()


# --------------------------
# Dataset Runner
# --------------------------
def run_on_folder(folder_path):
    """Run solver on all ARC JSONs in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"⚠️ Folder {folder_path} not found.")
        return

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        print(f"⚠️ No JSON files found in {folder_path}")
        return

    correct, total = 0, 0

    for file in json_files:
        with open(file, "r") as f:
            task = json.load(f)

        task_id = file.stem
        train_pairs = task["train"]
        test_pairs = task["test"]

        print(f"\n🧩 Task: {task_id}")

        for i, test_pair in enumerate(test_pairs):
            prediction = solve_arc_task(train_pairs, test_pair["input"])
            expected = test_pair["output"]

            is_correct = prediction == expected
            total += 1
            correct += int(is_correct)

            print(f"  ▶ Test {i+1}: {'✅ Correct' if is_correct else '❌ Incorrect'}")
            visualize_grid(test_pair["input"])
            print("Predicted:")
            visualize_grid(prediction)
            print("Expected:")
            visualize_grid(expected)

    print("\n📊 Final Results")
    print(f"✅ Accuracy: {correct}/{total} = {correct/total:.2%}")


# --------------------------
# Entry Point
# --------------------------
if __name__ == "__main__":
    # Change this to your ARC dataset folder
    folder_path = "agi-2-eval-set/data/evaluation"
    run_on_folder(folder_path)

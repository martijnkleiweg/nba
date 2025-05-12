# This script evaluates the performance of a data extraction model
# Student: Martinus Kleiweg
import pandas as pd
import json
import re
from thefuzz import fuzz
from thefuzz import process

# === Configuration ===
MANUAL_CSV = "manual.csv"
EXTRACTED_JSON = "bert_extracted.json"
MISMATCH_OUTPUT = "field_errors.csv"

FIELDS = [
    "Date of birth",
    "Place of birth",
    "Listed height",
    "Listed weight",
    "College(s) or Teams",
    "Position",
    "Nationality",
    "Career highlights and awards"
]

# === Utility ===

def normalize(val: str | None) -> str:
    """
    Lower-cases and strips excess whitespace.  Any of the no value
    markers (null, none, n/a, etc.) are mapped to the empty string so
    they behave like a missing value in the evaluation.
    """
    if val is None:
        return ""

    if not isinstance(val, str):
        val = str(val)

    val = (
        val.replace("\u00a0", " ")   # non-breaking space
           .replace("–", "-")        # en-dash
           .replace("’", "'")        # curly apostrophe
           .strip()
           .lower()
    )

    # treat null-ish tokens as empty
    if val in {"null", "none", "n/a", "na", "nan", "", "not provided", "not available"}:
        return ""

    # normalise internal whitespace
    return re.sub(r"\s+", " ", val)


# fuzzy to 100 for exact match
# fuzzy to 90 for close match
def fuzzy_match(a, b, threshold=100):
    return fuzz.token_sort_ratio(normalize(a), normalize(b)) >= threshold

# === Load data ===

manual = pd.read_csv(MANUAL_CSV)
with open(EXTRACTED_JSON, "r", encoding="utf-8") as f:
    extracted_data = json.load(f)

# === Normalize extracted JSON field names ===

for item in extracted_data:
    # Fix any mismatched keys to match the gold standard exactly
    if "College(s)" in item:
        item["College(s) or Teams"] = item.pop("College(s)")

# === Index extracted by prospect name ===

extracted_index = {normalize(item["prospect"]): item for item in extracted_data if "prospect" in item}

# === Match manual rows to extracted using fuzzy name match ===

matched_extracted = {}
for _, row in manual.iterrows():
    gold_name = normalize(row["Prospect"])
    best_match, score = process.extractOne(gold_name, extracted_index.keys())
    if score >= 90:
        matched_extracted[row["Prospect"]] = extracted_index[best_match]
    else:
        matched_extracted[row["Prospect"]] = {}  # No match found

# === Evaluate ===

results = {}
mismatch_rows = []

for field in FIELDS:
    tp = fp = fn = 0
    mismatches = []

    for _, row in manual.iterrows():
        prospect  = row["Prospect"]
        gold_raw  = str(row.get(field, ""))
        pred_raw  = str(matched_extracted.get(prospect, {}).get(field, ""))
        gold_val  = normalize(gold_raw)
        pred_val  = normalize(pred_raw)

        if not gold_val and not pred_val:
            # both empty → nothing to count
            continue

        if gold_val and pred_val:
            if fuzzy_match(gold_val, pred_val):
                tp += 1
            else:
                # single-penalty: count only FP
                fp += 1
                mismatches.append((prospect, field, gold_raw, pred_raw))
        elif gold_val and not pred_val:
            # gold present, prediction empty → FN
            fn += 1
            mismatches.append((prospect, field, gold_raw, ""))
        elif pred_val and not gold_val:
            # prediction present, gold empty → FP
            fp += 1
            mismatches.append((prospect, field, "", pred_raw))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results[field] = dict(precision=precision, recall=recall, f1=f1,
                          tp=tp, fp=fp, fn=fn)

    mismatch_rows.extend(mismatches)



# === Print results ===

print("\nEvaluation Results:\n")
for field in FIELDS:
    r = results[field]
    print(f"{field:<30} | Precision: {r['precision']:.2f} | Recall: {r['recall']:.2f} | F1: {r['f1']:.2f} | TP: {r['tp']} FP: {r['fp']} FN: {r['fn']}")

macro_precision = sum(r["precision"] for r in results.values()) / len(results)
macro_recall = sum(r["recall"] for r in results.values()) / len(results)
macro_f1 = sum(r["f1"] for r in results.values()) / len(results)

print("\nMacro-Averaged Results:")
print(f"Precision: {macro_precision:.2f}")
print(f"Recall:    {macro_recall:.2f}")
print(f"F1 Score:  {macro_f1:.2f}")

# === Save mismatches to CSV ===

mismatch_df = pd.DataFrame(mismatch_rows, columns=["Prospect", "Field", "Gold", "Predicted"])
mismatch_df.to_csv(MISMATCH_OUTPUT, index=False)
print(f"\nSaved mismatches to {MISMATCH_OUTPUT}")

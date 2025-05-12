# Student: Martinus Kleiweg
"""
Create ~835 examples (picks 1–60, drafts 2009–2022) by re‐ranking
each pick’s 5 highest‐VORP candidates (with fallbacks) and marking
the top‐VORP (or composite) as gold.

Inputs (hard‐coded):
  • DRAFT_CSV:  raw draft data with Year, Round, Pick, Position, Player, NBA_VORP_3yr, etc.
  • NEEDS_CSV:  team needs per year+team with a Context field.

Outputs:
  • train_samples_vorp.jsonl
  • train_samples_vorp.csv
"""

import csv, json, random

# ──────── configuration ────────
DRAFT_CSV    = "draft_nba.csv"
NEEDS_CSV    = "teamneeds.csv"
OUTPUT_JSONL = "train_samples_vorp.jsonl"
OUTPUT_CSV   = "train_samples_vorp.csv"

START_YEAR = 2009
END_YEAR   = 2022
CANDIDATES = 5

# which stats to show + composite
STAT_LABELS = {
    "ls_pts":  "PTS",
    "ls_3p%":  "3P%",
    "ls_ast":  "AST",
    "ls_trb":  "TRB",
    "ls_stl":  "STL",
    "ls_blk":  "BLK",
    "ls_fg%":  "FG%",
    "ls_ft%":  "FT%",
    "ls_tov":  "TOV",
    "ls_g":    "G",
}
FIELDS_TO_USE = list(STAT_LABELS.keys())


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(x):
    try: return int(x)
    except: return None


def to_float(x):
    try: return float(x)
    except: return None


def get(r, *keys, default=""):
    for k in keys:
        if k in r and r[k] != "":
            return r[k]
    return default


def format_candidate(r):
    p    = get(r, "player", "Player")
    pos  = get(r, "position", "Position")
    tm   = get(r, "PickedBy", "picked by", "Team", "team")
    hgt  = get(r, "listed height", "Height")
    wgt  = get(r, "listed weight", "Weight")
    dob  = get(r, "date of birth", "DOB")
    awd  = get(r, "awards", "Awards")
    strn = get(r, "strengths", "Strengths")
    wkn  = get(r, "weaknesses", "Weaknesses")
    stats = " | ".join(f"{STAT_LABELS[f]}: {r.get(f,'N/A')}" for f in FIELDS_TO_USE)
    return (f"- {p} ({pos}, {tm}) – Height: {hgt} | Weight: {wgt} | "
            f"DOB: {dob} | Awards: {awd} | Strengths: {strn} | "
            f"Weaknesses: {wkn} | Stats: {stats}")


def build_vorp_examples(draft_rows, needs_rows):
    # build needs lookup
    needs_idx = {}
    for nr in needs_rows:
        y = to_int(get(nr, "Year", "year"))
        t = get(nr, "Team", "team")
        c = get(nr, "Context", "context")
        if y and t:
            needs_idx[(y, t)] = c

    # group picks by year
    by_year = {}
    for r in draft_rows:
        yr = to_int(get(r, "Year", "year"))
        if yr is None or yr < START_YEAR or yr > END_YEAR:
            continue
        by_year.setdefault(yr, []).append(r)

    examples = []
    for yr in range(START_YEAR, END_YEAR + 1):
        # all candidates for this year, including undrafted
        year_all = by_year.get(yr, [])

        # annotate all year candidates
        for r in year_all:
            v = to_float(get(r, "nba_vorp", "NBA_VORP_3yr"))
            r["_vorp"] = v if v is not None else None
            comp = 0.0
            for f in FIELDS_TO_USE:
                x = to_float(r.get(f, ""))
                if x is not None:
                    comp += x
            r["_comp"] = comp

        # sort by Round, Pick and take historical picks 1–60
        year_picks = sorted(
            year_all,
            key=lambda r: (
                to_int(get(r, "Round", "round")) or 99,
                to_int(get(r, "Pick", "pick"))   or 999
            )
        )[:60]

        # a master pool sorted by _vorp desc, None at end over all candidates
                # a master pool sorted by _vorp desc, None at end (include undrafted)
        pool_all = sorted(
            by_year.get(yr, []),
            key=lambda r: (r["_vorp"] is None, -(r["_vorp"] or 0.0))
        )
        used = set()
        for pick_r in year_picks:
            pos = get(pick_r, "position", "Position")
            tm  = get(pick_r, "PickedBy", "picked by", "Team", "team")
            ctx = needs_idx.get((yr, tm), "")

            # build candidate pool in stages
            stage1 = [r for r in pool_all
                      if r["_vorp"] is not None
                         and get(r, "position", "Position") == pos
                         and get(r, "player","Player") not in used]
            stage2 = [r for r in pool_all
                      if r["_vorp"] is None
                         and get(r, "position","Position") == pos
                         and get(r,"player","Player") not in used]
            stage3 = [r for r in pool_all
                      if r["_vorp"] is not None
                         and get(r,"position","Position") != pos
                         and get(r,"player","Player") not in used]
            stage4 = [r for r in pool_all
                      if r["_vorp"] is None
                         and get(r,"position","Position") != pos
                         and get(r,"player","Player") not in used]

            candidates = []
            for stage in (stage1, stage2, stage3, stage4):
                for r in stage:
                    if len(candidates) < CANDIDATES:
                        candidates.append(r)
                    else:
                        break
                if len(candidates) >= CANDIDATES:
                    break

            # if still fewer (shouldn't happen), skip
            if not candidates:
                continue

            # pick gold = highest vorp if any, else highest composite
            gold_row = max(
                candidates,
                key=lambda r: (
                    r["_vorp"] if r["_vorp"] is not None else float("-inf"),
                    r["_comp"]
                )
            )
            gold = get(gold_row, "player", "Player")

            # deterministic shuffle
            rnd = random.Random(f"{yr}-{tm}-{pos}")
            rnd.shuffle(candidates)

            # build prompt/completion
            bullets   = "\n".join(format_candidate(r) for r in candidates)
            prompt    = (
                f"Team: {tm}\n"
                f"Position of Pick: {pos}\n"
                f"Context: {ctx}\n\n"
                f"Available Players:\n{bullets}\n\n"
                "Question: Which player should they draft?"
            )
            completion = f"Pick: {gold} ({pos})"
            text = prompt + "\n### Response:\n" + completion + " <|endoftext|>"

            examples.append({"team": tm, "year": yr, "text": text})
            used.add(gold)

    return examples


def main():
    draft = read_csv(DRAFT_CSV)
    needs = read_csv(NEEDS_CSV)

    exs = build_vorp_examples(draft, needs)
    print(f"Generated {len(exs)} examples (expect ~835)")

    # write JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as outj:
        for e in exs:
            outj.write(json.dumps(e, ensure_ascii=False) + "\n")

    # write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outc:
        w = csv.DictWriter(outc, fieldnames=["team", "year", "text"])
        w.writeheader()
        for e in exs:
            w.writerow(e)

    print("Done →", OUTPUT_JSONL, "and", OUTPUT_CSV)


if __name__ == "__main__":
    main()

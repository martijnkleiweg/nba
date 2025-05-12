# This script creates training samples for a draft analysis model.
# It generates JSONL files with prompts and completions based on NBA draft data.
# Student: Martinus Kleiweg

import pandas as pd
import json
import random
from tqdm import tqdm

# === Configuration ===
DRAFT_CSV = 'draft_nba.csv'
NEEDS_CSV = 'teamneeds.csv'
OUTPUT_JSONL = 'train_samples_final.jsonl'
OUTPUT_CSV = 'train_samples_final.csv'

START_YEAR = 2009
END_YEAR = 2022
CANDIDATES_PER_PICK = 5

# === Composite Score Weights per Position ===
FIT_WEIGHTS = {
    'PG': {'ls_ast': 0.25, 'ls_stl': 0.2, 'ls_pts': 0.15, 'ls_trb': 0.1, 'ls_blk': 0.05, 'ls_tov': -0.1},
    'SG': {'ls_pts': 0.3, 'ls_3p%': 0.15, 'ls_stl': 0.2, 'ls_trb': 0.1, 'ls_tov': -0.1},
    'SF': {'ls_pts': 0.25, 'ls_trb': 0.15, 'ls_ast': 0.1, 'ls_stl': 0.15, 'ls_blk': 0.1, 'ls_tov': -0.1},
    'PF': {'ls_trb': 0.3, 'ls_pts': 0.2, 'ls_blk': 0.15, 'ls_ast': 0.05, 'ls_stl': 0.1, 'ls_tov': -0.1},
    'C':  {'ls_blk': 0.3, 'ls_trb': 0.25, 'ls_pts': 0.15, 'ls_orb': 0.1, 'ls_ft%': 0.1, 'ls_tov': -0.1},
}

# === Stats to display ===
STAT_LABELS = {
    'ls_pts': 'PTS',
    'ls_3p%': '3P%',
    'ls_ast': 'AST',
    'ls_trb': 'TRB',
    'ls_stl': 'STL',
    'ls_blk': 'BLK',
    'ls_fg%': 'FG%',
    'ls_ft%': 'FT%',
    'ls_tov': 'TOV',
    'ls_g': 'G',
    'nba_vorp': 'NBA_VORP_3yr'
}
FIELDS_TO_USE = list(STAT_LABELS.keys())

def compute_fit(row, pos):
    """Compute a composite score for a player given a position."""
    score = 0.0
    for stat, weight in FIT_WEIGHTS.get(pos, {}).items():
        if stat in row and pd.notna(row[stat]):
            score += weight * float(row[stat])
    return score

def short_text(text, max_parts=2):
    """Shorten strengths/weaknesses nicely."""
    if pd.isna(text) or not text.strip():
        return "N/A"
    parts = [s.strip() for s in text.split('•')]
    return " • ".join(parts[:max_parts])

def format_candidate(c):
    """Format a single candidate line."""
    height = c.get('listed height', 'N/A')
    weight = c.get('listed weight', 'N/A')
    dob = c.get('date of birth', 'N/A')
    awards = c.get('awards', 'N/A')
    strengths = short_text(c.get('strengths', 'N/A'))
    weaknesses = short_text(c.get('weaknesses', 'N/A'))
    team = c.get('team', 'N/A')

    stats = " | ".join(f"{STAT_LABELS[f]}: {c.get(f, 'N/A')}" for f in FIELDS_TO_USE)

    return (f"- {c['player']} ({c['position']}, {team}) – "
            f"Height: {height} | Weight: {weight} | DOB: {dob} | "
            f"Awards: {awards} | Strengths: {strengths} | Weaknesses: {weaknesses} | Stats: {stats}")

def main():
    print(f"Loading draft data from {DRAFT_CSV}...")
    draft = pd.read_csv(DRAFT_CSV)
    print(f"Loading team needs from {NEEDS_CSV}...")
    needs = pd.read_csv(NEEDS_CSV)

    if 'picked by' in draft.columns:
        draft.rename(columns={'picked by': 'PickedBy'}, inplace=True)

    if 'year' in draft.columns:
        draft.rename(columns={'year': 'Year'}, inplace=True)

    if 'nba_vorp3' in draft.columns and 'nba_vorp' not in draft.columns:
        draft['nba_vorp'] = draft['nba_vorp3']

    # Only picks 1-60 (or 58 if missing picks)
    draft = draft[(draft['Year'] >= START_YEAR) & (draft['Year'] <= END_YEAR)]
    draft = draft[pd.notna(draft['Round']) & pd.notna(draft['Pick'])]
    draft = draft[(draft['Round'] == 1) | (draft['Round'] == 2)]  # Only real picks
    draft['Picked'] = False

    draft['composite_overall'] = draft.apply(lambda r: compute_fit(r, r['position']), axis=1)

    samples = []

    print(f"Processing {len(draft)} picks...")

    for idx, pick in tqdm(draft.sort_values(['Year', 'Round', 'Pick']).iterrows(), total=len(draft)):
        year = int(pick['Year'])
        player_name = pick['player']
        position = pick['position']
        team_name = pick['PickedBy'] if 'PickedBy' in pick else pick['Team']
        college_team = pick['team']
        context_row = needs[(needs['Year'] == year) & (needs['Team'] == team_name)]
        context = context_row['Context'].values[0] if not context_row.empty else ''
        analysis = pick.get('analysis', 'N/A')

        # Create full candidate pool: all players from this year (drafted or undrafted)
        year_pool = draft[(draft['Year'] == year)].copy()

        # Exclude already picked players
        available_pool = year_pool[~year_pool['Picked']].copy()

        # Candidates matching position
        candidates = available_pool[available_pool['position'] == position].copy()

        # If not enough, fill with best remaining undrafted players
        if len(candidates) < CANDIDATES_PER_PICK:
            extra_needed = CANDIDATES_PER_PICK - len(candidates)
            fallback = available_pool.copy()
            fallback = fallback[~fallback['player'].isin(candidates['player'])]
            fallback = fallback.sort_values('composite_overall', ascending=False)
            candidates = pd.concat([candidates, fallback.head(extra_needed)])

        # If too many, remove worst composite players
        candidates['fit_score'] = candidates.apply(lambda r: compute_fit(r, position), axis=1)
        candidates = candidates.sort_values('fit_score', ascending=False)

        # Force actual pick into candidates
        if player_name not in candidates['player'].values:
            picked_player_row = year_pool[year_pool['player'] == player_name]
            if not picked_player_row.empty:
                candidates = pd.concat([candidates, picked_player_row])

        # Ensure only 5 players
        while len(candidates) > CANDIDATES_PER_PICK:
            # Don't remove actual pick
            worst_idx = candidates[candidates['player'] != player_name]['fit_score'].idxmin()
            candidates = candidates.drop(worst_idx)

        candidates = candidates.sample(frac=1, random_state=random.randint(1, 10000))  # Shuffle

        # Mark as picked
        draft.loc[draft['player'] == player_name, 'Picked'] = True

        # Format candidates
        lines = [format_candidate(c) for _, c in candidates.iterrows()]

        prompt = (f"Team: {team_name}\n"
                  f"Position of Pick: {position}\n"
                  f"Context: {context}\n\n"
                  f"Available Players:\n" + "\n".join(lines) + "\n\n"
                  f"Question: Which player should they draft, and why?\n### Response:")

        completion = f" Pick: {player_name} ({position})\nExplanation: {analysis.strip() if pd.notna(analysis) else 'N/A'}"

        samples.append({"prompt": prompt, "completion": completion})

    # Write JSONL
    print(f"Writing {len(samples)} samples to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Write CSV for inspection
    print(f"Writing {len(samples)} samples to {OUTPUT_CSV}...")
    pd.DataFrame(samples).to_csv(OUTPUT_CSV, index=False)

    print("Done!")

if __name__ == '__main__':
    main()

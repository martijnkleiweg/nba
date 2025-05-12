# This script evaluates the performance of the GPT-3.5 model on NBA draft prompts.
# Student: Martinus Kleiweg

import json, re, csv, argparse, os, tqdm, time, textwrap
from pathlib import Path
from openai import OpenAI, RateLimitError, APIError

# ─────────── configuration ────────────
MODEL = "gpt-3.5-turbo-0125"
TEMPERATURE      = 0.4
MAX_RETRIES      = 4      # back-off on rate limits
BACKOFF_SEC      = 5
SYS_PROMPT = textwrap.dedent("""
    You are an NBA draft analyst. 
    After reading the prompt, answer with exactly three bullet lines,
    each starting with 'Pick:' followed by the PLAYER NAME only.

    • First bullet = the single best pick for the team.
    • Second & third = two runner-up options, if the first is gone.
    • No explanations, stats or positions – just the names.
""").strip()

NAME_RGX = re.compile(r"pick\s*:\s*([A-Za-z'.\- ]+)", re.I)

# ─────────── helper: talk to GPT with retries ───────────
def chat_completion(client, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[{"role":"system", "content": SYS_PROMPT},
                          {"role":"user",   "content": prompt}]
            )
            return resp.choices[0].message.content
        except (RateLimitError, APIError) as e:
            wait = BACKOFF_SEC * (attempt+1)
            print(f"Rate-limit/API error – retry in {wait}s …")
            time.sleep(wait)
    raise RuntimeError("Too many OpenAI API failures – aborting")

# ─────────── main ────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="test_samples*.jsonl")
    ap.add_argument("--csv",  required=True, help="destination CSV")
    ap.add_argument("--n", type=int, default=None,
                    help="number of examples to evaluate (default: all)")
    args = ap.parse_args()

    client = OpenAI()               # uses OPENAI_API_KEY env

    # load test prompts + gold picks
    prompts = []
    with Path(args.test).open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompt, complet = rec["text"].split("### Response:", 1)
            gold = NAME_RGX.search(complet).group(1).strip().lower()
            prompts.append((prompt.strip(), gold))
            if args.n and len(prompts) >= args.n:
                break

    rows = []
    for idx, (prompt, gold) in enumerate(tqdm.tqdm(prompts, desc="GPT-3.5")):
        reply = chat_completion(client, prompt + "\n### Response:")
        names = [m.lower().strip() for m in NAME_RGX.findall(reply)]
        # ensure length 3
        names = (names + [""]*3)[:3]
        top1_ok  = (names[0] == gold)
        top3_ok  = (gold in names)
        rows.append({
            "idx": idx+1,
            "gold": gold,
            "pred_1": names[0],
            "pred_2": names[1],
            "pred_3": names[2],
            "top1_correct": int(top1_ok),
            "top3_correct": int(top3_ok)
        })

    # write CSV
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.csv).open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    t1 = sum(r["top1_correct"] for r in rows)
    t3 = sum(r["top3_correct"] for r in rows)
    total = len(rows)
    print(f"\nTop-1 accuracy: {t1}/{total} = {t1/total:.1%}")
    print(f"Top-3 accuracy: {t3}/{total} = {t3/total:.1%}")
    print(f"📄  Results saved → {args.csv}")

if __name__ == "__main__":
    main()

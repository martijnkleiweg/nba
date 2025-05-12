# This script extracts basketball player data from Wikipedia using Gemini.
# Student: Martinus Kleiweg
import os
import json
import requests
import pandas as pd
import re
import time
from bs4 import BeautifulSoup
import google.generativeai as genai  # Gemini client
import logging

# --------------------------- Config --------------------------------- #
GEMINI_API_KEY      = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL_NAME   = 'gemini-2.0-flash'

INPUT_CSV   = 'sample_prospects.csv'
OUTPUT_JSON = 'gemini_extracted_llm_only.json'
LOG_FILE    = 'extraction.log'

WIKI_USER_AGENT       = 'WikiExtractorBot/1.2 (https://github.com/yourusername/yourrepo; yourcontact@example.com)'
REQUEST_TIMEOUT_SECONDS = 20

LLM_CALL_COUNT = 0

# --------------------------- Logging -------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# --------------------- Gemini configuration ------------------------ #
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info('Gemini API configured.')
else:
    logger.warning('GOOGLE_API_KEY not set. Gemini API disabled.')

# -------------------- Wikipedia retrieval -------------------------- #
def _get_wiki_page(name: str) -> str | None:
    headers = {'User-Agent': WIKI_USER_AGENT}
    url = f'https://en.wikipedia.org/wiki/{name.replace(" ", "_")}'
    logger.info(f'Fetching URL: {url}')
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f'Error fetching {url}: {e}')
        return None

def is_basketball_page(html: str | None) -> bool:
    if not html:
        return False
    soup = BeautifulSoup(html, 'html.parser')
    if soup.find('div', id='mw-normal-catlinks') and 'basketball' in soup.find('div', id='mw-normal-catlinks').get_text().lower():
        return True
    box = soup.find('table', class_=lambda c: c and 'infobox' in c)
    if box and 'basketball' in box.get_text().lower():
        return True
    p = soup.find('div', class_='mw-parser-output').find('p')
    return bool(p and 'basketball' in p.get_text().lower())

def get_wikipedia_page(player_name: str) -> str | None:
    logger.info(f'Searching Wikipedia for "{player_name}"')
    tried = set()
    def attempt(title: str):
        key = title.lower()
        if key in tried:
            return None
        tried.add(key)
        html = _get_wiki_page(title)
        if html and is_basketball_page(html):
            logger.info(f'Found page for "{title}"')
            return html
        time.sleep(0.5)
        return None

    variants = [
        player_name,
        f"{player_name} (basketball)",
        re.sub(r'\s+\b(Jr\.?|Sr\.?|II|III|IV|V)\b', '', player_name).strip()
    ]
    for v in variants:
        if page := attempt(v):
            return page

    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"opensearch","search":player_name,"limit":5,"format":"json"},
            headers={'User-Agent': WIKI_USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        for title in r.json()[1]:
            if page := attempt(title):
                return page
    except Exception:
        pass

    logger.warning(f'No page for "{player_name}"')
    return None

# ------------------------- Gemini extraction ------------------------ #
def extract_with_gemini(player_name: str, wiki_html: str) -> dict:
    global LLM_CALL_COUNT
    LLM_CALL_COUNT += 1

    # strip references & navboxes
    soup = BeautifulSoup(wiki_html, 'html.parser')
    main = soup.find('div', class_='mw-parser-output') or soup
    for tag in main.find_all(['sup','table','div'], class_=['reference','navbox','portal','metadata']):
        tag.decompose()

    text = re.sub(r'\n{3,}', '\n\n', main.get_text('\n', strip=True))
    snippet = text[:8000]

    # **Original prompt, kept exactly as before**:
    prompt = f"""You are given a Wikipedia article about the basketball player {player_name}. Extract the following information in JSON format.

- "Date of birth": The player's date of birth (e.g., "March 28, 1998").
- "Place of birth": The city, state/province (if available), and country (e.g., "Atlanta, Georgia, U.S.").
- "Listed height": The player's listed height in feet and inches (e.g., "6 ft 5 in").
- "Listed weight": The player's listed weight in pounds (e.g., "213 lb").
- "College(s) or Teams":
    - If the player played college basketball, return all colleges with years in parentheses.
    - If no college, return the most recent team before the draft.
- "Position": The player’s position(s).
- "Nationality": Single-nationality adjective.
- "Career highlights and awards": Comma-separated list of major awards.

Return only the JSON object with these exact keys:
["Date of birth","Place of birth","Listed height","Listed weight","College(s) or Teams","Position","Nationality","Career highlights and awards"]

Wikipedia text:
{snippet}
"""

    model = genai.GenerativeModel(
        GEMINI_MODEL_NAME,
        generation_config=genai.GenerationConfig(temperature=0.1)
    )
    response = model.generate_content(prompt, request_options={'timeout':180})
    raw = response.text.strip()
    match = re.search(r'\{.*?\}', raw, re.S)
    if not match:
        logger.error(f'No JSON for {player_name}')
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f'JSON decode error for {player_name}: {e}')
        return {}
    # ensure all keys exist
    for k in ["Date of birth","Place of birth","Listed height","Listed weight",
              "College(s) or Teams","Position","Nationality","Career highlights and awards"]:
        data.setdefault(k, "")
    return data

# ----------------------------- Main loop ---------------------------- #
def process_csv(input_csv=INPUT_CSV, output_json=OUTPUT_JSON):
    df = pd.read_csv(input_csv)
    results = []

    for idx, row in df.iterrows():
        name = str(row.get('prospect','')).strip()
        if not name:
            continue
        logger.info(f'({idx+1}/{len(df)}) Annotating {name}')
        html = get_wikipedia_page(name)
        if not html:
            results.append({'prospect': name, 'status': 'No Wikipedia page'})
            continue

        data = extract_with_gemini(name, html)
        data['prospect'] = name
        data['status']   = 'Success (LLM only)' if any(data.values()) else 'Failed'
        results.append(data)
        time.sleep(1.5)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. {LLM_CALL_COUNT} LLM calls. Results saved to {output_json}")

if __name__ == '__main__':
    if not os.path.exists(INPUT_CSV):
        logger.fatal(f'Missing input CSV: {INPUT_CSV}')
    else:
        process_csv()

# This script extracts basketball player data from Wikipedia using a hybrid approach of rule-based and LLM-based extraction.
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
from datetime import datetime

# --------------------------- Config --------------------------------- #
GEMINI_API_KEY        = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL_NAME     = 'gemini-2.0-flash'
INPUT_CSV             = 'sample_prospects.csv'
OUTPUT_JSON           = 'gemini_extracted_confidence.json'
LOG_FILE              = 'extraction.log'
WIKI_USER_AGENT       = 'WikiExtractorBot/1.2 (https://github.com/yourusername/yourrepo; yourcontact@example.com)'
REQUEST_TIMEOUT_SECONDS = 20
LLM_CALL_COUNT        = 0  # Counter for Gemini API calls

# --------------------------- Logging --------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# --------------------- Gemini configuration ------------------------- #
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info('Gemini API configured.')
    except Exception as e:
        logger.error(f'Error configuring Gemini API: {e}')
        GEMINI_API_KEY = None
else:
    logger.warning('GOOGLE_API_KEY not set. Gemini API disabled.')

# ---------------- Confidence validators ----------------------------- #
VALIDATORS = {
    "Date of birth":   lambda v: bool(re.match(r"^[A-Za-z]+ \d{1,2}, \d{4}$", v)),
    "Place of birth":  lambda v: "," in v,  # at least one comma
    "Listed height":   lambda v: bool(re.match(r"^\d+\s*ft\s*\d+\s*in$", v)),
    "Listed weight":   lambda v: bool(re.match(r"^\d+\s*lb$", v)),
    "College(s) or Teams": lambda v: bool(re.search(r"\(\d{4}(?:–\d{4})?\)", v)),
    "Position":        lambda v: bool(v.strip()),
    "Nationality":     lambda v: bool(v.strip()),
    "Career highlights and awards": lambda v: bool(v.strip()),
}

# ------------------ Wikipedia retrieval ----------------------------- #
def _get_wiki_page(name: str) -> str | None:
    headers = {'User-Agent': WIKI_USER_AGENT}
    title = name.replace(' ', '_')
    url = f'https://en.wikipedia.org/wiki/{title}'
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
    # categories
    cat_div = soup.find('div', id='mw-normal-catlinks')
    if cat_div and 'basketball' in cat_div.get_text(' ', strip=True).lower():
        return True
    # infobox
    box = soup.find('table', class_=lambda c: c and 'infobox' in c)
    if box and 'basketball' in box.get_text(' ', strip=True).lower():
        return True
    # first paragraph
    p = soup.find('div', class_='mw-parser-output').find('p')
    return bool(p and 'basketball' in p.get_text(' ', strip=True).lower())

def get_wikipedia_page(player_name: str) -> str | None:
    logger.info(f'Searching Wikipedia for "{player_name}"')
    tried = set()
    def attempt(nm: str):
        key = nm.lower().strip()
        if key in tried:
            return None
        tried.add(key)
        html = _get_wiki_page(nm)
        if html and is_basketball_page(html):
            logger.info(f'Found page for "{nm}"')
            return html
        time.sleep(0.5)
        return None

    # try direct, suffix, stripped suffix
    for variant in [
        player_name,
        f"{player_name} (basketball)",
        re.sub(r'\s+\b(Jr\.?|Sr\.?|II|III|IV|V)\b', '', player_name).strip()
    ]:
        if page := attempt(variant):
            return page

    # opensearch fallback
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
    except Exception as e:
        logger.error(f'Opensearch error: {e}')

    logger.warning(f'No page for "{player_name}"')
    return None

# --------------- Rule-based infobox parsing ------------------------ #
def parse_born_field(born_text):
    m = re.match(r'^\(\s*(\d{4}-\d{2}-\d{2})\s*\).*?\(age.*?\)\s*(.+)$', born_text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m2 = re.match(r'^\(\s*(\d{4}-\d{2}-\d{2})\)\s*(.+)$', born_text)
    if m2:
        return m2.group(1).strip(), m2.group(2).strip()
    return None, None

def rule_based_extract(html: str) -> dict:
    data = {k: "" for k in VALIDATORS}
    data['Career highlights and awards'] = ""
    soup = BeautifulSoup(html, 'html.parser')
    box = soup.find('table', class_=lambda c: c and 'infobox' in c)
    if not box:
        return data

    # Born
    born_th = box.find('th', string=lambda s: s and 'Born' in s)
    if born_th and (born_td := born_th.find_next_sibling('td')):
        raw = born_td.get_text(' ', strip=True)
        iso, pob = parse_born_field(raw)
        if iso:
            try:
                dt = datetime.strptime(iso, '%Y-%m-%d')
                data['Date of birth'] = dt.strftime('%B %-d, %Y')
            except:
                data['Date of birth'] = iso
        if pob:
            data['Place of birth'] = pob.strip(' ,')

    # --- Height (imperial only) ---
    ht_tag = box.find('span', class_='height')
    if ht_tag:
        ht_text = ht_tag.get_text(strip=True)
    else:
        ht_text = ""
        if ht_row := box.find('th', string=lambda s: s and 'Height' in s):
            ht_text = ht_row.find_next_sibling('td').get_text(' ', strip=True)
    # normalize spaces (and non-breaking spaces) before matching
    ht_text = ht_text.replace('\u00a0', ' ')
    ht_text = re.sub(r'\s+', ' ', ht_text)
    # look specifically for “X ft Y in”
    if m := re.search(r'(\d+)\s*ft\s*(\d+)\s*in', ht_text):
        data['Listed height'] = f"{m.group(1)} ft {m.group(2)} in"

    # --- Weight (imperial only) ---
    wt_tag = box.find('span', class_='weight')
    if wt_tag:
        wt_text = wt_tag.get_text(strip=True)
    else:
        wt_text = ""
        if wt_row := box.find('th', string=lambda s: s and 'Weight' in s):
            wt_text = wt_row.find_next_sibling('td').get_text(' ', strip=True)
    # normalize before matching
    wt_text = wt_text.replace('\u00a0', ' ')
    wt_text = re.sub(r'\s+', ' ', wt_text)
    # look specifically for “N lb”
    if m2 := re.search(r'(\d+)\s*lb', wt_text):
        data['Listed weight'] = f"{m2.group(1)} lb"

    # College(s) or Teams
    def extract_list(label):
        th = box.find('th', string=lambda s: s and label in s)
        if th and (td := th.find_next_sibling('td')):
            items = td.find_all('li')
            if items:
                return ', '.join(li.get_text(' ', strip=True) for li in items)
            return td.get_text(', ', strip=True)
        return ""
    data['College(s) or Teams'] = extract_list('College') or extract_list('Team')

    # Position & Nationality
    for key in ('Position','Nationality'):
        th = box.find('th', string=lambda s: s and key in s)
        if th and (td := th.find_next_sibling('td')):
            data[key] = td.get_text(' ', strip=True)

    # Career highlights and awards
    head = box.find('th', string=lambda s: s and 'Career highlights' in s)
    if head and (tr := head.find_parent('tr')) and (nxt := tr.find_next_sibling('tr')):
        if (ul := nxt.find('ul')):
            data['Career highlights and awards'] = ', '.join(
                li.get_text(' ', strip=True) for li in ul.find_all('li')
            )
        else:
            data['Career highlights and awards'] = nxt.get_text(', ', strip=True)

    return data

# ------------------------- Gemini extraction ------------------------ #
def extract_with_gemini(player_name: str, wiki_html: str) -> dict:
    global LLM_CALL_COUNT
    LLM_CALL_COUNT += 1
    model = genai.GenerativeModel(GEMINI_MODEL_NAME,
                                  generation_config=genai.GenerationConfig(temperature=0.1))
    soup = BeautifulSoup(wiki_html, 'html.parser')
    main = soup.find('div', class_='mw-parser-output') or soup
    for tag in main.find_all(['sup','table','div'], class_=['reference','navbox','portal','metadata']):
        tag.decompose()
    text = re.sub(r'\n{3,}', '\n\n', main.get_text('\n', strip=True))
    snippet = text[:8000]

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
    for k in VALIDATORS:
        data.setdefault(k, "")
    data.setdefault('Career highlights and awards', "")
    return data

# ----------------------------- Main loop ---------------------------- #
def process_csv(input_csv: str = INPUT_CSV, output_json: str = OUTPUT_JSON):
    df = pd.read_csv(input_csv)
    results = []
    core_keys = list(VALIDATORS.keys())

    for idx, row in df.iterrows():
        name = str(row.get('prospect','')).strip()
        if not name:
            continue
        logger.info(f'Processing {name} ({idx+1}/{len(df)})')

        html = get_wikipedia_page(name)
        if not html:
            results.append({'prospect': name, 'status': 'No Wikipedia page'})
            continue

        rule_data = rule_based_extract(html)

        # 1) check non‐empty & validator pass
        needs_llm = False
        for k in core_keys:
            val = rule_data.get(k, "")
            if not val or not VALIDATORS[k](val):
                needs_llm = True
                break

        if not needs_llm:
            rule_data['prospect'] = name
            rule_data['status']   = 'Success (rule-based)'
            results.append(rule_data)
        else:
            llm_data = extract_with_gemini(name, html)
            merged = {k: rule_data.get(k) or llm_data.get(k, "") for k in core_keys + ['Career highlights and awards']}
            merged['prospect'] = name
            merged['status']   = 'Success (hybrid)'
            results.append(merged)

        time.sleep(2)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f'Done. Saved to {output_json}. LLM used {LLM_CALL_COUNT} times.')

if __name__ == '__main__':
    if not os.path.exists(INPUT_CSV):
        logger.fatal(f'Missing input CSV: {INPUT_CSV}')
    else:
        process_csv()

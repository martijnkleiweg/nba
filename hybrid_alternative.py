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

# --------------------------- Config --------------------------------- #
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL_NAME = 'gemini-2.0-flash'

INPUT_CSV = 'sample_prospects.csv'
OUTPUT_JSON = 'gemini_extracted_data2.json'
LOG_FILE = 'extraction.log'

WIKI_USER_AGENT = (
    'WikiExtractorBot/1.2 '
    '(https://github.com/yourusername/yourrepo; yourcontact@example.com)'
)
REQUEST_TIMEOUT_SECONDS = 20

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
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info('Gemini API configured.')
    except Exception as e:
        logger.error(f'Error configuring Gemini API: {e}')
        GEMINI_API_KEY = None
else:
    logger.warning('GOOGLE_API_KEY not set. Gemini API disabled.')

# -------------------- Wikipedia retrieval -------------------------- #
def _get_wiki_page(name: str) -> str | None:
    headers = {'User-Agent': WIKI_USER_AGENT}
    title = name.replace(' ', '_')
    url = f'https://en.wikipedia.org/wiki/{title}'
    logger.info(f'Fetching URL: {url}')
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        if '<meta http-equiv="refresh"' in resp.text:
            logger.warning('Meta-refresh redirect detected.')
        return resp.text
    except Exception as e:
        logger.error(f'Error fetching {url}: {e}')
    return None

def is_basketball_page(html: str | None) -> bool:
    if not html:
        return False
    try:
        soup = BeautifulSoup(html, 'lxml')
    except:
        soup = BeautifulSoup(html, 'html.parser')
    # check categories
    cat_div = soup.find('div', id='mw-normal-catlinks')
    if cat_div:
        for a in cat_div.find_all('a'):
            if 'basketball' in a.get_text(strip=True).lower():
                return True
    # check infobox
    infobox = soup.find('table', class_=lambda c: c and 'infobox' in c)
    if infobox and 'basketball' in infobox.get_text(' ', strip=True).lower():
        return True
    # fallback to paragraphs
    div = soup.find('div', class_='mw-parser-output')
    if div:
        p = div.find('p')
        if p and 'basketball' in p.get_text(' ', strip=True).lower():
            return True
    return False

def get_wikipedia_page(player_name: str) -> str | None:
    logger.info(f'Searching Wikipedia for "{player_name}"')
    checked = set()
    def attempt(name_var: str):
        key = name_var.lower().strip()
        if key in checked:
            return None
        checked.add(key)
        html = _get_wiki_page(name_var)
        if html and is_basketball_page(html):
            logger.info(f'Found page for "{name_var}"')
            return html
        time.sleep(0.5)
        return None

    # 1) direct
    if res := attempt(player_name):
        return res
    # 2) suffix
    if res := attempt(f"{player_name} (basketball)"):
        return res
    # 3) strip suffixes
    no_suffix = re.sub(r'\s+\b(II|III|IV|V|Jr\.?|Sr\.?)+$', '', player_name).strip()
    if no_suffix.lower() != player_name.lower():
        if res := attempt(no_suffix):
            return res
        if res := attempt(f"{no_suffix} (basketball)"):
            return res
    # 4) comma form
    comma = re.sub(r'^([^,]+),\s*(Jr\.?|Sr\.?|II|III|IV|V)$', r'\1 \2', player_name).strip()
    if comma.lower() != player_name.lower():
        if res := attempt(comma):
            return res
        if res := attempt(f"{comma} (basketball)"):
            return res
    # 5) opensearch
    logger.info("Using opensearch fallback...")
    try:
        params = {"action":"opensearch","search":player_name,"limit":10,"format":"json"}
        r = requests.get("https://en.wikipedia.org/w/api.php", params=params, headers={'User-Agent':WIKI_USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        titles, urls = data[1], data[3]
        for title, url in zip(titles, urls):
            if res := attempt(title):
                return res
    except Exception as e:
        logger.error(f'Opensearch error: {e}')
    logger.warning(f'No page for "{player_name}"')
    return None

# ------------------ Rule-based infobox parsing --------------------- #
def extract_infobox(html: str) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    box = soup.find('table', class_=lambda c: c and 'infobox' in c)
    info = {}
    if not box:
        return info
    for tr in box.find_all('tr'):
        if tr.th and tr.td:
            k = tr.th.get_text(' ', strip=True)
            v = tr.td.get_text(' ', strip=True)
            info[k] = v
    return info

def rule_based_extract(html: str) -> dict:
    """
    Attempt rule-based extraction of the target fields from a Wikipedia infobox.
    Returns a dict with keys:
      'Date of birth', 'Place of birth',
      'Listed height', 'Listed weight',
      'College(s) or Teams', 'Position', 'Nationality',
      'Career highlights and awards'
    """
    from datetime import datetime

    # Initialize output with empty strings
    data = {
        'Date of birth': '',
        'Place of birth': '',
        'Listed height': '',
        'Listed weight': '',
        'College(s) or Teams': '',
        'Position': '',
        'Nationality': '',
        'Career highlights and awards': ''
    }

    soup = BeautifulSoup(html, 'html.parser')
    box = soup.find('table', class_=lambda c: c and 'infobox' in c)
    if not box:
        return data

    # --- Born field parsing ---
    born_th = box.find('th', string=lambda s: s and 'Born' in s)
    if born_th:
        born_td = born_th.find_next_sibling('td')
        if born_td:
            # 1) Date of birth from <span class="bday">
            bday_tag = born_td.find('span', class_='bday')
            if bday_tag:
                iso = bday_tag.get_text(strip=True)
                try:
                    dt = datetime.strptime(iso, '%Y-%m-%d')
                    data['Date of birth'] = dt.strftime('%B %-d, %Y')
                except ValueError:
                    data['Date of birth'] = iso

            # 2) Place-of-birth via <a> tags
            place = ""
            bday_tag = born_td.find('span', class_='bday')
            if bday_tag:
                # collect all <a> siblings that come *after* the date span
                for sib in bday_tag.next_siblings:
                    if getattr(sib, 'name', None) == 'a' and sib.get('href','').startswith('/wiki/'):
                        place = place + sib.get_text(strip=True) + ", "
            # if no links found, fall back to your cleaned text
            if not place:
                raw = born_td.get_text(' ', strip=True)
                raw = re.sub(r'^\(?\d{4}-\d{2}-\d{2}\)?\s*', '', raw)    # drop ISO date
                raw = re.sub(r'\[\d+\]', '', raw)                       # drop footnotes
                raw = re.sub(r'\([^)]*\)', '', raw)                     # strip any other parentheses
                place = raw
            data['Place of birth'] = place.strip(' ,')

    # --- Height & Weight ---
    # Height
    ht_tag = box.find('span', class_='height')
    if ht_tag:
        data['Listed height'] = ht_tag.get_text(strip=True)
    else:
        ht_th = box.find('th', string=lambda s: s and 'Height' in s)
        if ht_th:
            ht_td = ht_th.find_next_sibling('td')
            if ht_td:
                data['Listed height'] = ht_td.get_text(' ', strip=True).split('(')[0].strip()

    # Weight
    wt_tag = box.find('span', class_='weight')
    if wt_tag:
        data['Listed weight'] = wt_tag.get_text(strip=True)
    else:
        wt_th = box.find('th', string=lambda s: s and 'Weight' in s)
        if wt_th:
            wt_td = wt_th.find_next_sibling('td')
            if wt_td:
                data['Listed weight'] = wt_td.get_text(' ', strip=True).split('(')[0].strip()

    # --- College(s) or Teams helper ---
    def extract_list_field(label: str) -> str:
        th = box.find('th', string=lambda s: s and label in s)
        if not th:
            return ''
        td = th.find_next_sibling('td')
        if not td:
            return ''
        ul = td.find('ul')
        if ul:
            return ', '.join(li.get_text(' ', strip=True) for li in ul.find_all('li'))
        parts = [p.strip() for p in td.get_text(', ', strip=True).split(',') if p.strip()]
        return ', '.join(parts)

    data['College(s) or Teams'] = extract_list_field('College') or extract_list_field('Teams')

    # --- Position & Nationality ---
    pos_th = box.find('th', string=lambda s: s and 'Position' in s)
    if pos_th:
        pos_td = pos_th.find_next_sibling('td')
        if pos_td:
            data['Position'] = pos_td.get_text(' ', strip=True)

    nat_th = box.find('th', string=lambda s: s and 'Nationality' in s)
    if nat_th:
        nat_td = nat_th.find_next_sibling('td')
        if nat_td:
            data['Nationality'] = nat_td.get_text(' ', strip=True)

    # --- Career highlights and awards ---
    head = box.find('th', string=lambda s: s and 'Career highlights' in s)
    if head:
        tr = head.find_parent('tr')
        if tr:
            nxt = tr.find_next_sibling('tr')
            if nxt:
                ul = nxt.find('ul')
                if ul:
                    data['Career highlights and awards'] = ', '.join(
                        li.get_text(' ', strip=True) for li in ul.find_all('li')
                    )
                else:
                    data['Career highlights and awards'] = nxt.get_text(', ', strip=True)

    return data





# ------------------------- Gemini extraction ------------------------ #
def extract_with_gemini(player_name: str, wiki_html: str) -> dict | None:
    if not GEMINI_API_KEY:
        return None
    model = genai.GenerativeModel(GEMINI_MODEL_NAME,
                                  generation_config=genai.GenerationConfig(temperature=0.1))
    soup = BeautifulSoup(wiki_html, 'html.parser')
    main = soup.find('div', class_='mw-parser-output') or soup
    for tag in main.find_all(['sup','table','div'], class_=['reference','navbox','portal','metadata']):
        tag.decompose()
    text = main.get_text('\n', strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)
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
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f'JSON decode error for {player_name}: {e}')
        return None
    for k in ['Date of birth','Place of birth','Listed height','Listed weight',
              'College(s) or Teams','Position','Nationality','Career highlights and awards']:
        data.setdefault(k,'')
    return data

# ----------------------------- Main loop ---------------------------- #
def process_csv_gemini(input_csv: str = INPUT_CSV, output_json: str = OUTPUT_JSON):
    if not GEMINI_API_KEY:
        logger.fatal('GOOGLE_API_KEY not set.')
        return
    df = pd.read_csv(input_csv)
    results = []
    core_keys = ['Date of birth','Place of birth','Listed height','Listed weight',
                 'College(s) or Teams','Position','Nationality']

    for idx, row in df.iterrows():
        name = str(row.get('prospect','')).strip()
        if not name:
            continue
        logger.info(f'Processing {name} ({idx+1}/{len(df)})')
        html = get_wikipedia_page(name)
        if not html:
            results.append({'prospect':name,'status':'No Wikipedia page'})
            continue

        rule_data = rule_based_extract(html)
        if all(rule_data.get(k) for k in core_keys):
            rule_data['prospect'] = name
            rule_data['status'] = 'Success (rule-based)'
            results.append(rule_data)
        else:
            llm_data = extract_with_gemini(name, html) or {}
            merged = {k: rule_data.get(k) or llm_data.get(k,'') for k in core_keys + ['Career highlights and awards']}
            merged['prospect'] = name
            merged['status'] = 'Success (hybrid)'
            results.append(merged)

        time.sleep(2)

    with open(output_json,'w',encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f'Results saved to {output_json}')

if __name__ == '__main__':
    if not os.path.exists(INPUT_CSV):
        logger.fatal(f'Input file {INPUT_CSV} not found.')
    else:
        process_csv_gemini()
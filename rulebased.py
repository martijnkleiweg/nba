# This script extracts basketball player data from Wikipedia using a rule-based extraction.
# Student: Martinus Kleiweg
import requests
import pandas as pd
import re
import time
import unicodedata
import json
import string
from bs4 import BeautifulSoup

# Normalize and slugify names
def normalize_name(name):
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    translator = str.maketrans('', '', string.punctuation)
    return re.sub(r'[^a-zA-Z0-9\s]', '', name).strip().lower()

# Original methods from your provided code
def _get_wiki_page(name):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
    resp = requests.get(url, headers=headers)
    return resp.text if resp.status_code == 200 else None

def is_basketball_page(html):
    soup = BeautifulSoup(html, "html.parser")
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    content_div = soup.find("div", {"class": "mw-parser-output"})
    paragraphs = content_div.find_all("p", recursive=False) if content_div else []
    for p in paragraphs[:3]:  # Check the first 3 paragraphs
        if "basketball" in p.get_text(strip=True).lower():
            return True
    return bool(infobox)

def get_wikipedia_page(player_name):
    html = _get_wiki_page(player_name)
    if html and is_basketball_page(html):
        return html

    new_name = re.sub(r"\s+(II|III|IV)$", "", player_name)
    for suffix in ["", " (basketball)"]:
        html = _get_wiki_page(new_name + suffix)
        if html and is_basketball_page(html):
            return html

    if player_name.lower() == "alex sarr":
        html = _get_wiki_page("Alexandre Sarr")
        if html and is_basketball_page(html):
            return html

    # Use Wikipedia API search as ultimate fallback
    headers = {'User-Agent': 'Mozilla/5.0'}
    params = {"action": "opensearch", "search": player_name, "limit": 5, "namespace": 0, "format": "json"}
    response = requests.get("https://en.wikipedia.org/w/api.php", headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        for url in data[3]:
            page_resp = requests.get(url, headers=headers)
            if page_resp.status_code == 200 and is_basketball_page(page_resp.text):
                return page_resp.text

    print(f"Could not find Wikipedia basketball page for {player_name}")
    return None

def extract_infobox(html):
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", class_=lambda c: c and "infobox" in c)
    data = {}
    if tbl:
        for row in tbl.find_all("tr"):
            if row.th and row.td:
                data[row.th.get_text(" ",strip=True)] = row.td.get_text(" ",strip=True)
    return data

def parse_born_field(born_text):
    match = re.search(r'\((\d{4}-\d{2}-\d{2})\)', born_text)
    dob = match.group(1) if match else None
    pob = born_text.split(')')[-1].strip() if ')' in born_text else None
    return dob, pob

def remove_metric_conversion(val):
    return val.split('(')[0].strip()

def clean_infobox_height_weight(info):
    for k in ["Listed height", "Listed weight"]:
        if k in info:
            info[k] = remove_metric_conversion(info[k])

def extract_awards(html):
    soup = BeautifulSoup(html, "html.parser")
    head = soup.find(lambda tag: tag.name=="th" and "Career highlights and awards" in tag.get_text())
    if head:
        nxt = head.find_parent("tr").find_next_sibling("tr")
        if nxt and nxt.find("ul"):
            return ", ".join(li.get_text(" ",strip=True) for li in nxt.find("ul").find_all("li"))
    return ""

# regex fallback extraction for nationality and other fields
def regex_fallback(html, field):
    soup = BeautifulSoup(html, "html.parser")
    text = ' '.join(p.get_text(" ", strip=True) for p in soup.select('div.mw-parser-output > p')[:5])

    patterns = {
        "Full name": r"(?:full name|born as|birth name) is ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
        "Date of birth": r"born.*?(\d{1,2}\s\w+\s\d{4}|\w+\s\d{1,2},\s\d{4})",
        "Place of birth": r"born.*?in\s([\w\s,]+)",
        "Listed height": r"(\d\.\d{2}\s?m|\d\sft\s\d{1,2}\sin)",
        "Listed weight": r"(\d{2,3}\s?(?:kg|lb))",
        "College(s)": r"college basketball.*?at ([A-Za-z\s]+)",
        "Career teams": r"plays? for ([A-Za-z\s]+)",
        "Position": r"(point guard|shooting guard|small forward|power forward|center)",
        "Nationality": r"is an? ([A-Za-z]+) (?:professional )?basketball player"
}


    pattern = patterns.get(field)
    if pattern:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    return None

# Main extraction function
def extract_player(player, college=None):
    html = get_wikipedia_page(player)
    if not html:
        return {"prospect": player, "error": "Page not found"}

    info = extract_infobox(html)
    clean_infobox_height_weight(info)

    dob, pob = parse_born_field(info.get("Born", "")) if "Born" in info else (None, None)

    fields = ["Full name", "Date of birth", "Place of birth", "Listed height",
              "Listed weight", "College(s)", "Position", "Nationality"]

    data = {
        "prospect": player,
        "Full name": info.get("Name") or regex_fallback(html, "Full name") or player,
        "Date of birth": dob or regex_fallback(html, "Date of birth"),
        "Place of birth": pob or regex_fallback(html, "Place of birth"),
        "Listed height": info.get("Listed height") or regex_fallback(html, "Listed height"),
        "Listed weight": info.get("Listed weight") or regex_fallback(html, "Listed weight"),
        "College(s)": info.get("College") or college or regex_fallback(html, "College(s)") or regex_fallback(html, "Career teams"),
        "Position": info.get("Position") or regex_fallback(html, "Position"),
        "Nationality": info.get("Nationality") or regex_fallback(html, "Nationality"),
        "Career highlights and awards": extract_awards(html) or regex_fallback(html, "Career highlights and awards")
    }

    return data

# Process CSV file
def process_csv(input_csv, output_json="extracted_players.json"):
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        print(f"Processing {row['prospect']}...")
        result = extract_player(row['prospect'], row.get('college'))
        results.append(result)
        time.sleep(2)  # Avoid rate limiting
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Extraction saved to {output_json}")

if __name__ == "__main__":
    process_csv("sample_prospects.csv")

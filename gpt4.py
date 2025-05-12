# This script extracts basketball player data from Wikipedia using GPT-4o.
# Student: Martinus Kleiweg
import os
import re
import time
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
import openai
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Wiki Page
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
    for p in paragraphs[:3]:
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

# Use GPT to extract structured data from page
def extract_with_llm(player_name, wiki_html):
    from bs4 import BeautifulSoup
    import openai
    import json

    # Extract plain text from HTML
    soup = BeautifulSoup(wiki_html, "html.parser")
    text = soup.get_text(" ", strip=True)

    # Inline prompt with detailed instruction
    prompt = f"""
You are given a Wikipedia article about the basketball player {player_name}. Extract the following information in JSON format.

- "Date of birth": The player's date of birth (e.g., "March 28, 1998").
- "Place of birth": The city, state/province (if available), and country (e.g., "Atlanta, Georgia, U.S.").
- "Listed height": The player's listed height in feet and inches (e.g., "6 ft 5 in").
- "Listed weight": The player's listed weight in pounds (e.g., "213 lb").
- "College(s) or Teams": 
    - If the player played college basketball, return all colleges with years in parentheses (e.g., "Florida State (2017–2021)").
    - If no college, return the most recent team they played for before entering the NBA draft (e.g., "Metropolitans 92").
- "Position": The player’s position(s), like "Small forward / shooting guard".
- "Nationality": Nationality as a single adjective, e.g., "American", "French".
- "Career highlights and awards": A single comma-separated list of major awards and honors (e.g., "NZNBL champion (2024), Second-team All-ACC (2021), McDonald's All-American (2017)").

Return only the JSON, no explanation. The keys must match exactly:

["Date of birth", "Place of birth", "Listed height", "Listed weight", "College(s) or Teams", "Position", "Nationality", "Career highlights and awards"]

Wikipedia text:
{text[:8000]}
"""

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing response for {player_name}: {e}")
        return None


# Main extraction function
def extract_player(player, college=None):
    html = get_wikipedia_page(player)
    if not html:
        return {"prospect": player, "error": "No Wikipedia page"}
    extracted = extract_with_llm(player, html)
    if not extracted:
        return {"prospect": player, "error": "LLM extraction failed"}
    extracted["prospect"] = player
    return extracted

# Batch processing
def process_csv(input_csv, output_json="extracted_players_llm.json"):
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        name = row["prospect"]
        print(f"Extracting {name}...")
        data = extract_player(name)
        results.append(data)
        time.sleep(2)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved LLM-extracted data to {output_json}")

if __name__ == "__main__":
    process_csv("sample_prospects.csv")

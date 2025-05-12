# This script extracts basketball player data from Wikipedia using a BERT NER model.
# Student: Martinus Kleiweg
import os
import json
import requests
import pandas as pd
import re
import time
import logging
from bs4 import BeautifulSoup
from datetime import datetime
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

# ------------------ Configuration ------------------ #
MODEL_DIR = "ner_model/final"
INPUT_CSV = "sample_prospects.csv"
OUTPUT_JSON = "bert_extracted_hybrid2.json"
WIKI_USER_AGENT = 'WikiExtractorBot/1.2 (https://github.com/yourusername/yourrepo; yourcontact@example.com)'
REQUEST_TIMEOUT_SECONDS = 20

# ------------------ Logging ------------------ #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------ Load Model ------------------ #
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForTokenClassification.from_pretrained(MODEL_DIR)
id2label = model.config.id2label
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ------------------ Wikipedia Fetching ------------------ #
def get_wikipedia_page(name: str) -> str | None:
    headers = {'User-Agent': WIKI_USER_AGENT}
    url = f'https://en.wikipedia.org/wiki/{name.replace(" ", "_")}'
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f"Error fetching Wikipedia page for {name}: {e}")
        return None

def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(["table", "sup", "style", "script"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)[:8000]

# ------------------ Rule-based for height/weight ------------------ #
def rule_based_height_weight(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    box = soup.find('table', class_=lambda c: c and 'infobox' in c)
    height, weight = '', ''
    if not box:
        return height, weight

    ht_tag = box.find('span', class_='height')
    if ht_tag:
        height = ht_tag.get_text(strip=True)
    else:
        ht_row = box.find('th', string=lambda s: s and 'Height' in s)
        if ht_row:
            td = ht_row.find_next_sibling('td')
            if td:
                height = td.get_text(' ', strip=True)

    wt_tag = box.find('span', class_='weight')
    if wt_tag:
        weight = wt_tag.get_text(strip=True)
    else:
        wt_row = box.find('th', string=lambda s: s and 'Weight' in s)
        if wt_row:
            td = wt_row.find_next_sibling('td')
            if td:
                weight = td.get_text(' ', strip=True)

    return height, weight

# ------------------ BERT Entity Extraction ------------------ #
def extract_bert_entities(text: str):
    inputs = tokenizer(text, return_offsets_mapping=True, truncation=True, return_tensors="pt", max_length=512)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()

    spans = []
    current = None

    for i, label_id in enumerate(predictions):
        if i >= len(offset_mapping):
            continue
        start, end = offset_mapping[i]
        if start == end:
            continue
        label = id2label[label_id]
        if label == "O":
            if current:
                spans.append(current)
                current = None
            continue
        prefix, tag = label.split("-")
        if prefix == "B":
            if current:
                spans.append(current)
            current = {"start": start, "end": end, "label": tag}
        elif prefix == "I" and current and current["label"] == tag:
            current["end"] = end
        else:
            if current:
                spans.append(current)
            current = None
    if current:
        spans.append(current)

    return spans

def spans_to_field_dict(text: str, spans: list[dict]) -> dict:
    out = {
        "Date of birth": "",
        "Place of birth": "",
        "College(s) or Teams": "",
        "Position": "",
        "Nationality": "",
        "Career highlights and awards": ""
    }
    span_texts = {}
    for span in spans:
        label = span["label"]
        extracted = text[span["start"]:span["end"]].strip()
        if label not in span_texts:
            span_texts[label] = []
        if extracted not in span_texts[label]:
            span_texts[label].append(extracted)

    for key in out:
        tag = key.upper().replace(" ", "_").replace("-", "_")
        parts = span_texts.get(tag, [])
        out[key] = ', '.join(parts)

    return out

# ------------------ Main Loop ------------------ #
def process_csv(input_csv=INPUT_CSV, output_json=OUTPUT_JSON):
    df = pd.read_csv(input_csv)
    results = []

    for idx, row in df.iterrows():
        name = str(row.get('prospect')).strip()
        if not name:
            continue
        logger.info(f"Processing {name} ({idx+1}/{len(df)})")
        html = get_wikipedia_page(name)
        if not html:
            results.append({'prospect': name, 'status': 'No Wikipedia page'})
            continue
        text = extract_main_text(html)
        spans = extract_bert_entities(text)
        field_data = spans_to_field_dict(text, spans)
        height, weight = rule_based_height_weight(html)

        final = {
            "Date of birth": field_data["Date of birth"],
            "Place of birth": field_data["Place of birth"],
            "Listed height": height,
            "Listed weight": weight,
            "College(s) or Teams": field_data["College(s) or Teams"],
            "Position": field_data["Position"],
            "Nationality": field_data["Nationality"],
            "Career highlights and awards": field_data["Career highlights and awards"],
            "prospect": name,
            "status": "Success (hybrid)"
        }

        results.append(final)
        time.sleep(1.5)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Done. Saved to {output_json}")

# ------------------ Run ------------------ #
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        logger.fatal(f"Missing input file: {INPUT_CSV}")
    else:
        process_csv()

# Wikipedia NBA Draft Prospect Extractor

This project implements multiple approaches for extracting structured information about NBA draft prospects from Wikipedia pages. The methods include rule-based parsing, fine-tuned BERT NER, GPT-4 completion extraction, Gemini API usage, and hybrid pipelines combining these methods. Evaluation scripts are provided to benchmark each approach.

## Project Structure

- `rulebased.py` – Extracts data using regex and HTML rules.
- `bert.py` – Uses a fine-tuned BERT token classification model to extract information.
- `gpt4.py` – Calls GPT-4 via OpenAI API to perform extraction.
- `gemini.py` – Uses Google Gemini API for structured extraction.
- `hybrid.py` – Combines rule-based and Gemini methods, calling Gemini only when necessary.
- `hybrid_alternative.py` – An alternative hybrid strategy with slightly different fallback logic.
- `evaluate_wiki.py` – Loads the gold-standard and extracted results, and computes macro-F1, precision, and recall across fields.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/martijnkleiweg/wiki-nba-draft-extraction.git
cd wiki-nba-draft-extraction
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## API Keys

To use GPT-4 or Gemini models, you must set your API keys:

### OpenAI (for GPT-4)

```bash
export OPENAI_API_KEY=your_openai_key_here
```

### Gemini (Google Generative AI)

```bash
export GOOGLE_API_KEY=your_gemini_key_here
```

## Running the Extraction

Run individual methods:

```bash
python rulebased.py
python bert.py
python gpt4.py
python gemini.py
```

Run hybrid methods:

```bash
python hybrid.py
python hybrid_alternative.py
```

Evaluate output against the gold standard:

```bash
python evaluate_wiki.py
```

Make sure each extractor saves its results in the expected location (`outputs/rulebased.jsonl`, etc.) for `evaluate_wiki.py` to pick up.

## Output Format

All extractors produce a `.jsonl` file with one record per player, including the extracted fields:

```json
{
  "Prospect": "Victor Wembanyama",
  "DOB": "2004-01-04",
  "Listed height": "7 ft 4 in",
  ...
}
```

## Evaluation Metrics

The `evaluate_wiki.py` script computes:
- Precision, recall, and F1-score per field
- Macro-averaged F1 across all fields
- Logging of missing or misclassified fields

## License

MIT License. See `LICENSE` for details.

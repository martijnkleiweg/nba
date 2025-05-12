import requests
import pandas as pd
import re
import time
import unicodedata
import string
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
import urllib.parse
import psycopg2

SKIP_IF_NO_WIKI = True

ALT_STATS_SLUGS = {
    "ron holland": "hollaro01d",
    "alex sarr": "alexandre-sarr-1",
    "matas buzelis": "buzelma01d"
}

###############################################################################
# 1) A Simple Rate Limiter to Avoid 429 Errors
###############################################################################

class SRRateLimiter:
    """
    A rolling-window rate limiter allowing up to `max_requests` (default=20)
    in any `window_secs` (default=60) second period.
    If we exceed that, we sleep until we are under the threshold.
    """
    def __init__(self, max_requests=20, window_secs=60):
        self.max_requests = max_requests
        self.window_secs = window_secs
        self.request_times = []  # store timestamps (float time.time())

    def wait_if_needed(self):
        """
        Called before every request. If we have used up our quota of
        requests in the last window_secs, we sleep until we can proceed.
        """
        now = time.time()

        # remove timestamps older than window_secs
        cutoff = now - self.window_secs
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.pop(0)

        # if we already have self.max_requests in the last window_secs, wait
        while len(self.request_times) >= self.max_requests:
            earliest = self.request_times[0]
            wait_time = (earliest + self.window_secs) - now
            wait_time = max(wait_time, 0)
            print(f"[RateLimiter] Hit {self.max_requests} requests in {self.window_secs}s. Sleeping {wait_time:.2f}s...")
            time.sleep(wait_time)

            now = time.time()
            cutoff = now - self.window_secs
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.pop(0)

        # now we can proceed
        self.request_times.append(time.time())

# We'll create one global rate limiter
rate_limiter = SRRateLimiter(max_requests=20, window_secs=60)

###############################################################################
# 2) Basic Setup
###############################################################################

def normalize_name(name):
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    translator = str.maketrans('', '', string.punctuation)
    name = name.translate(translator)
    return re.sub(r'[^a-zA-Z0-9\s]', '', name).strip().lower()

def slugify_player_name(player_name):
    simple_name = normalize_name(player_name)
    return "-".join(simple_name.split())

def build_gleague_slug(full_name: str) -> str:
    parts = full_name.lower().split()
    if len(parts) < 2:
        last = parts[-1] if parts else ''
        first = 'x'
    else:
        first = parts[0]
        last = parts[-1]
    last = re.sub(r'[^a-z]', '', last)
    first = re.sub(r'[^a-z]', '', first)
    return last[:5] + first[:2] + "01d"

def get_alt_slug(player_name):
    norm = normalize_name(player_name)
    if norm in ALT_STATS_SLUGS:
        return ALT_STATS_SLUGS[norm]
    return build_gleague_slug(player_name)


###############################################################################
# 3) Wikipedia Functions
###############################################################################

def _get_wiki_page(name):
    headers = {'User-Agent': 'Mozilla/5.0'}
    title = name.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{title}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    return None

def is_basketball_page(html):
    soup = BeautifulSoup(html, "html.parser")
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    first_paragraph = None
    content_div = soup.find("div", {"class": "mw-parser-output"})
    if content_div:
        for p in content_div.find_all("p", recursive=False):
            if p.get_text(strip=True):
                first_paragraph = p.get_text(strip=True)
                break
    if infobox and first_paragraph and "basketball" in first_paragraph.lower():
        return True
    return False

def get_wikipedia_page(player_name):
    html = _get_wiki_page(player_name)
    if html and is_basketball_page(html):
        return html
    # Fallback: remove trailing roman numerals
    new_name = re.sub(r"\s+(II|III|IV)$", "", player_name)
    if new_name != player_name:
        html = _get_wiki_page(new_name)
        if html and is_basketball_page(html):
            return html
        # Also try "... (basketball)"
        html = _get_wiki_page(new_name + " (basketball)")
        if html and is_basketball_page(html):
            return html
    if player_name.lower() == "alex sarr":
        html = _get_wiki_page("Alexandre Sarr")
        if html and is_basketball_page(html):
            return html
    # Fallback: opensearch
    headers = {'User-Agent': 'Mozilla/5.0'}
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "search": player_name,
        "limit": 5,
        "namespace": 0,
        "format": "json"
    }
    search_response = requests.get(search_url, headers=headers, params=params)
    if search_response.status_code == 200:
        data = search_response.json()
        # Each search result is typically [searchTerm, titles[], desc[], urls[]]
        titles = data[1]
        descriptions = data[2]
        urls = data[3]
        candidates = []
        for t, desc, candidate_url in zip(titles, descriptions, urls):
            if "basketball" in t.lower() or "basketball" in desc.lower():
                candidates.append(candidate_url)
        if candidates:
            chosen_url = candidates[0]
        elif urls:
            chosen_url = urls[0]
        else:
            chosen_url = None
        if chosen_url:
            new_response = requests.get(chosen_url, headers=headers)
            if new_response.status_code == 200:
                return new_response.text
            else:
                print(f"Fallback failed to retrieve {chosen_url}")
                return None
        else:
            print(f"No candidates found in opensearch for {player_name}")
            return None
    else:
        print(f"Failed to search Wikipedia for {player_name}")
        return None

def extract_infobox(html):
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", class_=lambda c: c and "infobox" in c)
    data={}
    if tbl:
        for row in tbl.find_all("tr"):
            if row.th and row.td:
                k=row.th.get_text(" ",strip=True)
                v=row.td.get_text(" ",strip=True)
                data[k]=v
    else:
        print("No infobox found in wiki HTML.")
    return data

def parse_born_field(born_text):
    # matches: ( 2005-03-08 )... or ( 2005-03-08 )...
    pat1=r'^\(\s*(?P<dob>\d{4}-\d{2}-\d{2})\s*\).*?\(age.*?\)\s*(?P<pob>.+)$'
    m = re.match(pat1, born_text)
    if m:
        return m.group("dob").strip(), m.group("pob").strip()
    pat2=r'^\(\s*(?P<dob>\d{4}-\d{2}-\d{2})\s*\)\s*(?P<pob>.+)$'
    m2 = re.match(pat2, born_text)
    if m2:
        return m2.group("dob").strip(), m2.group("pob").strip()
    return None,None

def remove_metric_conversion(val):
    return val.split('(')[0].strip()

def clean_infobox_height_weight(info):
    for k in list(info.keys()):
        if "height" in k.lower() or "weight" in k.lower():
            info[k] = remove_metric_conversion(info[k])

def extract_awards(html):
    soup = BeautifulSoup(html, "html.parser")
    head = soup.find(lambda tag: tag.name=="th" and "Career highlights and awards" in tag.get_text())
    if head:
        tr=head.find_parent("tr")
        if tr:
            nxt=tr.find_next_sibling("tr")
            if nxt:
                ul=nxt.find("ul")
                if ul:
                    lis=[li.get_text(" ",strip=True) for li in ul.find_all("li")]
                    return ", ".join(lis)
    return ""


###############################################################################
# 4) Searching sports-ref / bball-ref
###############################################################################

def fallback_cbb_slug(player_name, max_number=10):
    base_slug = slugify_player_name(player_name)
    base_url = "https://www.sports-reference.com/cbb/players"
    for i in range(1, max_number + 1):
        test_url = f"{base_url}/{base_slug}-{i}.html"
        # Throttle
        rate_limiter.wait_if_needed()
        try:
            r = requests.get(test_url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
            if r.status_code == 200:
                print(f"Found cbb fallback url: {test_url}")
                return test_url
        except requests.exceptions.RequestException as e:
            print(f"Fallback cbb slug error {test_url}: {e}")
    return None

def search_player_selenium(player_name, section="cbb", max_retries=3, retry_delay=5):
    # This uses Selenium for the search page
    # We can't easily apply the RateLimiter for the Selenium driver.get call
    # but we can do a small time.sleep. We'll do the time.sleep anyway.
    time.sleep(3)

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    try:
        driver = webdriver.Chrome(options=chrome_options)
    except WebDriverException as e:
        print("WebDriver error:", e)
        return None
    driver.set_page_load_timeout(180)

    base = f"https://www.sports-reference.com/{section}/search/search.fcgi"
    q = urllib.parse.urlencode({"search": player_name, "exact": "1"})
    search_url = f"{base}?{q}"

    for attempt in range(1, max_retries + 1):
        try:
            driver.get(search_url)
            break
        except Exception as ee:
            print(f"Attempt {attempt} of {max_retries} for {search_url} failed: {ee}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                driver.quit()
                return None

    curr = driver.current_url
    if f"/{section}/players/" in curr:
        print("SR direct result:", curr)
        driver.quit()
        return curr

    time.sleep(3)
    html = driver.page_source
    driver.quit()

    sp = BeautifulSoup(html, "html.parser")
    container = sp.find("div", class_=f"ac-dataset-{section}__players")
    if container:
        suggestions = container.find_all("div", class_="ac-suggestion")
        normp = normalize_name(player_name)
        for s in suggestions:
            link = s.find("a")
            if link and 'href' in link.attrs:
                text = normalize_name(link.get_text())
                if normp in text or text in normp:
                    out = "https://www.sports-reference.com" + link['href']
                    return out
    else:
        print(f"No SR container for {section} with {player_name}")
    return None

def search_player_sports_ref(player_name):
    u = search_player_selenium(player_name, "cbb")
    if u:
        return u

    print(f"Trying fallback cbb slug for {player_name}...")
    fallback_url = fallback_cbb_slug(player_name)
    if fallback_url:
        return fallback_url

    print(f"Trying sports-reference 'international' for {player_name}")
    return search_player_selenium(player_name, "international")

def search_player_bball_ref_international(player_name):
    alt = get_alt_slug(player_name)
    h = {"User-Agent":"Mozilla/5.0"}
    if alt:
        # Rate-limit
        rate_limiter.wait_if_needed()
        test = f"https://www.basketball-reference.com/international/players/{alt}.html"
        try:
            r = requests.get(test, headers=h, timeout=15)
            if r.status_code==200:
                print(f"Found BRef intl alt slug: {test}")
                return test
        except requests.exceptions.RequestException as e:
            print(f"BRef intl alt slug error {test}: {e}")

    slug = slugify_player_name(player_name)
    base = "https://www.basketball-reference.com/international/players"
    for i in range(1,6):
        trial = f"{base}/{slug}-{i}.html"
        rate_limiter.wait_if_needed()
        try:
            r2 = requests.get(trial, headers=h, timeout=15)
            if r2.status_code==200:
                print(f"BRef intl found: {trial}")
                return trial
        except requests.exceptions.RequestException as e:
            print(f"BRef intl error {trial}: {e}")
    return None

def search_player_bball_ref_gleague(player_name):
    alt = get_alt_slug(player_name)
    h = {"User-Agent":"Mozilla/5.0"}
    if alt:
        letter = alt[0]
        test = f"https://www.basketball-reference.com/gleague/players/{letter}/{alt}.html"
        rate_limiter.wait_if_needed()
        try:
            r = requests.get(test, headers=h, timeout=15)
            if r.status_code==200:
                print(f"BRef GLeague alt slug found: {test}")
                return test
        except requests.exceptions.RequestException as e:
            print(f"GLeague alt slug error {test}: {e}")

    slug = slugify_player_name(player_name)
    letter = slug[0] if slug else ""
    base = f"https://www.basketball-reference.com/gleague/players/{letter}"
    for i in range(1,6):
        check = f"{base}/{slug}-{i}.html"
        rate_limiter.wait_if_needed()
        try:
            rr = requests.get(check, headers=h, timeout=15)
            if rr.status_code==200:
                print(f"BRef GLeague found: {check}")
                return check
        except requests.exceptions.RequestException as e:
            print(f"BRef GLeague error {check}: {e}")
    return None

def search_player(player_name):
    sr = search_player_sports_ref(player_name)
    if sr:
        return sr

    print(f"Trying bball-ref 'international' for {player_name}...")
    bri = search_player_bball_ref_international(player_name)
    if bri:
        return bri

    print(f"Trying bball-ref 'G-League' for {player_name}...")
    brg = search_player_bball_ref_gleague(player_name)
    if brg:
        return brg

    return None

###############################################################################
# 5) Stats Extraction with get_url_with_retry
###############################################################################

def get_url_with_retry(url, headers=None, max_retries=3, retry_delay=5):
    """
    We'll do the rate_limiter and multiple attempts if status != 200 or request fails.
    """
    if headers is None:
        headers = {"User-Agent":"Mozilla/5.0"}

    for attempt in range(1, max_retries + 1):
        # Rate limit
        rate_limiter.wait_if_needed()

        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                return r
            else:
                print(f"[Attempt {attempt}] {url} => status {r.status_code}")
        except requests.exceptions.Timeout as e:
            print(f"[Attempt {attempt}] Timeout for {url}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"[Attempt {attempt}] RequestError for {url}: {e}")

        if attempt < max_retries:
            print(f"Sleeping {retry_delay}s then retrying {url} ...")
            time.sleep(retry_delay)

    print(f"All {max_retries} attempts failed for {url}")
    return None

def parse_all_srch_tables_for_cbb(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    candidate_tables = []
    all_tbls = soup.find_all("table")
    for tbl in all_tbls:
        thd = tbl.find("thead")
        if thd and "Season" in thd.get_text():
            candidate_tables.append(tbl)

    cmt = soup.find_all(string=lambda txt: isinstance(txt, Comment))
    for co in cmt:
        if "table" in co:
            c_soup = BeautifulSoup(co, "html.parser")
            hidden_tbls = c_soup.find_all("table")
            for ht in hidden_tbls:
                hh = ht.find("thead")
                if hh and "Season" in hh.get_text():
                    candidate_tables.append(ht)

    for candidate in candidate_tables:
        try:
            df_list = pd.read_html(str(candidate))
            df = df_list[0]
            if "Season" in df.columns:
                return df
        except:
            pass
    return None

def get_sr_per_game_table(player_url):
    r = get_url_with_retry(player_url)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", id="per_game")
    if not table:
        table = soup.find("table", id="players_per_game")

    if table:
        try:
            df = pd.read_html(str(table))[0]
            df.rename(columns={
                "year_id":"Season","team_name_abbr":"Team","conf_abbr":"Conf",
                "class":"Class","pos":"Pos"
            }, inplace=True, errors="ignore")
            return df
        except Exception as e:
            print(f"Default parse error SR table: {e}")

    print("No 'per_game' or 'players_per_game' table. Trying fallback cbb parse.")
    fallback_df = parse_all_srch_tables_for_cbb(r.text)
    if fallback_df is not None:
        fallback_df.rename(columns={
            "year_id":"Season","team_name_abbr":"Team","conf_abbr":"Conf",
            "class":"Class","pos":"Pos"
        }, inplace=True, errors="ignore")
        return fallback_df

    return None

def get_international_stats_br(player_url):
    r = get_url_with_retry(player_url)
    if not r:
        return None
    soup = BeautifulSoup(r.text,"html.parser")
    candidate_tables = []
    all_tbl = soup.find_all("table")
    for tbl in all_tbl:
        hd = tbl.find("thead")
        if hd and "Season" in hd.get_text():
            candidate_tables.append(tbl)

    cmt = soup.find_all(string=lambda txt: isinstance(txt, Comment))
    for co in cmt:
        if "table" in co:
            co_soup = BeautifulSoup(co, "html.parser")
            hidden_tbl = co_soup.find_all("table")
            for ht in hidden_tbl:
                if ht.find("thead") and "Season" in ht.find("thead").get_text():
                    candidate_tables.append(ht)

    for candidate in candidate_tables:
        try:
            dfl = pd.read_html(str(candidate))
            df = dfl[0]
            if "Season" in df.columns:
                return df
        except:
            pass
    return None

def get_bball_ref_gleague_stats(url):
    r = get_url_with_retry(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text,"html.parser")

    candidate_tables = []
    alltbl = soup.find_all("table")
    for tbl in alltbl:
        hd = tbl.find("thead")
        if hd and "Season" in hd.get_text():
            candidate_tables.append(tbl)

    cmt=soup.find_all(string=lambda txt:isinstance(txt,Comment))
    for co in cmt:
        if "table" in co:
            c_soup=BeautifulSoup(co,"html.parser")
            hidden=c_soup.find_all("table")
            for ht in hidden:
                if ht.find("thead") and "Season" in ht.find("thead").get_text():
                    candidate_tables.append(ht)

    if not candidate_tables:
        print("No G-League table found on BRef page.")
        return None

    best_df=None
    for t in candidate_tables:
        try:
            dfl = pd.read_html(str(t))
            df = dfl[0]
            if "Season" in df.columns:
                best_df=df
                break
        except:
            pass

    if best_df is None or best_df.empty:
        print("Failed to parse any G-League table with 'Season'.")
        return None
    return best_df

def get_college_stats(player_url):
    if "basketball-reference.com/gleague" in player_url:
        return get_bball_ref_gleague_stats(player_url)
    elif "basketball-reference.com/international" in player_url:
        return get_international_stats_br(player_url)
    elif "sports-reference.com" in player_url:
        return get_sr_per_game_table(player_url)
    else:
        print("Unrecognized stats page (not sports-ref or bball-ref). Skipping parse.")
        return None

###############################################################################
# 6) Season Parsing
###############################################################################

def parse_season_str(season_str):
    season_str = season_str.strip().replace("–","-")
    m = re.match(r"^(\d{4})-(\d{2,4})$", season_str)
    if not m:
        return (None, None)
    start_yr = int(m.group(1))
    end_part = m.group(2)
    if len(end_part) == 2:
        end_2 = int(end_part)
        end_yr = 2000 + end_2 if end_2 < 50 else 1900 + end_2
    else:
        end_yr = int(end_part)
    return (start_yr, end_yr)

def extract_stats_for_draft_year(df, draft_year):
    if df is None or df.empty:
        return {}
    if "Season" not in df.columns:
        return {}

    seasons_info = []
    for i, row in df.iterrows():
        season_str = str(row["Season"])
        start_yr, end_yr = parse_season_str(season_str)
        if end_yr is not None:
            seasons_info.append((i, start_yr, end_yr))

    if not seasons_info:
        return {}

    exact_matches = [(i, s, e) for (i, s, e) in seasons_info if e == draft_year]
    if exact_matches:
        best_i, best_start, best_end = max(exact_matches, key=lambda x: x[1])
    else:
        valid_idxs = [(i, s, e) for (i, s, e) in seasons_info if e < draft_year]
        if not valid_idxs:
            return {}
        best_i, best_start, best_end = max(valid_idxs, key=lambda x: x[2])

    last_row = df.loc[best_i]
    valid_for_avg = [i for (i, s, e) in seasons_info if e <= best_end]
    subset = df.loc[valid_for_avg]
    numeric_cols = subset.select_dtypes(include=['number']).columns

    out = {}
    for c in df.columns:
        val = last_row.get(c, None)
        if isinstance(val, str):
            val = val.strip()
        out["LS_"+c] = val

    if len(numeric_cols) > 0:
        means = subset[numeric_cols].mean()
        for c in df.columns:
            if c in numeric_cols:
                out["AVG_"+c] = means.get(c, None)
            else:
                if c=="Season":
                    out["AVG_"+c]="Average"
                else:
                    out["AVG_"+c]=None
    else:
        for c in df.columns:
            if c=="Season":
                out["AVG_"+c]="Average"
            else:
                out["AVG_"+c]=None

    return out

###############################################################################
# 7) Main script
###############################################################################

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enrich draft CSV with Wikipedia and basketball-reference data")
    parser.add_argument("input_csv", help="Path to input CSV file containing a 'Player' column")
    parser.add_argument("output_csv", help="Path to output enriched CSV")
    args = parser.parse_args()

    try:
        df_players = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error reading {args.input_csv}: {e}")
        return

    enriched_rows = []
    for idx, row in df_players.iterrows():
        player_name = str(row.get("Player", "")).strip()
        if not player_name:
            print(f"Skipping row {idx}: empty Player")
            enriched_rows.append(row.to_dict())
            continue
        print(f"Processing {player_name}...")

        # A) Wikipedia data
        wiki_html = get_wikipedia_page(player_name)
        wiki_infobox = {}
        if wiki_html:
            wiki_infobox = extract_infobox(wiki_html)
            if "Born" in wiki_infobox:
                dob, pob = parse_born_field(wiki_infobox["Born"]); 
                if dob: wiki_infobox["Date of Birth"] = dob
                if pob: wiki_infobox["Place of Birth"] = pob
                del wiki_infobox["Born"]
            clean_infobox_height_weight(wiki_infobox)
            aw = extract_awards(wiki_html)
            if aw: wiki_infobox["Awards"] = aw

        # B) Stats data
        stats_url = search_player(player_name)
        stats_dict = {}
        if stats_url:
            print(f"  Found stats URL: {stats_url}")
            df_stats = get_college_stats(stats_url)
            if df_stats is not None:
                draft_year = int(row.get("Year", 0))
                stats_dict = extract_stats_for_draft_year(df_stats, draft_year)
            else:
                print(f"  No stats table for {player_name}")
        else:
            print(f"  No stats URL for {player_name}")

        # Build output row
        row_out = row.to_dict()
        # merge Wikipedia fields
        for field in ["Position","College","Listed height","Listed weight","Nationality","Date of Birth","Place of Birth","Awards"]:
            if field in wiki_infobox:
                row_out[field] = wiki_infobox[field]
        # merge stats fields
        for k, v in stats_dict.items():
            row_out[k] = v

        enriched_rows.append(row_out)
        time.sleep(1)

    df_out = pd.DataFrame(enriched_rows)
    df_out.to_csv(args.output_csv, index=False)
    print(f"Enriched CSV written to {args.output_csv}")

if __name__ == "__main__":
    main()

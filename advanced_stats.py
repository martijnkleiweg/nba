# This script fetches advanced NBA stats from Basketball Reference for each player
# Student: Martinus Kleiweg
import re
import time
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
import urllib.parse
import argparse

###############################
# 0) Rate Limiter (to avoid 429s)
###############################
class SRRateLimiter:
    def __init__(self, max_requests=20, window_secs=60):
        self.max_requests = max_requests
        self.window_secs = window_secs
        self.request_times = []

    def wait_if_needed(self):
        now = time.time()
        cutoff = now - self.window_secs

        # before purge
        print(f"[RateLimiter] currently {len(self.request_times)} reqs in last {self.window_secs}s")

        # purge old timestamps
        while self.request_times and self.request_times[0] < cutoff:
            popped = self.request_times.pop(0)
            print(f"[RateLimiter] purged timestamp {popped:.2f}")

        # if at capacity, sleep
        if len(self.request_times) >= self.max_requests:
            sleep_for = (self.request_times[0] + self.window_secs) - now
            print(f"[RateLimiter] limit reached ({len(self.request_times)}). Sleeping {sleep_for:.2f}s…")
            time.sleep(max(sleep_for, 0))

            # purge again
            now = time.time()
            cutoff = now - self.window_secs
            while self.request_times and self.request_times[0] < cutoff:
                popped = self.request_times.pop(0)
                print(f"[RateLimiter] purged timestamp after sleep {popped:.2f}")

        # record this request
        self.request_times.append(time.time())
        print(f"[RateLimiter] proceeding, new count {len(self.request_times)}")

rate_limiter = SRRateLimiter()

###############################
# A) Find B-Ref NBA page via Selenium search
###############################
def normalize_name(name):
    return re.sub(r'[^a-z0-9\s]+','', name.strip().lower())

def search_nba_player_bbr(player_name, max_retries=3, retry_delay=2):
    print(f"→ Searching BRef for '{player_name}'…")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    try:
        driver = webdriver.Chrome(options=chrome_options)
    except WebDriverException as e:
        print("WebDriver error:", e)
        return None

    base = "https://www.basketball-reference.com/search/search.fcgi"
    q = urllib.parse.urlencode({"search": player_name})
    search_url = f"{base}?{q}"

    for attempt in range(1, max_retries+1):
        try:
            print(f"  [Selenium] loading search page (attempt {attempt})")
            driver.get(search_url)
            print(f"  [Selenium] loaded, current_url = {driver.current_url}")
            break
        except Exception as e:
            print(f"  [Selenium] load error ({e}), retrying in {retry_delay}s…")
            time.sleep(retry_delay)
    else:
        driver.quit()
        print("  ✗ Could not load search page")
        return None

    current = driver.current_url
    if "/players/" in current and "search" not in current:
        print("  ✓ Direct player page:", current)
        driver.quit()
        return current

    html = driver.page_source
    driver.quit()
    soup = BeautifulSoup(html, "html.parser")
    div = soup.find("div", class_="search-item-name")
    if div and div.a and div.a["href"].startswith("/players/"):
        resolved = "https://www.basketball-reference.com" + div.a["href"]
        print("  ✓ Resolved via search-item:", resolved)
        return resolved

    print("  ✗ No player page found in search results")
    return None

###############################
# B) Rate-limited GET + parse "Advanced" table
###############################
def get_url_with_retry(url, max_retries=3, retry_delay=3):
    headers = {"User-Agent":"Mozilla/5.0"}
    for attempt in range(1, max_retries+1):
        print(f"[get_url] attempt {attempt} → GET {url}")
        rate_limiter.wait_if_needed()
        try:
            r = requests.get(url, headers=headers, timeout=30)
            print(f"[get_url] status {r.status_code} for {url}")
            if r.status_code == 200:
                return r
        except Exception as e:
            print(f"[get_url] exception {e} on attempt {attempt}")
        time.sleep(retry_delay)
    print(f"[get_url] ✗ All attempts failed for {url}")
    return None

def parse_nba_advanced_stats(bbr_url):
    print(f"[parse_nba_advanced_stats] fetching {bbr_url}")
    r = get_url_with_retry(bbr_url)
    if not r:
        print(f"[parse_nba_advanced_stats] failed to retrieve page")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    tbl = soup.find("table", id="advanced")
    if not tbl:
        print(f"[parse_nba_advanced_stats] no 'advanced' table found")
        return None

    try:
        df = pd.read_html(StringIO(str(tbl)))[0]
        print(f"[parse_nba_advanced_stats] parsed table, shape = {df.shape}")
    except Exception as e:
        print(f"[parse_nba_advanced_stats] parse error: {e}")
        return None

    return df

###############################
# C) Compute ×-year post-draft averages & composite (2-dec rounding)
###############################
def parse_season_str(season):
    m = re.match(r"^(\d{4})-(\d{2})$", str(season).strip())
    return int(m.group(1)) if m else None

def compute_player_score(df_adv, draft_year, years_post=3):
    if df_adv is None or df_adv.empty:
        print("[compute_player_score] no advanced data")
        return None

    df_adv['Start_Year'] = df_adv['Season'].apply(parse_season_str)
    print(f"[compute_player_score] after parsing Season, shape = {df_adv.shape}")
    df_adv = df_adv[df_adv['Start_Year'].notnull()].copy()
    print(f"[compute_player_score] after dropping null Start_Year, shape = {df_adv.shape}")

    last_year = draft_year + years_post - 1
    df_post = df_adv[df_adv['Start_Year'].between(draft_year, last_year)]
    print(f"[compute_player_score] post-draft seasons ({draft_year}–{last_year}): {df_post.shape[0]} rows")
    if df_post.empty:
        return None

    def pick_tot_or_last(grp):
        if 'Tm' in grp.columns:
            tot = grp[grp['Tm']=='TOT']
            if not tot.empty:
                return tot.iloc[0]
        return grp.iloc[-1]

    df_post = df_post.groupby('Season', group_keys=False).apply(pick_tot_or_last).reset_index(drop=True)
    print(f"[compute_player_score] after grouping, shape = {df_post.shape}")

    rename_map = {'PER':'PER','TS%':'TS_pct','WS/48':'WS_per_48','BPM':'BPM','VORP':'VORP'}
    for old, new in rename_map.items():
        df_post[new] = pd.to_numeric(df_post.get(old), errors='coerce')

    metrics = ['PER','TS_pct','WS_per_48','BPM','VORP']
    means = df_post[metrics].mean()
    stats = {m: round(float(means[m]),2) if pd.notna(means[m]) else None for m in metrics}
    print(f"[compute_player_score] means = {stats}")

    def safe(x): return x if x is not None else 0.0
    comp = 0.4*safe(stats['BPM']) + 0.4*(safe(stats['WS_per_48'])*100) + 0.2*(safe(stats['TS_pct'])*100)
    stats['composite_score'] = round(comp,2)
    print(f"[compute_player_score] composite_score = {stats['composite_score']}")

    return stats

###############################
# D) Main: CSV I/O
###############################
def main():
    parser = argparse.ArgumentParser(description="Add NBA advanced stats to draft CSV")
    parser.add_argument('input_csv', help='Input CSV with Player and Year columns')
    parser.add_argument('output_csv', help='Output CSV path')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    new_cols = ['nba_per','nba_ts_pct','nba_ws_per_48','nba_bpm','nba_vorp','nba_composite_score']
    for col in new_cols:
        df[col] = None

    for idx, row in df.iterrows():
        name = str(row.get('Player') or row.get('player','')).strip()
        year = int(row.get('Year') or row.get('year',0))
        print(f"\nProcessing row {idx}: {name} (draft {year})")
        if not name or year == 0:
            print("  Skipping invalid name/year")
            continue

        url = search_nba_player_bbr(name)
        if not url:
            print("  ✗ No page found")
            continue

        df_adv = parse_nba_advanced_stats(url)
        stats = compute_player_score(df_adv, year, years_post=3)
        if not stats:
            print("  ✗ No stats computed")
            continue

        print(f"  ✓ Stats computed, writing back to DataFrame")
        df.at[idx, 'nba_per']             = stats['PER']
        df.at[idx, 'nba_ts_pct']          = stats['TS_pct']
        df.at[idx, 'nba_ws_per_48']       = stats['WS_per_48']
        df.at[idx, 'nba_bpm']             = stats['BPM']
        df.at[idx, 'nba_vorp']            = stats['VORP']
        df.at[idx, 'nba_composite_score'] = stats['composite_score']
        time.sleep(2)

    df.to_csv(args.output_csv, index=False)
    print(f"\nEnriched CSV saved to {args.output_csv}")

if __name__ == '__main__':
    main()

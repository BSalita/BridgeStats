import requests
from bs4 import BeautifulSoup
import polars as pl
from typing import Dict, List, Tuple
from endplay.parsers import pbn
from urllib.parse import urlparse, parse_qsl
import streamlit as st
import mlBridgeLib.mlBridgeEndplayLib as mlBridgeEndplayLib
#from mlBridgeLib.mlBridgePostmortemLib import PBNResultsCalculator
import sys # todo: used for exit(1). keep or return None?
import re
import random
import xml.etree.ElementTree as ET
import html
import pathlib
from tqdm import tqdm

from mlBridgeLib.mlBridgeAugmentLib import AllAugmentations

class BridgeWebResultsParser:
    """
    Parser for bridge tournament results from BridgeWebs HTML pages.
    It can handle both pages with XML data streams and direct HTML result pages.
    """
    
    def __init__(self, url: str, timeout: int = 30):
        self.url = url
        self.timeout = timeout
        self.soup = None
        self.all_results = {}

    def _fetch_html(self, url) -> bool:
        """Fetch HTML content from the primary URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            self.soup = BeautifulSoup(response.content, 'html.parser')
            return True
        except requests.RequestException as e:
            print(f"Error fetching URL: {e}")
            return False

    def _get_xml_sections(self) -> List[Dict[str, str]]:
        """Finds all XML-based result sections from the main page."""
        sections = []
        buttons = self.soup.find_all('div', class_='button_href_one')
        for button in buttons:
            onclick = button.get('onclick')
            if onclick and 'resultsMsec' in onclick:
                match = re.search(r"resultsMsec\s*\(\s*(\d+)\s*\)", onclick)
                if match:
                    msec = match.group(1)
                    sections.append({'text': button.get_text(strip=True), 'msec': msec})
        return sections

    def _fetch_xml_data(self, referer_url: str, msec: str, club: str, ekey: str) -> str | None:
        """Fetches the XML data for a specific result section."""
        cookies = {
            'chart_settings2': '%7B%22linearDateAxis%22%3Atrue%2C%22movingAverage%22%3Atrue%2C%22dateRange%22%3A%223%22%2C%22NumberOfPointsMovAvg%22%3A%225%22%7D',
            'cbwsec': f'pid&display_rank&sessid&{random.randint(10**15, 10**16-1)}_{random.randint(10**15, 10**16-1)}&wd&1',
            'cbwopt': 'doc_height&1036&doc_width&1158&win_height&1036&bw_tz&-120&bw_size&1158&res2022&1',
        }
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
            'Connection': 'keep-alive',
            'Referer': referer_url,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
        }
        parsed_referer = urlparse(referer_url)
        base_url = f"{parsed_referer.scheme}://{parsed_referer.netloc}{parsed_referer.path}"
        params = {
            'xml': '1', 'club': club, 'pid': 'xml_results_rank', 'msec': msec,
            'mod': 'Results', 'ekey': ekey, 'rand': str(random.random()),
        }
        try:
            response = requests.get(base_url, params=params, cookies=cookies, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"An error occurred during XML data request: {e}")
            return None

    def _parse_xml_results_table(self, soup: BeautifulSoup) -> Tuple[List[Dict], List[Dict]]:
        """Parses the results table from XML-based HTML content."""
        results_table = soup.find('table', class_='brx_table')
        if not results_table: return [], []

        rows = results_table.find_all('tr')
        north_south_results, east_west_results = [], []
        current_section = None
        for row in rows:
            cells = row.find_all('td')
            if not cells: continue

            if len(cells) == 1 and 'brx_title' in cells[0].get('class', []):
                row_text = cells[0].get_text(strip=True)
                if 'North / South' in row_text: current_section = 'ns'
                elif 'East / West' in row_text: current_section = 'ew'
                continue
            
            if any('brx_title' in c.get('class', []) for c in cells): continue

            if current_section and len(cells) >= 5 and ('brx_even_lg' in row.get('class', []) or 'brx_odd_lg' in row.get('class', [])):
                if not cells[0].get_text(strip=True).isdigit(): continue
                try:
                    result = {
                        'position': cells[0].get_text(strip=True),
                        'pair_number': cells[1].get_text(strip=True),
                        'players': cells[2].get_text(strip=True).replace('&amp;', '&'),
                        'score_percent': float(cells[-3].get_text(strip=True)),
                        'match_points': float(cells[-4].get_text(strip=True)),
                        'direction': 'North/South' if current_section == 'ns' else 'East/West'
                    }
                    if current_section == 'ns': north_south_results.append(result)
                    else: east_west_results.append(result)
                except (ValueError, IndexError) as e:
                    print(f"Skipping XML results row: {row.get_text(strip=True)} due to error: {e}")
        return north_south_results, east_west_results

    def _parse_tournament_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Parses general tournament info from either XML or HTML."""
        info = {}
        title_elem = soup.find('span', class_='page_title res_header')
        if title_elem: info['title'] = title_elem.get_text(strip=True)
        
        info_table = soup.find('td', id='results_info')
        if info_table:
            for row in info_table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 3:
                    key = cells[0].get_text(strip=True).lower().replace(' ', '_')
                    info[key] = cells[2].get_text(strip=True)
        return info

    def _parse_mpts_table(self, soup: BeautifulSoup) -> pl.DataFrame:
        """Parses the player masterpoints table from XML-based HTML."""
        all_brx_tables = soup.find_all('table', class_='brx_table')
        if len(all_brx_tables) < 2: return pl.DataFrame()
        
        mpts_data = []
        for row in all_brx_tables[1].find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2 and 'brx_title' not in cells[0].get('class', []):
                mpts_str = cells[1].get_text(strip=True)
                mpts_data.append({
                    'player': cells[0].get_text(strip=True),
                    'mpts': int(mpts_str) if mpts_str.isdigit() else 0
                })
        return pl.DataFrame(mpts_data)

    def _parse_direct_html_table(self) -> Tuple[List[Dict], List[Dict]]:
        """Parses the results table from a direct HTML page."""
        all_tables = self.soup.find_all('table')
        results_table = None
        for table in all_tables:
            text = table.get_text()
            if 'North / South' in text and 'East / West' in text:
                results_table = table
                break
        if not results_table:
            raise ValueError("Could not find a results table with North/South and East/West sections")

        rows = results_table.find_all('tr')
        north_south_results, east_west_results = [], []
        current_section = None
        col_indices = {}

        for row in rows:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue

            row_text = ' '.join([cell.get_text(strip=True) for cell in cells])

            # Find header row to map column names to indices
            if 'Players' in row_text and ('Score%' in row_text or 'Score' in row_text):
                headers = [cell.get_text(strip=True).lower() for cell in cells]
                
                # Define aliases for the columns we need
                col_aliases = {
                    'pos': ['pos', 'position'],
                    'pair': ['no', 'pair'],
                    'players': ['players'],
                    'mp': ['matchpoints', 'mp', 'mps'],
                    'score': ['score%', 'score']
                }
                
                # Find the index for each required column
                for key, aliases in col_aliases.items():
                    for alias in aliases:
                        if alias in headers:
                            col_indices[key] = headers.index(alias)
                            break
                continue

            if 'North / South' in row_text:
                current_section = 'ns'
                continue
            elif 'East / West' in row_text:
                current_section = 'ew'
                continue

            # If we haven't found a valid header yet, skip
            required_keys = ['pos', 'pair', 'players', 'mp', 'score']
            if not all(k in col_indices for k in required_keys):
                continue

            # This should be a data row
            if current_section and cells[col_indices['pos']].get_text(strip=True).isdigit():
                # Check if we have enough cells for the required columns
                max_needed_index = max(col_indices[k] for k in required_keys)
                if len(cells) <= max_needed_index:
                    continue
                
                try:
                    result = {
                        'position': cells[col_indices['pos']].get_text(strip=True),
                        'pair_number': int(cells[col_indices['pair']].get_text(strip=True)),
                        'players': cells[col_indices['players']].get_text(strip=True).replace('&amp;', '&'),
                        'match_points': float(cells[col_indices['mp']].get_text(strip=True) or '0'),
                        'score_percent': float(cells[col_indices['score']].get_text(strip=True) or '0'),
                        'direction': 'North/South' if current_section == 'ns' else 'East/West'
                    }
                    if current_section == 'ns':
                        north_south_results.append(result)
                    else:
                        east_west_results.append(result)
                except (ValueError, IndexError) as e:
                    print(f"Skipping direct HTML row: {row.get_text(strip=True)} due to error: {e}")
                    
        return north_south_results, east_west_results

    def _parse_as_direct_html(self, club, event):
        """Parses the results from a direct HTML page."""
        print("No XML sections found. Parsing as direct HTML page.")
        try:
            ns_res, ew_res = self._parse_direct_html_table()
            info = self._parse_tournament_info(self.soup)

            if club:
                info['club'] = club
            if event:
                info['event'] = event
            # Extract results_session from the URL and add it to info
            if event:
                info['results_session'] = event

            self.all_results['html_parsed_results'] = {
                'north_south_results': pl.DataFrame(ns_res),
                'east_west_results': pl.DataFrame(ew_res),
                'tournament_info': pl.DataFrame([info])
            }
        except Exception as e:
            print(f"Error during direct HTML parsing: {e}")

    def get_first_result(self):
        """Returns the first set of results found."""
        if not self.all_results:
            return None
        first_key = next(iter(self.all_results))
        return self.all_results[first_key]

    def get_all_results(self) -> Dict:
        """
        Fetches and parses bridge results from the provided URL.
        Handles both XML-based and direct HTML pages.
        """
        if not self._fetch_html(self.url):
            return {}

        parsed_url = urlparse(self.url)
        query_params = dict(parse_qsl(parsed_url.query))
        club = query_params.get('club')
        event = query_params.get('event')

        sections = self._get_xml_sections()
        
        # If no sections found, assume it might be a single-session page
        # that requires an XML fetch.
        if not sections and club and event:
            sections = [{'text': 'main', 'msec': '1'}]

        if sections:
            for section in sections:
                xml_data = self._fetch_xml_data(self.url, section['msec'], club, event)
                if not xml_data:
                    continue
                
                try:
                    xml_root = ET.fromstring(xml_data)
                    results_html = html.unescape(xml_root.find('results').text)
                    section_soup = BeautifulSoup(results_html, 'html.parser')

                    ns_res, ew_res = self._parse_xml_results_table(section_soup)
                    
                    # If we got no player data from XML, this stream is likely not useful
                    if not ns_res and not ew_res:
                        continue

                    info = self._parse_tournament_info(section_soup)
                    if club:
                        info['club'] = club
                    if event:
                        info['event'] = event
                    if event:
                        info['results_session'] = event
                        
                    mpts_df = self._parse_mpts_table(section_soup)
                    
                    self.all_results[section['text']] = {
                        'north_south_results': pl.DataFrame(ns_res),
                        'east_west_results': pl.DataFrame(ew_res),
                        'tournament_info': pl.DataFrame([info]),
                        'player_mpts': mpts_df,
                        'msec': section['msec']
                    }
                except Exception as e:
                    print(f"Error processing section {section['text']}: {e}")

        # If after trying all that, we still have no results, *then* try direct HTML.
        if not self.all_results:
            self._parse_as_direct_html(club, event)

        return self.all_results


def read_pbn_file_from_url(source_url=None, msec='1'):
    """
    Extracts pid, event, club from the source URL and downloads PBN data
    
    Args:
        source_url: URL in the form of https://www.bridgewebs.com/cgi-bin/bwoq/bw.cgi?pid=display_rank&event=20250526_1&club=irelandimps
        msec: Section identifier (e.g., '1' for first section, '2' for second section)
    """
    if not source_url:
        st.error("PBN URL is not defined.")
        return None
    
    # Parse the source URL to extract parameters
    parsed_url = urlparse(source_url)
    query_params = dict(parse_qsl(parsed_url.query))
    
    event = query_params.get('event')
    club = query_params.get('club')

    # Fallback for older URL formats where club/event are in the path
    if not club and '/bwor/' in parsed_url.path:
        path_parts = parsed_url.path.split('/')
        if len(path_parts) > 1:
            club = path_parts[-2]

    try:
        if not event or not club:
            raise ValueError("Required parameters 'event' or 'club' not found in URL")
    except (KeyError, ValueError, IndexError) as e:
        st.error(f"Could not parse URL to get PBN file from '{source_url}': {e}")
        return None

    # Construct the PBN file URL
    url = "https://www.bridgewebs.com/cgi-bin/bwoq/bw.cgi"
    
    # Query parameters for PBN download
    params = {
        'pid': 'display_hands',
        'msec': msec,
        'event': event,
        'wd': '1',
        'club': club,
        'deal_format': 'pbn'
    }
    
    # Headers with the source URL as referer
    headers = {
        'Referer': source_url
    }
    
    print(f"Making request to: {url}")
    print(f"Parameters: {params}")
    print(f"Referer: {source_url}")
    
    # Make the GET request (equivalent to curl --silent --show-error --fail)
    response = requests.get(url, params=params, headers=headers)
    
    # Raise an exception for bad status codes (equivalent to --fail)
    response.raise_for_status()
    
    file_content = response.content.decode('utf-8')
            
    return file_content


def merge_parsed_and_pbn_dfs(path_url,boards,parser_dfs):
    df = mlBridgeEndplayLib.endplay_boards_to_df({path_url:boards})
    parser_df_ns = parser_dfs[0].with_columns([
        pl.col('pair_number').cast(pl.Utf8),
        pl.col('players').str.split('&').list.get(0).str.strip_chars().alias('Player_Name_N'),
        pl.col('players').str.split('&').list.get(1).str.strip_chars().alias('Player_Name_S')
    ]).drop('players')
    parser_df_ns = parser_df_ns.rename({
        'pair_number': 'PairId_NS',
        'position': 'Position_NS',
        'match_points': 'MatchPoints_NS',
        'score_percent': 'ScorePercent_NS',
    }).drop('direction')
    df = df.join(parser_df_ns, left_on='PairId_NS', right_on='PairId_NS', how='left')
    parser_df_ew = parser_dfs[1].with_columns([
        pl.col('pair_number').cast(pl.Utf8),
        pl.col('players').str.split('&').list.get(0).str.strip_chars().alias('Player_Name_E'),
        pl.col('players').str.split('&').list.get(1).str.strip_chars().alias('Player_Name_W')
    ]).drop('players')
    parser_df_ew = parser_df_ew.rename({
        'pair_number': 'PairId_EW',
        'position': 'Position_EW',
        'match_points': 'MatchPoints_EW',
        'score_percent': 'ScorePercent_EW',
    }).drop('direction')
    df = df.join(parser_df_ew, left_on='PairId_EW', right_on='PairId_EW', how='left')
    df = mlBridgeEndplayLib.convert_endplay_df_to_mlBridge_df(df)
    #pmb = PBNResultsCalculator()
    return df


def _filter_dataframe_for_pair(df, pair_direction, pair_number, player_direction, player_id, partner_direction, partner_id):

    df = df.with_columns(
        pl.col('PairId_'+pair_direction).eq(str(pair_number)).alias('Boards_I_Played')
    )

    df = df.with_columns(
        pl.col('Boards_I_Played').and_(pl.col('Declarer_Direction').eq(player_direction)).alias('Boards_I_Declared'),
        pl.col('Boards_I_Played').and_(pl.col('Declarer_Direction').eq(partner_direction)).alias('Boards_Partner_Declared'),
    )

    df = df.with_columns(
        pl.col('Boards_I_Played').alias('Boards_We_Played'),
        pl.col('Boards_I_Played').alias('Our_Boards'),
        (pl.col('Boards_I_Declared') | pl.col('Boards_Partner_Declared')).alias('Boards_We_Declared'),
    )

    df = df.with_columns(
        (pl.col('Boards_I_Played') & ~pl.col('Boards_We_Declared') & pl.col('Contract').ne('PASS')).alias('Boards_Opponent_Declared'),
    )

    return df


def process_bridgewebs_url(url: str, single_dummy_sample_count: int = 10, progress_class=tqdm):
    """
    Processes a BridgeWebs URL to fetch, parse, augment, and filter bridge game data.

    This function is designed for non-Streamlit, text-based script usage.

    Args:
        url (str): The URL of the BridgeWebs results page.
        single_dummy_sample_count (int): The number of single dummy productions for augmentation.
        progress_class: The progress bar class to use (e.g., tqdm).

    Returns:
        tuple: A tuple containing the processed DataFrame and a dictionary with player/game info,
               or (None, None) if processing fails.
    """
    with progress_class(total=6, desc="Processing BridgeWebs URL") as pbar:
        pbar.set_description("Parsing BridgeWebs results")
        parser = BridgeWebResultsParser(url)
        all_results = parser.get_all_results()
        pbar.update(1)

        if not all_results:
            print("Could not parse any results from the URL.")
            return None, None

        first_section_key = next(iter(all_results))
        results = all_results[first_section_key]

        parser_dfs = (
            results.get('north_south_results', pl.DataFrame()),
            results.get('east_west_results', pl.DataFrame()),
            results.get('tournament_info', pl.DataFrame())
        )

        pbar.set_description("Downloading PBN file")
        file_content = read_pbn_file_from_url(url, msec='1')
        pbar.update(1)

        pbar.set_description("Parsing PBN file")
        boards = pbn.loads(file_content)
        print(f"Parsed {len(boards)} boards from PBN file")
        pbar.update(1)

        pbar.set_description("Merging parsed and PBN data")
        path_url = pathlib.Path(url)
        df = merge_parsed_and_pbn_dfs(path_url, boards, parser_dfs)
        pbar.update(1)

        pbar.set_description("Augmenting data")
        augmenter = AllAugmentations(df, None, sd_productions=single_dummy_sample_count, progress=pbar)
        df, _ = augmenter.perform_all_augmentations()
        assert df.select(pl.col(pl.Object)).is_empty(), f"Found Object columns: {[col for col, dtype in df.schema.items() if dtype == pl.Object]}"
        pbar.update(1)
        
        pbar.set_description("Filtering data for top pair")
        info = {}
        info['session_id'] = parser_dfs[2]['results_session'].item()
        info['group_id'] = parser_dfs[2]['club'].item()

        combined_ns_ew_pairs = pl.concat([parser_dfs[0], parser_dfs[1]])

        pair_row = combined_ns_ew_pairs.sort('score_percent', descending=True).head(1)
        pair_number = pair_row['pair_number'].item()

        info['pair_direction'] = 'NS' if pair_row['direction'].item()[0] == 'N' else 'EW'
        player_direction = info['pair_direction'][0]
        partner_direction = info['pair_direction'][1]
        info['player_id'] = pair_row['players'].str.split('&').list.get(0).str.strip_chars().item()
        info['partner_id'] = pair_row['players'].str.split('&').list.get(1).str.strip_chars().item()
        info['opponent_pair_direction'] = 'EW' if info['pair_direction'] == 'NS' else 'NS'
        
        df = _filter_dataframe_for_pair(df, info['pair_direction'], pair_number, player_direction, info['player_id'], partner_direction, info['partner_id'])
        pbar.update(1)

    return df, info


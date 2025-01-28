from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

import requests
import pandas as pd
import time
from urllib3.util import Retry
from requests.adapters import HTTPAdapter


def get_nba_boxscores():
    # Create path for dk_import folder
    current_dir = Path(__file__).parent  # Gets the directory of current script
    dk_import_dir = current_dir.parent / 'dk_import'  # Gets sibling directory

    # Create dk_import directory if it doesn't exist
    dk_import_dir.mkdir(exist_ok=True)

    # Create full file path
    file_path = dk_import_dir / 'nba_boxscores.csv'

    # Setup retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # NBA Stats API endpoint for player box scores
    url = "https://stats.nba.com/stats/leaguegamelog"

    # Required headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.nba.com/',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }

    # Parameters for the API request
    params = {
        'Counter': '0',
        'DateFrom': '',
        'DateTo': '',
        'Direction': 'DESC',
        'LeagueID': '00',
        'PlayerOrTeam': 'P',
        'Season': '2024-25',
        'SeasonType': 'Regular Season',
        'Sorter': 'DATE'
    }

    try:
        # Add delay to respect rate limits
        time.sleep(1)

        response = session.get(url, headers=headers, params=params)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Extract headers and rows
        headers = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Save to CSV
        df.to_csv(file_path, index=False)
        print("Data successfully downloaded and saved to 'nba_boxscores.csv'")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None
    finally:
        session.close()


# Execute the function
boxscores_df = get_nba_boxscores()


def setup_driver():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    download_dir = os.path.join(os.path.dirname(current_dir), 'dk_import')

    chrome_options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)

    return webdriver.Chrome(options=chrome_options), download_dir


def download_talent_projections():
    driver, download_dir = setup_driver()

    try:
        print("Downloading talent projections...")
        driver.get("https://apanalytics.shinyapps.io/DARKO/_w_66db5831/#tab-7640-1")
        wait = WebDriverWait(driver, 20)
        time.sleep(5)

        # Find Current Player Skill Projections tab by text
        tabs = driver.find_elements(By.TAG_NAME, "a")
        skill_tab = None
        for tab in tabs:
            if tab.text.strip() == "Current Player Skill Projections":
                skill_tab = tab
                break

        if skill_tab:
            print("Clicking talent projections tab...")
            driver.execute_script("arguments[0].click();", skill_tab)
            time.sleep(5)

            # Find download button
            buttons = driver.find_elements(By.TAG_NAME, "a")
            download_button = None
            for button in buttons:
                if "Download Data" in button.text and button.is_displayed():
                    download_button = button
                    break

            if download_button:
                print("Downloading talent data...")
                driver.execute_script("arguments[0].click();", download_button)
                time.sleep(5)

                # Rename file
                list_of_files = os.listdir(download_dir)
                latest_file = max([os.path.join(download_dir, f) for f in list_of_files if f.endswith('.csv')],
                                  key=os.path.getctime)
                new_filename = os.path.join(download_dir, 'darko.csv')
                if os.path.exists(new_filename):
                    os.remove(new_filename)
                os.rename(latest_file, new_filename)
                print("Talent projections downloaded successfully")
            else:
                print("Could not find talent download button")
        else:
            print("Could not find talent projections tab")

    except Exception as e:
        print(f"Error downloading talent projections: {e}")

    finally:
        driver.quit()


def download_daily_projections():
    driver, download_dir = setup_driver()

    try:
        print("Downloading daily projections...")
        driver.get("https://apanalytics.shinyapps.io/DARKO/_w_66db5831/#tab-7640-1")
        wait = WebDriverWait(driver, 20)
        time.sleep(5)

        # Find Daily Player Per-Game Projections tab by text
        tabs = driver.find_elements(By.TAG_NAME, "a")
        daily_tab = None
        for tab in tabs:
            if tab.text.strip() == "Daily Player Per-Game Projections":
                daily_tab = tab
                break

        if daily_tab:
            print("Clicking daily projections tab...")
            driver.execute_script("arguments[0].click();", daily_tab)
            time.sleep(5)

            # Find download button
            buttons = driver.find_elements(By.TAG_NAME, "a")
            download_button = None
            for button in buttons:
                if "Download Data" in button.text and button.is_displayed():
                    download_button = button
                    break

            if download_button:
                print("Downloading daily data...")
                driver.execute_script("arguments[0].click();", download_button)
                time.sleep(5)

                # Rename file
                list_of_files = os.listdir(download_dir)
                latest_file = max([os.path.join(download_dir, f) for f in list_of_files if f.endswith('.csv')],
                                  key=os.path.getctime)
                new_filename = os.path.join(download_dir, 'darko_daily.csv')
                if os.path.exists(new_filename):
                    os.remove(new_filename)
                os.rename(latest_file, new_filename)
                print("Daily projections downloaded successfully")
            else:
                print("Could not find daily download button")
        else:
            print("Could not find daily projections tab")

    except Exception as e:
        print(f"Error downloading daily projections: {e}")

    finally:
        driver.quit()


if __name__ == "__main__":
    print("Starting downloads...")
    #download_talent_projections()
    #time.sleep(5)  # Wait between downloads
    #download_daily_projections()
    #time.sleep(5)  # Wait between downloads
    get_nba_boxscores()
    print("All downloads completed")
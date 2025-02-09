from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import requests
import pandas as pd
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import shutil



class DailyDownload:
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.download_dir = self.current_dir.parent / 'dk_import'
        self.download_dir.mkdir(exist_ok=True)

    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time
    import os
    import shutil

    def download_and_rename_csv(self, username, password):
        chrome_options = webdriver.ChromeOptions()
        download_dir = os.getcwd()
        # Create path for dk_import directory (sibling to current directory)
        dk_import_dir = os.path.join(os.path.dirname(os.getcwd()), 'dk_import')

        # Ensure dk_import directory exists
        if not os.path.exists(dk_import_dir):
            os.makedirs(dk_import_dir)
            print(f"Created directory: {dk_import_dir}")

        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)

        try:
            driver = webdriver.Chrome(options=chrome_options)

            print("Navigating to login page...")
            driver.get("https://basketballmonster.com/login.aspx")

            print("Finding login elements...")
            username_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "UsernameTB"))
            )
            password_field = driver.find_element(By.ID, "PasswordTB")

            username_field.clear()
            username_field.send_keys(username)
            password_field.clear()
            password_field.send_keys(password)

            login_button = driver.find_element(By.ID, "LoginButton")
            login_button.click()

            time.sleep(3)

            print("Login successful, navigating to projections...")
            driver.get("https://basketballmonster.com/dailyprojections.aspx")

            time.sleep(5)

            print("Looking for export button...")
            export_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='CSVBUTTON']"))
            )

            print("Found export button, clicking...")
            driver.execute_script("arguments[0].click();", export_button)

            print("Waiting for download...")
            time.sleep(15)

            downloads = [f for f in os.listdir(download_dir) if f.endswith('.csv')]
            if downloads:
                downloaded_file = max([os.path.join(download_dir, f) for f in downloads], key=os.path.getctime)
                new_filename = os.path.join(dk_import_dir, 'bbm.csv')

                if os.path.exists(new_filename):
                    os.remove(new_filename)

                shutil.move(downloaded_file, new_filename)
                print(f"File successfully downloaded and renamed to: {new_filename}")
            else:
                print("No CSV file was downloaded")

            time.sleep(5)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            if 'driver' in locals():
                print("Current URL:", driver.current_url)
                print("Page source:", driver.page_source[:1000])

        finally:
            if 'driver' in locals():
                driver.quit()

    def setup_driver(self):
        chrome_options = webdriver.ChromeOptions()
        prefs = {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        return webdriver.Chrome(options=chrome_options)

    def download_talent_projections(self):
        driver = self.setup_driver()

        try:
            print("Downloading talent projections...")
            driver.get("https://apanalytics.shinyapps.io/DARKO/_w_66db5831/#tab-7640-1")
            wait = WebDriverWait(driver, 20)
            time.sleep(5)

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

                    list_of_files = os.listdir(self.download_dir)
                    latest_file = max([os.path.join(self.download_dir, f) for f in list_of_files if f.endswith('.csv')],
                                    key=os.path.getctime)
                    new_filename = os.path.join(self.download_dir, 'darko.csv')
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

    def download_daily_projections(self):
        driver = self.setup_driver()

        try:
            print("Downloading daily projections...")
            driver.get("https://apanalytics.shinyapps.io/DARKO/_w_66db5831/#tab-7640-1")
            wait = WebDriverWait(driver, 20)
            time.sleep(5)

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

                    list_of_files = os.listdir(self.download_dir)
                    latest_file = max([os.path.join(self.download_dir, f) for f in list_of_files if f.endswith('.csv')],
                                    key=os.path.getctime)
                    new_filename = os.path.join(self.download_dir, 'darko_daily.csv')
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

    def get_nba_boxscores(self):
        file_path = self.download_dir / 'nba_boxscores.csv'

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        url = "https://stats.nba.com/stats/leaguegamelog"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.nba.com/',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }

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
            time.sleep(1)
            response = session.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            df = pd.DataFrame(rows, columns=headers)
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

    def download_all(self):
        print("Starting downloads...")
        self.download_talent_projections()
        time.sleep(5)
        self.download_daily_projections()
        time.sleep(5)
        self.get_nba_boxscores()
        print("All downloads completed")

if __name__ == "__main__":
    downloader = DailyDownload()
    #downloader.download_all()
    USERNAME = "marvalgames"
    PASSWORD = "NWMUCBPOUD"
    downloader.download_and_rename_csv(USERNAME, PASSWORD)

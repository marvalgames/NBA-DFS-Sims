import csv
import datetime
import itertools
import json
import math
import multiprocessing
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import timezone

import requests
import numpy as np
import pulp as plp
import pytz
from numba import jit
from scipy.stats import multivariate_normal
import zipfile
import uuid  # For generating unique keys


@jit(nopython=True)
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2


class NBA_Swaptimizer_Sims:
    site = None
    config = None
    problem = None
    output_dir = None
    num_uniques = None
    contest_id = None
    contest_name = None
    field_size = None
    num_lineups = None
    use_contest_data = True
    payout_structure = {}
    entry_fee = None
    team_list = []
    lineups = []
    player_lineups = {}
    player_keys = []
    contest_lineups = {}
    output_lineups = []
    field_lineups = {}
    player_dict = {}
    at_least = {}
    at_most = {}
    team_limits = {}
    matchups = set()
    matchup_list = []
    matchup_limits = {}
    matchup_at_least = {}
    ids_to_gametime = {}
    locked_matchups = {}
    time_remaining_dict = {}
    global_team_limit = None
    projection_minimum = None
    user_lineups = False
    contest_entries = {}
    max_salary = 49000
    projection_minimum = 16
    randomness_amount = 0
    num_minutes_per_player = 48
    optimal_score = 0
    min_salary = None
    teams_dict = defaultdict(list)
    missing_ids = {}
    lineup_sets = 5

    def print(self, *args, **kwargs):
        """Override to allow progress capturing"""
        print(*args, **kwargs)  # Default to regular print unless overridden


    def __init__(self, num_iterations, site=None, num_uniques=1, num_lineup_sets=5, min_salary=49000, projection_minimum=16,
                 contest_path=None, is_subprocess=False):
        self.is_subprocess = is_subprocess
        self.live_games = True
        self.entry_lineups = None
        self.user_entries = 6
        self.lineup_sets = num_lineup_sets
        self.site = site
        self.num_iterations = num_iterations
        self.num_uniques = int(num_uniques)
        self.min_salary = int(min_salary)
        self.projection_minimum = int(projection_minimum)
        self.roster_construction = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        self.max_salary = 50000
        self.load_config()
        self.load_rules()
        self.get_live_scores()
        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, self.config["player_path"]),
        )
        print('player path: ', player_path)
        self.load_player_ids(player_path)
        self.get_optimal()

        # Use provided contest_path if available
        print(contest_path)
        if contest_path and os.path.exists(contest_path):
            self.load_contest_data(contest_path)
            self.print(f"Contest data loaded from: {contest_path}")
        else:
            # Fallback to default path
            default_contest_path = os.path.join(
                os.path.dirname(__file__),
                "../{}_data/{}".format(self.site, self.config["contest_structure_path"]),
            )
            self.load_contest_data(default_contest_path)
            self.print("Contest payout structure loaded from default path.")

        # Load live contest path
        try:
            # Step 1: Define the folder path using the original logic
            folder_path = os.path.join(
                os.path.dirname(__file__),
                f"../{self.site}_data/"
            )

            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"The folder {folder_path} does not exist.")

            # Step 2: Locate the files starting with "contest-standings" with full paths
            contest_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith("contest-standings")
            ]

            if not contest_files:
                raise FileNotFoundError("No file starting with 'contest-standings' found in the folder.")

            # Sort files by modification time (newest first)
            contest_files = sorted(contest_files, key=os.path.getmtime, reverse=True)

            contest_file_path = os.path.join(folder_path, contest_files[0])  # Use the first match
            print(f"Found contest file: {contest_file_path}")

            # Step 3: Check if the file is a .zip and extract it
            if contest_file_path.endswith(".zip"):
                with zipfile.ZipFile(contest_file_path, 'r') as zip_ref:
                    extracted_files = zip_ref.namelist()
                    zip_ref.extractall(folder_path)
                    print(f"Extracted files: {extracted_files}")

                    if len(extracted_files) != 1:
                        raise ValueError("The .zip file should contain exactly one file.")

                    # Set the live_contest_path to the extracted file
                    live_contest_path = os.path.join(folder_path, extracted_files[0])
                    print(f"Live contest path set to extracted file: {live_contest_path}")
            else:
                # If not a zip file, use the file directly
                live_contest_path = contest_file_path
                print(f"Live contest path set to file: {live_contest_path}")

            # Step 4: Load the contest data
            self.extract_player_points(live_contest_path)
            self.load_live_contest(live_contest_path)
            print("Live contest loaded.")
            # Call the function
            #self.inspect_contest_lineups(self.contest_lineups)
            # Run simulation steps

        except Exception as e:
            print(f"An error occurred: {e}")

        if "late_swap_path" in self.config.keys():
            late_swap_path = os.path.join(
                os.path.dirname(__file__),
                "../dk_data/{}".format(self.config["late_swap_path"]),
            )
            self.load_player_lineups(late_swap_path)

    @staticmethod
    def init_worker():
        """Initialize each worker process with shared resources."""
        global rng
        rng = np.random.default_rng()  # Initialize RNG once per worker

    # Load config from file
    def load_config(self):
        with open(
                os.path.join(os.path.dirname(__file__), "../config.json"),
                encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    def get_optimal(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/
        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        problem = plp.LpProblem("NBA", plp.LpMaximize)
        lp_variables = {}
        for player, attributes in self.player_dict.items():
            player_id = attributes["ID"]
            for pos in attributes["Position"]:
                lp_variables[(player, pos, player_id)] = plp.LpVariable(
                    name=f"{player}_{pos}_{player_id}", cat=plp.LpBinary
                )

        # set the objective - maximize fpts & set randomness amount from config
        problem += (
            plp.lpSum(
                self.player_dict[player]["fieldFpts"]
                * lp_variables[(player, pos, attributes["ID"])]
                for player, attributes in self.player_dict.items()
                for pos in attributes["Position"]
            ),
            "Objective",
        )

        # Set the salary constraints
        max_salary = 50000
        min_salary = 49000

        if self.projection_minimum is not None:
            min_salary = self.min_salary

        # Maximum Salary Constraint
        problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, pos, attributes["ID"])]
                for player, attributes in self.player_dict.items()
                for pos in attributes["Position"]
            )
            <= max_salary,
            "Max Salary",
        )

        # Minimum Salary Constraint
        problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, pos, attributes["ID"])]
                for player, attributes in self.player_dict.items()
                for pos in attributes["Position"]
            )
            >= min_salary,
            "Min Salary",
        )

        # Must not play all 8 or 9 players from the same team (8 if dk, 9 if fd)
        for matchup in self.matchup_list:
            problem += (
                plp.lpSum(
                    lp_variables[(player, pos, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                    if attributes["Matchup"] == matchup
                )
                <= 8
            )

        if self.global_team_limit is not None:
            for teamIdent in self.team_list:
                problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Team"] == teamIdent
                    )
                    <= int(self.global_team_limit),
                    f"Global team limit - at most {self.global_team_limit} players from {teamIdent}",
                )

        # Constraints for specific positions
        for pos in ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]:
            problem += (
                plp.lpSum(
                    lp_variables[(player, pos, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    if pos in attributes["Position"]
                )
                == 1,
                f"Must have at 1 {pos}",
            )

        # Constraint to ensure each player is only selected once
        for player in self.player_dict:
            player_id = self.player_dict[player]["ID"]
            problem += (
                plp.lpSum(
                    lp_variables[(player, pos, player_id)]
                    for pos in self.player_dict[player]["Position"]
                )
                <= 1,
                f"Can only select {player} once",
            )

        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print(
                "282 Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                    len(self.output_lineups), self.lineups
                )
            )

        ## Check for infeasibility
        if plp.LpStatus[problem.status] != "Optimal":
            print(
                "290 Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                    len(self.lineups), self.num_lineups
                )
            )
        # Get the lineup and add it to our list
        player_unique_keys = [
            player for player in lp_variables if lp_variables[player].varValue != 0
        ]

        players = []
        for p in player_unique_keys:
            players.append(p[0])

        # Creating a list of player names
        players = [p[0] for p in player_unique_keys]

        # Printing neatly

        fpts_proj = sum(self.player_dict[player]["fieldFpts"] for player in players)
        self.optimal_score = float(fpts_proj)
        print(f"optimal score: {self.optimal_score}")

    def load_contest_data(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if self.field_size is None:
                    self.field_size = int(row["field size"])
                if self.entry_fee is None:
                    self.entry_fee = float(row["entry fee"])
                # multi-position payouts
                if "-" in row["place"]:
                    indices = row["place"].split("-")
                    # have to add 1 to range to get it to generate value for everything
                    for i in range(int(indices[0]), int(indices[1]) + 1):
                        # Where I'm from, we 0 index things. Thus, -1 since Payout starts at 1st place
                        if i >= self.field_size:
                            break
                        self.payout_structure[i - 1] = float(
                            row["payout"].split(".")[0].replace(",", "")
                        )
                # single-position payouts
                else:
                    if int(row["place"]) >= self.field_size:
                        break
                    self.payout_structure[int(row["place"]) - 1] = float(
                        row["payout"].split(".")[0].replace(",", "")
                    )

    @staticmethod
    def extract_players(lineup_string, positions_order):
        players = {}
        words = lineup_string.split()
        pos = None  # to keep track of current position
        player_name = []

        for word in words:
            if word in positions_order:
                if pos:
                    players[pos] = " ".join(player_name) if player_name else "LOCKED"
                pos = word
                player_name = []
            else:
                player_name.append(word)

        # for the last player in the string
        if pos:
            players[pos] = " ".join(player_name) if player_name else "LOCKED"

        # In case there are positions that weren't found in the lineup_string
        for position in positions_order:
            if position not in players:
                players[position] = "LOCKED"

        return players

    def update_bayesian_projection(self, player_key):
        player = self.player_dict[player_key]
        total_game_minutes = self.num_minutes_per_player  # total minutes in a game per player
        actual_fpts = player['ActualFpts']
        minutes_played = total_game_minutes - player['Minutes Remaining']

        # Calculate the game progress for the logarithmic scaling
        game_progress = minutes_played / total_game_minutes
        log_base = 10  # Define the base for logarithmic scaling

        if player['Minutes Remaining'] == 48:
            updated_projection = player['BayesianProjectedFpts']
            posterior_variance = player['BayesianProjectedVar']
        elif 48 > player['Minutes Remaining'] > 0:
            # Prior beliefs
            prior_mean = player['Fpts']
            prior_variance = player['StdDev'] ** 2

            # Points per minute based on prior mean
            ppm = prior_mean / total_game_minutes

            # Remaining variance considers the proportion of the game left
            remaining_variance = prior_variance * (player['Minutes Remaining'] / total_game_minutes)

            # Actual performance so far
            actual_ppm = actual_fpts / minutes_played

            # The scaling factor for the updated projection remains the same
            scaling_factor_projection = (minutes_played / total_game_minutes)

            # Weighted actual performance more as more of the game is played
            weighted_actual_ppm = actual_ppm * scaling_factor_projection + ppm * (1 - scaling_factor_projection)

            # Calculate the updated projection for the remaining minutes
            updated_remaining_projection = weighted_actual_ppm * player['Minutes Remaining']

            # Combine actual points with updated remaining projection for total updated projection
            updated_projection = actual_fpts + updated_remaining_projection

            minutes_ratio = minutes_played / total_game_minutes
            log_decay_factor_variance = math.log(minutes_ratio + 1, log_base) / math.log(total_game_minutes + 1,
                                                                                         log_base)

            # Adjust the scaling factor for variance to account for the decay
            scaling_factor_variance = log_decay_factor_variance

            # Adjust the remaining variance
            adjusted_remaining_variance = remaining_variance * (1 - scaling_factor_variance)

            # Update the posterior variance
            posterior_variance = adjusted_remaining_variance

        else:
            # If the game is over, the final score is just the actual points
            updated_projection = actual_fpts
            posterior_variance = 0

        # Update the player's projections
        if updated_projection < 1:
            posterior_variance = 0
            updated_projection = 0


        player['BayesianProjectedFpts'] = updated_projection
        player['BayesianProjectedVar'] = posterior_variance
        return player

    def format_games_table(self, games_info):
        try:
            from tabulate import tabulate
            has_tabulate = True
        except ImportError:
            has_tabulate = False

        # Convert team IDs to common names
        team_ids = {
            1610612737: 'ATL',
            1610612738: 'BOS',
            1610612739: 'CLE',
            1610612740: 'NOP',
            1610612741: 'CHI',
            1610612742: 'DAL',
            1610612743: 'DEN',
            1610612744: 'GSW',
            1610612745: 'HOU',
            1610612746: 'LAC',
            1610612747: 'LAL',
            1610612748: 'MIA',
            1610612749: 'MIL',
            1610612750: 'MIN',
            1610612751: 'BKN',
            1610612752: 'NYK',
            1610612753: 'ORL',
            1610612754: 'IND',
            1610612755: 'PHI',
            1610612756: 'PHX',
            1610612757: 'POR',
            1610612758: 'SAC',
            1610612759: 'SAS',
            1610612760: 'OKC',
            1610612761: 'TOR',
            1610612762: 'UTA',
            1610612763: 'MEM',
            1610612764: 'WAS',
            1610612765: 'DET',
            1610612766: 'CHA'

        }

        # Format game data
        formatted_games = []
        for game in games_info:
            away_team = team_ids.get(game[7], str(game[7]))
            home_team = team_ids.get(game[6], str(game[6]))
            game_time = game[4].strip()
            tv_info = game[11] if game[11] else ""
            arena = game[15]
            matchup = f"{away_team} @ {home_team}"

            formatted_games.append([
                matchup,
                game_time,
                tv_info,
                arena
            ])

        # Create table
        headers = ["Matchup", "Time/Status", "TV", "Arena"]

        return "\n" + tabulate(formatted_games, headers=headers, tablefmt="grid")


    def get_live_scores(self):
        game_date = datetime.datetime.now().date()

        team_id_to_abbreviation = {
            1610612737: 'ATL',
            1610612738: 'BOS',
            1610612739: 'CLE',
            1610612740: 'NOP',
            1610612741: 'CHI',
            1610612742: 'DAL',
            1610612743: 'DEN',
            1610612744: 'GSW',
            1610612745: 'HOU',
            1610612746: 'LAC',
            1610612747: 'LAL',
            1610612748: 'MIA',
            1610612749: 'MIL',
            1610612750: 'MIN',
            1610612751: 'BKN',
            1610612752: 'NYK',
            1610612753: 'ORL',
            1610612754: 'IND',
            1610612755: 'PHI',
            1610612756: 'PHX',
            1610612757: 'POR',
            1610612758: 'SAC',
            1610612759: 'SAS',
            1610612760: 'OKC',
            1610612761: 'TOR',
            1610612762: 'UTA',
            1610612763: 'MEM',
            1610612764: 'WAS',
            1610612765: 'DET',
            1610612766: 'CHA'
        }

        # Format the date into the string format the NBA API expects ('YYYY-MM-DD')
        # Late Swap Realtime
        live = self.live_games
        live = False
        if live:
            formatted_date = game_date.strftime('%Y-%m-%d')
        else:
            formatted_date = '2024-11-26'

        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Referer': 'https://www.nba.com/'}

        # Insert the formatted date into the URL
        scoreboard_url = f'https://stats.nba.com/stats/scoreboardv2?DayOffset=0&GameDate={formatted_date}&LeagueID=00'
        scoreboard_json = requests.get(scoreboard_url, headers=headers).json()
        if live:
            games_info = scoreboard_json['resultSets'][0]['rowSet']
        else:
            games_info = [

                ['2025-01-12T00:00:00', 1, '0022400539', 3, 'Final', '20250112/MILNYK', 1610612752, 1610612749, '2024',
                 4, '     ', None, 'MSG', 'FDSNWI', 'Q4       - ', 'Madison Square Garden', 1, 0],

                ['2025-01-12T00:00:00', 2, '0022400540', 3, 'Final', '20250112/DENDAL', 1610612742, 1610612743, '2024',
                 4, '     ', None, 'KFAA', 'ALT', 'Q4       - ', 'American Airlines Center', 1, 0],

                ['2025-01-12T00:00:00', 3, '0022400541', 3, 'Final', '20250112/SACCHI', 1610612741, 1610612758, '2024',
                 4, '     ', None, 'CHSN', 'NBCSCA', 'Q4       - ', 'United Center', 1, 0],

                ['2025-01-12T00:00:00', 4, '0022400542', 2, 'End of 3rd Qtr      ', '20250112/NOPBOS', 1610612738,
                 1610612740, '2024', 3, '     ', None, 'NBCSB', 'GCSEN', 'Q3       - ', 'TD Garden', 0, 0],

                ['2025-01-12T00:00:00', 5, '0022400543', 2, 'End of 3rd Qtr      ', '20250112/INDCLE', 1610612739,
                 1610612754, '2024', 3, '     ', None, 'FDSNOH', 'FDSNIN', 'Q3       - ', 'Rocket Mortgage FieldHouse',
                 0, 0],

                ['2025-01-12T00:00:00', 6, '0022400544', 2, '3rd Qtr             ', '20250112/PHIORL', 1610612753,
                 1610612755, '2024', 3, '3:53 ', None, 'FDSNFL', 'NBCSP', 'Q3 3:53  - ', 'Kia Center', 0, 0],

                ['2025-01-12T00:00:00', 7, '0022400545', 2, '3rd Qtr             ', '20250112/OKCWAS', 1610612764,
                 1610612760, '2024', 3, '2:44 ', None, 'MNMT', 'FDSNOK', 'Q3 2:44  - ', 'Capital One Arena', 0, 0],

                ['2025-01-12T00:00:00', 8, '0022400546', 1, '8:00 pm ET', '20250112/BKNUTA', 1610612762, 1610612751,
                 '2024', 0, '     ', None, 'KJZZ', 'YES', 'Q0       - ', 'Delta Center', 0, 0],

                ['2025-01-12T00:00:00', 9, '0022400547', 1, '9:00 pm ET', '20250112/CHAPHX', 1610612756, 1610612766,
                 '2024', 0, '     ', None, 'KTVK/KPHE', 'FDSNSE-CHA', 'Q0       - ', 'Footprint Center', 0, 0]]



        # After getting games_info, print the formatted table
        print("\nNBA Games:")
        self.print(self.format_games_table(games_info))
        # NBA regulation game length in minutes
        regulation_game_length = 48
        overtime_period_length = 5  # NBA overtime period length in minutes

        eastern = pytz.timezone('US/Eastern')
        if live:
            current_time_utc = datetime.datetime.now(timezone.utc)  # Current time in UTC
        else:
            current_time_utc = pytz.utc.localize(datetime.datetime(2024, 11, 26, 19, 35))  # Testing as aware datetime

        for game in games_info:
            game_id = game[2]
            home_team_id = game[6]
            visitor_team_id = game[7]
            game_status = game[4].strip()
            live_period = game[9]
            live_pc_time = game[10].strip()

            # Check if the game has a status indicating it's locked
            if 'Final' in game_status or 'Qtr' in game_status or 'Halftime' in game_status:
                game_locked = True
            else:
                # Handle scheduled games
                game_locked = False
                try:
                    # Convert game start time to datetime object
                    # Assuming game start time is in the format '10:00 pm ET'
                    game_date_str = game[0].split('T')[0]  # Extract the date part
                    game_start_time_str = game_status.replace('ET', '').strip()
                    game_start_time = datetime.datetime.strptime(game_date_str + ' ' + game_start_time_str,
                                                                 '%Y-%m-%d %I:%M %p')

                    # Convert to UTC
                    game_start_time_utc = eastern.localize(game_start_time).astimezone(pytz.utc)
                    # Check if current UTC time is past the game start time
                    if current_time_utc >= game_start_time_utc:
                        game_locked = True
                except ValueError:
                    # Handle parsing errors
                    print(f"Error parsing start time for game {game_id}, {game}")
            # Calculate the total time remaining
            if live_period <= 4:  # Regulation time
                total_minutes_remaining = (4 - live_period) * 12  # Time for the remaining quarters
                if live_pc_time:
                    minutes, seconds = map(int, live_pc_time.split(":"))
                    total_minutes_remaining += minutes  # Add remaining minutes for the current quarter

            else:  # Overtime
                completed_overtimes = live_period - 4
                total_minutes_remaining = (completed_overtimes * overtime_period_length)
                if live_pc_time:
                    minutes, seconds = map(int, live_pc_time.split(":"))
                    total_minutes_remaining += (
                            overtime_period_length - minutes)  # Add remaining minutes for the current overtime
                    if seconds > 0:
                        total_minutes_remaining -= 1  # Subtract a minute if there are seconds remaining

            # For finished games, set the remaining time to 0
            if 'Final' in game[4]:
                total_minutes_remaining = 0

            # Mapping team IDs to abbreviations and adding to the dictionary
            home_team_abbreviation = team_id_to_abbreviation.get(home_team_id, 'Unknown')
            visitor_team_abbreviation = team_id_to_abbreviation.get(visitor_team_id, 'Unknown')
            matchup = (visitor_team_abbreviation, home_team_abbreviation)
            self.matchup_list.append(matchup)
            # Assuming team_id_to_abbreviation is a dictionary that maps team IDs to their abbreviations
            # Initialize dictionary entries if they don't exist
            if home_team_abbreviation not in self.time_remaining_dict:
                self.time_remaining_dict[home_team_abbreviation] = {'Minutes Remaining': 0, 'GameLocked': False,
                                                                    'GameTime': None,
                                                                    'Opp': team_id_to_abbreviation.get(visitor_team_id,
                                                                                                       'Unknown'),
                                                                    'Matchup': matchup}
            if visitor_team_abbreviation not in self.time_remaining_dict:
                self.time_remaining_dict[visitor_team_abbreviation] = {'Minutes Remaining': 0, 'GameLocked': False,
                                                                       'GameTime': None,
                                                                       'Opp': team_id_to_abbreviation.get(home_team_id,
                                                                                                          'Unknown'),
                                                                       'Matchup': matchup}
            self.matchups.add(matchup)
            self.time_remaining_dict[home_team_abbreviation]['Minutes Remaining'] = total_minutes_remaining
            self.time_remaining_dict[visitor_team_abbreviation]['Minutes Remaining'] = total_minutes_remaining
            self.time_remaining_dict[home_team_abbreviation]['GameLocked'] = game_locked
            self.time_remaining_dict[visitor_team_abbreviation]['GameLocked'] = game_locked
            if game_locked:
                current_day = datetime.datetime.now().date()
                self.time_remaining_dict[home_team_abbreviation]['GameTime'] = datetime.datetime.combine(current_day,
                                                                                                         datetime.time(
                                                                                                             0, 1))
                self.time_remaining_dict[visitor_team_abbreviation]['GameTime'] = datetime.datetime.combine(current_day,
                                                                                                            datetime.time(
                                                                                                                0, 1))

                # Inside the else block where time parsing occurs
            else:
                date_part = datetime.datetime.strptime(game[0], '%Y-%m-%dT%H:%M:%S')

                # Convert game status to time
                time_part_str = game[4]

                try:
                    # First, handle special cases
                    if time_part_str == "Tipoff" or time_part_str == "Tipoff              ":
                        # Use a default time for tipoff (e.g., current time)
                        time_part = datetime.datetime.now()
                    elif "1st OT" in time_part_str:
                        time_part_str = time_part_str.replace("1st OT", "Q5")
                        clean_time_str = time_part_str.replace("ET", "").strip()
                        time_part = datetime.datetime.strptime(clean_time_str, '%I:%M %p')
                    else:
                        # Regular time parsing
                        clean_time_str = time_part_str.replace("ET", "").strip()
                        time_part = datetime.datetime.strptime(clean_time_str, '%I:%M %p')

                    # Combine date and time
                    game_datetime = datetime.datetime.combine(
                        date_part.date(),
                        time_part.time()
                    )

                    self.time_remaining_dict[home_team_abbreviation]['GameTime'] = game_datetime
                    self.time_remaining_dict[visitor_team_abbreviation]['GameTime'] = game_datetime

                except ValueError as e:
                    # Fallback: use current time if parsing fails
                    print(f"Warning: Could not parse time '{time_part_str}'. Using current time as fallback.")
                    fallback_time = datetime.datetime.now()
                    self.time_remaining_dict[home_team_abbreviation]['GameTime'] = fallback_time
                    self.time_remaining_dict[visitor_team_abbreviation]['GameTime'] = fallback_time

                #
                # date_part = datetime.datetime.strptime(game[0], '%Y-%m-%dT%H:%M:%S')
                # # Convert '9:00 pm ET' to 24-hour format and handle timezone
                # time_part_str = game[4]
                # # Handle special cases like '1st OT'
                # if "1st OT" in time_part_str:
                #     time_part_str = time_part_str.replace("1st OT", "Q5")
                #     print(f"Replaced invalid time data with: '{time_part_str}'")
                #
                # # Remove 'ET' and parse the time
                # try:
                #     # Clean the time string: remove 'ET' and strip extra whitespace
                #     clean_time_str = time_part_str.replace("ET", "").strip()
                #     time_part = datetime.datetime.strptime(clean_time_str, '%I:%M %p')
                #     print(f"Parsed time: {time_part.time()}")
                # except ValueError as e:
                #     print(f"Error parsing time: {e}")
                #
                # # Remove 'ET' and strip whitespace, then parse time
                # # time_part = datetime.datetime.strptime(time_part_str[:-3].strip(), '%I:%M %p')
                #
                #
                # # Combine date and time parts
                # combined_datetime = datetime.datetime.combine(date_part.date(), time_part.time())
                #
                # # Assume the input is for the Eastern Time timezone
                # eastern = pytz.timezone('US/Eastern')
                # localized_datetime = eastern.localize(combined_datetime)
                #
                # self.time_remaining_dict[home_team_abbreviation]['GameTime'] = localized_datetime
                # self.time_remaining_dict[visitor_team_abbreviation]['GameTime'] = localized_datetime

    def extract_player_points(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Player'] == '':
                    break
                else:
                    name = row['Player']
                    pos = row['Roster Position']
                    for k, v in self.player_dict.items():
                        if v['Name'] == name and pos in v['Position']:
                            self.player_dict[k]['ActualFpts'] = float(row['FPTS'])
                            self.update_bayesian_projection(k)

    def get_username(self, text):
        # The regex will match any text up until it possibly encounters a space followed by (digit/digit)
        match = re.search(r"^(.+?)(?:\s*\(\d+/\d+\))?$", text)
        if match:
            return match.group(1).strip()  # Strip to remove any trailing whitespaces
        return None  # or an appropriate value indicating no match

    def load_live_contest(self, path):
        match = re.search(r'contest-standings-(\d+).csv', path)
        positions_order = ["C", "F", "G", "PF", "PG", "SF", "SG", "UTIL"]
        total_minutes_for_full_lineup = len(self.roster_construction) * self.num_minutes_per_player

        if match:
            self.contest_id = match.group(1)
        else:
            print('Unable to find contest id for loading live lineups for contest simulation')
        players_not_found = []
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name = self.get_username(row['EntryName'])
                lineup_updated_proj = 0
                lineup_updated_var = 0
                if name in self.contest_entries.keys():
                    self.contest_entries[name]['Entries'] += 1
                else:
                    self.contest_entries[name] = {'Entries': 1, 'ROI': 0, 'Top1': 0, 'Cashes': 0, 'Wins': 0}
                lineup_dict = {
                    "contest_id": self.contest_id,
                    "EntryId": row['EntryId'],
                    'User': name,
                    "Type": 'opp'
                }
                lineup_proj_fpts = 0
                lineup_proj_stdv = 0
                lineup_salary = 0
                lineup_proj_fieldfpts = 0

                if not row["Lineup"].strip():
                    for pos in positions_order:
                        lineup_dict[f"{pos}_is_locked"] = True
                        lineup_dict[pos] = ''
                    lineup_dict["Points"] = 0
                    lineup_dict["TimeRemaining"] = 0
                    lineup_dict["ProjectedFpts"] = lineup_proj_fpts
                    lineup_dict["ProjectedStdDev"] = lineup_proj_stdv
                    lineup_dict['OriginalLineup'] = {'C': '', 'F': '', 'G': '', 'PF': '', 'PG': '', 'SF': '', 'SG': '',
                                                     'UTIL': ''},
                    lineup_dict['LockedSalary'] = 50000
                    lineup_dict['LockedPlayers'] = len(self.roster_construction)
                    lineup_dict['UnlockedPlayers'] = 0
                    total_minutes_for_full_lineup = len(self.roster_construction) * self.num_minutes_per_player
                    minutes_for_locked_players = total_minutes_for_full_lineup
                    minutes_used_by_actual_players = total_minutes_for_full_lineup
                    lineup_dict['Salary'] = 0
                    lineup_dict['SalaryRemaining'] = 0
                    lineup_dict["LockedPlayerMinutes"] = 0
                    lineup_dict['ProjectedFieldFpts'] = 0
                    lineup_dict['TotalMinutesForLineup'] = 0
                    lineup_dict["BayesianProjectedFpts"] = 0
                    lineup_dict["BayesianProjectedVar"] = 0
                    lineup_dict['EmptyLu'] = True
                    lineup_dict['UserLu'] = False
                    self.contest_lineups[str(row["EntryId"])] = lineup_dict

                else:
                    lineup_minutes_remaining = 0
                    extracted_players = self.extract_players(row["Lineup"], positions_order)
                    for pos, player_name in extracted_players.items():
                        locked_key = f"{pos}_is_locked"
                        if player_name == "LOCKED":
                            lineup_dict[locked_key] = False
                            lineup_dict[pos] = player_name
                            lineup_minutes_remaining += self.num_minutes_per_player
                            continue
                        else:
                            lineup_dict[locked_key] = True

                            for k, v in self.player_dict.items():
                                player_found = False

                                transformed_player_name = player_name

                                if v['Name'] == transformed_player_name and pos in v['Position']:
                                    lineup_proj_fpts += v['Fpts']
                                    lineup_salary += v['Salary']
                                    lineup_dict[pos] = v['ID']
                                    lineup_updated_proj += v['BayesianProjectedFpts']
                                    lineup_updated_var += v['BayesianProjectedVar']
                                    lineup_minutes_remaining += v['Minutes Remaining']
                                    lineup_proj_fieldfpts += v['fieldFpts']
                                    player_found = True
                                    break

                            if not player_found:
                                if transformed_player_name in self.missing_ids.keys():
                                    lineup_salary += self.missing_ids[transformed_player_name]['Salary']
                                    lineup_dict[pos] = self.missing_ids[transformed_player_name]['ID']
                                    self.player_dict[(
                                        transformed_player_name,
                                        str(self.missing_ids[transformed_player_name]['Position']),
                                        self.missing_ids[transformed_player_name]['Team'])] = {
                                        "Fpts": 0,
                                        "fieldFpts": 0,
                                        "Position": self.missing_ids[transformed_player_name]['Position'],
                                        "Name": player_name,
                                        "DK Name": player_name,
                                        "Matchup":
                                            self.time_remaining_dict[self.missing_ids[transformed_player_name]['Team']][
                                                'Matchup'],
                                        "Team": self.missing_ids[transformed_player_name]['Team'],
                                        "Opp":
                                            self.time_remaining_dict[self.missing_ids[transformed_player_name]['Team']][
                                                'Opp'],
                                        "ID": self.missing_ids[transformed_player_name]['ID'],
                                        "UniqueKey": self.missing_ids[transformed_player_name]['UniqueKey'],
                                        "Salary": self.missing_ids[transformed_player_name]['Salary'],
                                        "StdDev": 0,
                                        "Ceiling": 0,
                                        "Ownership": 0,
                                        "Correlations": {},
                                        "Player Correlations": {},
                                        "In Lineup": False,
                                        "Minutes": 0,
                                        "Minutes Remaining": 0,
                                        "BayesianProjectedFpts": 0,
                                        "BayesianProjectedVar": 0,
                                        "ActualFpts": 0,
                                        "GameLocked": True,
                                        "GameTime": None
                                    }
                                    self.teams_dict[self.missing_ids[transformed_player_name]['Team']].append(
                                        {
                                            "Fpts": 0,
                                            "fieldFpts": 0,
                                            "Position": self.missing_ids[transformed_player_name]['Position'],
                                            "Name": player_name,
                                            "DK Name": player_name,
                                            "Matchup": self.time_remaining_dict[
                                                self.missing_ids[transformed_player_name]['Team']]['Matchup'],
                                            "Team": self.missing_ids[transformed_player_name]['Team'],
                                            "Opp": self.time_remaining_dict[
                                                self.missing_ids[transformed_player_name]['Team']]['Opp'],
                                            "ID": self.missing_ids[transformed_player_name]['ID'],
                                            "UniqueKey": self.missing_ids[transformed_player_name]['UniqueKey'],
                                            "Salary": self.missing_ids[transformed_player_name]['Salary'],
                                            "StdDev": 0,
                                            "Ceiling": 0,
                                            "Ownership": 0,
                                            "Correlations": {},
                                            "Player Correlations": {},
                                            "In Lineup": False,
                                            "Minutes": 0,
                                            "Minutes Remaining": 0,
                                            "BayesianProjectedFpts": 0,
                                            "BayesianProjectedVar": 0,
                                            "ActualFpts": 0,
                                            "GameLocked": True,
                                            "GameTime": None
                                        }
                                    )
                                else:
                                    players_not_found.append(player_name)

                    lineup_dict["Points"] = float(row["Points"])
                    lineup_dict["TimeRemaining"] = lineup_minutes_remaining
                    lineup_dict["ProjectedFpts"] = lineup_proj_fpts
                    lineup_dict["ProjectedStdDev"] = lineup_proj_stdv
                    lineup_dict['OriginalLineup'] = extracted_players
                    actual_minutes_used = total_minutes_for_full_lineup - lineup_minutes_remaining
                    efficiency_factor = 0.1  # A constant that scales the uncertainty based on minutes remaining
                    minutes_proportion_remaining = lineup_minutes_remaining / total_minutes_for_full_lineup
                    lineup_dict["BayesianProjectedFpts"] = lineup_updated_proj
                    lineup_dict["BayesianProjectedVar"] = lineup_updated_var
                    minutes_for_locked_players = row["Lineup"].count("LOCKED") * self.num_minutes_per_player
                    lineup_dict['Salary'] = lineup_salary
                    lineup_dict['TotalMinutesForLineup'] = total_minutes_for_full_lineup
                    lineup_dict['LockedSalary'] = lineup_salary
                    lineup_dict['SalaryRemaining'] = self.max_salary - lineup_salary
                    lineup_dict['ProjectedFieldFpts'] = lineup_proj_fieldfpts
                    lineup_dict["LockedPlayerMinutes"] = minutes_for_locked_players
                    lineup_dict['UsedPlayerMinutes'] = actual_minutes_used
                    lineup_dict['UnlockedPlayers'] = row['Lineup'].count("LOCKED")
                    lineup_dict['LockedPlayers'] = len(self.roster_construction) - lineup_dict['UnlockedPlayers']
                    lineup_dict['EmptyLu'] = False
                    lineup_dict['UserLu'] = False
                    self.contest_lineups[str(row["EntryId"])] = lineup_dict
        random_keys = random.sample(list(self.contest_lineups.keys()), 5)
        self.num_lineups = len(self.contest_lineups)
        if len(players_not_found) > 0:
            print(f'Players not found: {set(players_not_found)}')
            for p in set(players_not_found):
                if p not in self.missing_ids.keys():
                    print(f'Missing player: {p}, missing id keys: {self.missing_ids.keys()}')
                else:
                    print(f'Found player: {self.missing_ids[p]}')


    def inspect_contest_lineups(self):
        import json

        print("Inspecting `self.contest_lineups`...")

        # Print the total number of lineups
        print(f"Total Lineups: {len(self.contest_lineups)}\n")

        # Print details of a few sample entries
        print("Sample Lineups:")
        for key in list(self.contest_lineups.keys())[:1]:  # Print up to 5 sample entries
            print(f"Entry ID: {key}")
            print("Lineup Details:")
            print(json.dumps(self.contest_lineups[key], indent=4))
            print("\n---\n")

        print("End of sample inspection.\n")

    def print_user_lineups(self):
        print("Lineups where 'UserLu' is True:")

        user_lineups = {key: value for key, value in self.contest_lineups.items() if value.get("UserLu")}

        if not user_lineups:
            print("No lineups with 'UserLu' set to True were found.")
        else:
            for entry_id, lineup in user_lineups.items():
                print(f"Entry ID: {entry_id}")
                for k, v in lineup.items():
                    print(f"  {k}: {v}")
                print("\n---\n")

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = "Name"
                player_name = row[name_key]
                team = row["TeamAbbrev"]
                position = str(row["Position"])
                if 'UTIL' not in position:
                    position = [pos for pos in row["Position"].split("/")]
                    position.sort()
                    if any(pos in ["PG", "SG"] for pos in position):
                        position.append("G")
                    if any(pos in ["SF", "PF"] for pos in position):
                        position.append("F")
                    position.append("UTIL")
                    #position = str(position)

                match = re.search(pattern=r"(\w{2,4}@\w{2,4})", string=row["Game Info"])
                team_opp = ''
                if match:
                    opp = match.groups()[0].split("@")
                    for m in opp:
                        if m != team:
                            team_opp = m
                    opp = tuple(opp)
                player_found = False
                for k, v in self.player_dict.items():
                    if player_name == v['Name']:
                        player_found = True
                        v["ID"] = str(row["ID"])
                        v["UniqueKey"] = str(row["ID"])
                if player_found == False:
                    self.missing_ids[player_name] = {'Position': position, 'Team': team, 'ID': str(row["ID"]),
                                                        "UniqueKey": str(row["ID"]), "Salary": int(row["Salary"])}

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.team_limits = self.config["team_limits"]
        self.global_team_limit = int(self.config["global_team_limit"])
        self.randomness_amount = float(self.config["randomness"])
        self.matchup_limits = self.config["matchup_limits"]
        self.matchup_at_least = self.config["matchup_at_least"]
        self.default_var = float(self.config["default_var"])
        self.max_pct_off_optimal = float(self.config['max_pct_off_optimal'])
        #self.projection_minimum = int(self.config["projection_minimum"])
        #self.min_salary = int(self.config["min_lineup_salary"])
        #self.projection_minimum = int(self.projection_minimum)
        #self.min_salary = int(self.min_salary)

    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"]
                try:
                    fpts = float(row["fpts"])
                except:
                    fpts = 0
                    print(
                        "unable to load player fpts: "
                        + player_name
                        + ", fpts:"
                        + row["fpts"]
                    )
                if "fieldfpts" in row:
                    if row["fieldfpts"] == "":
                        fieldFpts = fpts
                    else:
                        fieldFpts = float(row["fieldfpts"])
                else:
                    fieldFpts = fpts
                position = [pos for pos in row["position"].split("/")]
                position.sort()
                if any(pos in ["PG", "SG"] for pos in position):
                    position.append("G")
                if any(pos in ["SF", "PF"] for pos in position):
                    position.append("F")
                position.append("UTIL")
                pos = position[0]
                if "stddev" in row:
                    if row["stddev"] == "" or float(row["stddev"]) == 0:
                        stddev = fpts * self.default_var
                    else:
                        stddev = float(row["stddev"])
                else:
                    stddev = fpts * self.default_var
                if stddev == 0:
                    stddev = 0.001
                # check if ceiling exists in row columns
                if "ceiling" in row:
                    if row["ceiling"] == "" or float(row["ceiling"]) == 0:
                        ceil = fpts + stddev
                    else:
                        ceil = float(row["ceiling"])
                else:
                    ceil = fpts + stddev
                if row["salary"]:
                    sal = int(row["salary"].replace(",", ""))
                if "minutes" in row:
                    mins = row["minutes"]
                else:
                    mins = 0
                if pos == "PG":
                    corr = {
                        "PG": 1,
                        "SG": -0.066989,
                        "SF": -0.066989,
                        "PF": -0.066989,
                        "C": -0.043954,
                        "Opp PG": 0.020682,
                        "Opp SG": 0.020682,
                        "Opp SF": 0.015477,
                        "Opp PF": 0.015477,
                        "Opp C": 0.000866,
                    }
                elif pos == "SG":
                    corr = {
                        "PG": -0.066989,
                        "SG": 1,
                        "SF": -0.066989,
                        "PF": -0.066989,
                        "C": -0.043954,
                        "Opp PG": 0.020682,
                        "Opp SG": 0.020682,
                        "Opp SF": 0.015477,
                        "Opp PF": 0.015477,
                        "Opp C": 0.000866,
                    }
                elif pos == "SF":
                    corr = {
                        "PG": -0.066989,
                        "SG": -0.066989,
                        "SF": 1,
                        "PF": -0.002143,
                        "C": -0.082331,
                        "Opp PG": 0.015477,
                        "Opp SG": 0.015477,
                        "Opp SF": 0.015477,
                        "Opp PF": 0.015477,
                        "Opp C": -0.012331,
                    }
                elif pos == "PF":
                    corr = {
                        "PG": -0.066989,
                        "SG": -0.066989,
                        "SF": -0.002143,
                        "PF": 1,
                        "C": -0.082331,
                        "Opp PG": 0.015477,
                        "Opp SG": 0.015477,
                        "Opp SF": 0.015477,
                        "Opp PF": 0.015477,
                        "Opp C": -0.012331,
                    }
                elif pos == "C":
                    corr = {
                        "PG": -0.043954,
                        "SG": -0.043954,
                        "SF": -0.082331,
                        "PF": -0.082331,
                        "C": 1,
                        "Opp PG": 0.000866,
                        "Opp SG": 0.000866,
                        "Opp SF": -0.012331,
                        "Opp PF": -0.012331,
                        "Opp C": -0.073081,
                    }
                team = row["team"]
                try:
                    own = float(row["own%"].replace("%", ""))
                except:
                    own = 0
                if own == 0:
                    own = 0.1
                pos_str = str(position)
                player_data = {
                    "Fpts": fpts,
                    "fieldFpts": fieldFpts,
                    "Position": position,
                    "Name": player_name,
                    "DK Name": row["name"],
                    "Matchup": self.time_remaining_dict[team]['Matchup'],
                    "Team": team,
                    "Opp": self.time_remaining_dict[team]['Opp'],
                    "ID": "",
                    "UniqueKey": "",
                    "Salary": int(row["salary"].replace(",", "")),
                    "StdDev": stddev,
                    "Ceiling": ceil,
                    "Ownership": own,
                    "Correlations": corr,
                    "Player Correlations": {},
                    "In Lineup": False,
                    "Minutes": mins,
                    "Minutes Remaining": self.time_remaining_dict[team]['Minutes Remaining'],
                    "BayesianProjectedFpts": fpts,
                    "BayesianProjectedVar": stddev ** 2,
                    "ActualFpts": 0,
                    "GameLocked": self.time_remaining_dict[team]['GameLocked'],
                    "GameTime": self.time_remaining_dict[team]['GameTime']
                }

                # Print player name and GameLocked status

                # Check if player is in player_dict and get Opp, ID, Opp Pitcher ID and Opp Pitcher Name
                if (player_name, pos_str, team) in self.player_dict:
                    player_data["Opp"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("Opp", "")
                    player_data["ID"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("ID", "")


                self.player_dict[(player_name, pos_str, team)] = player_data
                self.teams_dict[team].append(
                    player_data
                )  # Add player data to their respective team



    def assign_unique_keys_to_contest_lineups(self):
        for entry_id, lineup_info in self.contest_lineups.items():
            # Generate a new unique key
            new_unique_key = str(uuid.uuid4())  # Example: 'a7b2c3d4-e5f6-7g8h-9i10-jk11lm12no13'
            lineup_info["unique_key"] = new_unique_key  # Add the unique key to the lineup

    # Load user lineups for late swap
    def load_player_lineups(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if row["entry id"] != "":
                    PG_id = re.search(r"\((\d+)\)", row["pg"]).group(1)
                    SG_id = re.search(r"\((\d+)\)", row["sg"]).group(1)
                    SF_id = re.search(r"\((\d+)\)", row["sf"]).group(1)
                    PF_id = re.search(r"\((\d+)\)", row["pf"]).group(1)
                    C_id = re.search(r"\((\d+)\)", row["c"]).group(1)
                    G_id = re.search(r"\((\d+)\)", row["g"]).group(1)
                    F_id = re.search(r"\((\d+)\)", row["f"]).group(1)
                    UTIL_id = re.search(r"\((\d+)\)", row["util"]).group(1)

                    if str(row['entry id']) in self.contest_lineups.keys():
                        lu = self.contest_lineups[str(row['entry id'])]
                        self.player_keys.append(str(row['entry id']))
                        lu['UserLu'] = True
                        lu['Type'] = 'user'

                        for pos in ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]:
                            if lu[pos + "_is_locked"] == False:
                                if pos == "PG":
                                    lu[pos] = PG_id
                                elif pos == "SG":
                                    lu[pos] = SG_id
                                elif pos == "SF":
                                    lu[pos] = SF_id
                                elif pos == "PF":
                                    lu[pos] = PF_id
                                elif pos == "C":
                                    lu[pos] = C_id
                                elif pos == "G":
                                    lu[pos] = G_id
                                elif pos == "F":
                                    lu[pos] = F_id
                                elif pos == "UTIL":
                                    lu[pos] = UTIL_id
                        # Create duplicates with different Type values
                        # self.contest_lineups[row['entry id']] = lu; # set already in load live
                        for i in range(1, self.lineup_sets):  # Create 4 duplicates
                            new_entry_id = f"{row['entry id']}_{i}"  # Ensure this ID is unique

                            lineup = lu.copy()  # Copy the original lineup to avoid overwriting
                            lineup['EntryId'] = new_entry_id
                            lineup['Type'] = f"user{i}"  # Assign a unique Type

                            self.contest_lineups[new_entry_id] = lineup

                            # Append both the original and new entry IDs to player_keys
                            self.player_keys.append(new_entry_id)
                    else:
                        print(f'Lineup {row["entry id"]} not found in contest file.')

        self.user_lineups = int(len(self.player_keys) / self.lineup_sets)
        self.print(f"Successfully loaded {len(self.player_keys)} lineups for {self.user_lineups} entries in late swap.")
        self.print(f"Total lineups in contest_lineups: {len(self.contest_lineups)}")

    def swaptimize(self):
        # Initialize a dictionary to hold lineups temporarily for each entry
        self.entry_lineups = {pk: [] for pk in self.player_keys}

        for pk in self.player_keys:
            lineup_obj = self.contest_lineups[pk]
            self.print(
                f"Swaptimizing lineup {pk}"
            )

            # Initialize salary backoff parameters
            original_min_salary = self.min_salary if self.min_salary is not None else 49000
            temp_min_salary = original_min_salary
            min_salary_floor = original_min_salary * 0.6  # 60% of original as floor
            backoff_factor = 0.95
            max_attempts = 4  # Maximum number of attempts with salary reduction
            solution_found = False

            # Before setting the minimum projected points constraint, calculate the total projection
            total_projection = 0
            for position in ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]:
                player_id = lineup_obj[position]
                # Find the player in player_dict and add their projection
                for player, attributes in self.player_dict.items():
                    if str(attributes["ID"]) == str(player_id):
                        total_projection += attributes["BayesianProjectedFpts"]

            # Set the minimum projected points using the total projection
            min_projected_points = total_projection * 0.98  # add to config - suggest lower values for contrarian / aggressive
            print(f"Minimum required projection: {min_projected_points:.2f}")

            while not solution_found and max_attempts > 0:

                problem = plp.LpProblem("NBA", plp.LpMaximize)
                lp_variables = {}
                for player, attributes in self.player_dict.items():
                    player_id = attributes["ID"]

                    for pos in attributes["Position"]:
                        lp_variables[(player, pos, player_id)] = plp.LpVariable(
                            name=f"{player}_{pos}_{player_id}", cat=plp.LpBinary
                        )

                # set the objective - maximize fpts & set randomness amount from config
                if self.randomness_amount != 0:
                    problem += (
                        plp.lpSum(
                            np.random.normal(
                                self.player_dict[player]["Fpts"],
                                (
                                        self.player_dict[player]["StdDev"]
                                        * self.randomness_amount
                                        / 100
                                ),
                            )
                            * lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                        ),
                        "Objective",
                    )
                else:
                    problem += (
                        plp.lpSum(
                            self.player_dict[player]["Fpts"]
                            * lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                        ),
                        "Objective",
                    )

                # Set the salary constraints
                max_salary = 50000

                # Maximum Salary Constraint
                problem += (
                    plp.lpSum(
                        self.player_dict[player]["Salary"]
                        * lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                    )
                    <= max_salary,
                    "Max Salary",
                )




                # Minimum Projected Points Constraint
                problem += (
                    plp.lpSum(
                        self.player_dict[player]["BayesianProjectedFpts"]  # Make sure "Fpts" matches your data column name
                        * lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                    )
                    >= min_projected_points,
                    "Min Projected Points",
                )



                # Minimum Salary Constraint (now uses temp_min_salary)
                problem += (
                    plp.lpSum(
                        self.player_dict[player]["Salary"]
                        * lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                    )
                    >= temp_min_salary,
                    "Min Salary",
                )

                # Must not play all 8 or 9 players from the same team (8 if dk, 9 if fd)
                for matchup in self.matchup_list:
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                            if attributes["Matchup"] == matchup
                        )
                        <= 8
                    )

                # Address limit rules if any
                for limit, groups in self.at_least.items():
                    for group in groups:
                        problem += (
                            plp.lpSum(
                                lp_variables[(player, pos, attributes["ID"])]
                                for player, attributes in self.player_dict.items()
                                for pos in attributes["Position"]
                                if attributes["Name"] in group
                            )
                            >= int(limit),
                            f"At least {limit} players {group}",
                        )

                for limit, groups in self.at_most.items():
                    for group in groups:
                        problem += (
                            plp.lpSum(
                                lp_variables[(player, pos, attributes["ID"])]
                                for player, attributes in self.player_dict.items()
                                for pos in attributes["Position"]
                                if attributes["Name"] in group
                            )
                            <= int(limit),
                            f"At most {limit} players {group}",
                        )

                for matchup, limit in self.matchup_limits.items():
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                            if attributes["Matchup"] == matchup
                        )
                        <= int(limit),
                        "At most {} players from {}".format(limit, matchup),
                    )

                for matchup, limit in self.matchup_at_least.items():
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                            if attributes["Matchup"] == matchup
                        )
                        >= int(limit),
                        "At least {} players from {}".format(limit, matchup),
                    )

                # Address team limits
                for teamIdent, limit in self.team_limits.items():
                    problem += plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                        if team == teamIdent
                    ) <= int(limit), "At most {} players from {}".format(limit, teamIdent)

                if self.global_team_limit is not None:
                    for teamIdent in self.team_list:
                        problem += (
                            plp.lpSum(
                                lp_variables[(player, pos, attributes["ID"])]
                                for player, attributes in self.player_dict.items()
                                for pos in attributes["Position"]
                                if attributes["Team"] == teamIdent
                            )
                            <= int(self.global_team_limit),
                            f"Global team limit - at most {self.global_team_limit} players from {teamIdent}",
                        )

                # Force players to be used if they are locked
                POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
                FORCE_PLAYERS = []

                # Identify forced players
                for position in POSITIONS:
                    if lineup_obj[position + "_is_locked"]:
                        player_id = lineup_obj[position]
                        for p_tuple, attributes in self.player_dict.items():
                            if str(attributes["ID"]) == str(player_id):
                                FORCE_PLAYERS.append((p_tuple, position, attributes["ID"]))

                # Create a set of forced player IDs for quick lookup
                forced_player_ids = {player_id for _, _, player_id in FORCE_PLAYERS}

                # Add constraints to force players
                for forced_player in FORCE_PLAYERS:
                    problem += (
                        lp_variables[forced_player] == 1,
                        f"Force player {forced_player}",
                    )

                # Exclude players who are locked AND not forced
                for player, attributes in self.player_dict.items():
                    player_id = attributes["ID"]
                    player_game_locked = attributes["GameLocked"]

                    for pos in attributes["Position"]:
                        variable_key = (player, pos, player_id)

                        if player_game_locked and player_id not in forced_player_ids:
                            problem += (
                                lp_variables[variable_key] == 0,
                                f"Exclude locked player {player} at {pos}",
                            )

                # Constraints for specific positions
                for pos in ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]:
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            if pos in attributes["Position"]
                        )
                        == 1,
                        f"Must have at 1 {pos}",
                    )

                # Constraint to ensure each player is only selected once
                for player in self.player_dict:
                    player_id = self.player_dict[player]["ID"]
                    problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, player_id)]
                            for pos in self.player_dict[player]["Position"]
                        )
                        <= 1,
                        f"Can only select {player} once",
                    )

                # Replace the lineup uniqueness constraint section with this:
                i = 0
                if self.output_lineups:  # Only add these constraints if we have previous lineups
                    for lineup, _ in self.output_lineups:
                        player_ids = [tpl[2] for tpl in lineup]
                        player_keys_to_exclude = []
                        for key, attr in self.player_dict.items():
                            if attr["ID"] in player_ids:
                                for pos in attr["Position"]:
                                    player_keys_to_exclude.append((key, pos, attr["ID"]))
                        problem += (
                            plp.lpSum(lp_variables[x] for x in player_keys_to_exclude)
                            <= len(lineup) - self.num_uniques,
                            f"Lineup {i}",
                        )
                        i += 1



                try:
                    problem.solve(plp.PULP_CBC_CMD(msg=0))
                    if plp.LpStatus[problem.status] == "Optimal":
                        # Solution found
                        selected_vars = [
                            player for player in lp_variables if lp_variables[player].varValue != 0
                        ]
                        for player in selected_vars:
                            lineup_obj[player[1]] = player[2]
                        self.output_lineups.append((selected_vars, lineup_obj))
                        count = len(self.output_lineups)
                        print(f"Swapped lineup : {count} (min salary: ${temp_min_salary:,})")
                        solution_found = True
                    else:
                        # No solution at this salary level, try lower
                        temp_min_salary = temp_min_salary * backoff_factor
                        max_attempts -= 1
                        #print(f"No solution found. Reducing minimum salary to ${temp_min_salary:,.0f}")

                        if temp_min_salary < min_salary_floor:
                            print(f"Hit minimum salary floor (${min_salary_floor:,}). Giving up on this lineup.")
                            break

                except plp.PulpSolverError:
                    print(f"Solver error at minimum salary ${temp_min_salary:,}")
                    temp_min_salary = temp_min_salary * backoff_factor
                    max_attempts -= 1
                    if temp_min_salary < min_salary_floor:
                        print(f"Hit minimum salary floor (${min_salary_floor:,}). Giving up on this lineup.")
                        break

            if solution_found:
                # Assuming this is in your results processing section where you're creating the optimal lineup
                optimal_lineup = []
                total_salary = 0
                total_points = 0  # Initialize total points counter
                for player, attributes in self.player_dict.items():
                    for pos in attributes["Position"]:
                        if (
                                lp_variables[(player, pos, attributes["ID"])].varValue == 1
                        ):  # If player is selected
                            optimal_lineup.append(
                                {
                                    "Position": pos,
                                    "Name": player,
                                    "Salary": attributes["Salary"],
                                    "BayesianProjectedFpts": attributes["BayesianProjectedFpts"],  # Make sure this matches your column name
                                    "ID": attributes["ID"],
                                }
                            )
                            total_salary += attributes["Salary"]
                            total_points += attributes["BayesianProjectedFpts"]  # Add player's points to total

                # Print the results
                self.print(f"\nOptimal Lineup: {i}")
                for player in optimal_lineup:
                    print(
                        f"{player['Position']}: {player['Name']} (Salary: ${player['Salary']})"
                    )
                self.print(f"\nTotal Salary: ${total_salary}")
                self.print(f"Total Projected Points: {total_points:.2f}")  # Print total points with 2 decimal places
            if not solution_found:
                print(f"Failed to find valid lineup for {pk} after all attempts")
                # Create a tuple with the original lineup structure
                original_lineup = []
                for position in ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]:
                    player_id = lineup_obj[position]
                    # Find the corresponding player tuple
                    for player, attributes in self.player_dict.items():
                        if str(attributes["ID"]) == str(player_id):
                            original_lineup.append((player, position, player_id))

                # Add the original lineup to output_lineups
                self.output_lineups.append((original_lineup, lineup_obj))
                self.print(f"Retained original lineup: {i} ")

                if len(self.output_lineups) == 0:
                    print("No valid lineups found at all - stopping process")
                    break


    def convert_player_dict_to_pid_keys(self):
        self.player_dict = {v['ID']: v for v in self.player_dict.values()}

    @staticmethod
    def _generate_lineups_wrapper(args):
        """Static wrapper method to unpack arguments for generate_lineups"""
        return NBA_Swaptimizer_Sims.generate_lineups(*args)

    def compute_best_guesses_parallel(self):
        self.convert_player_dict_to_pid_keys()
        self.first_idx = list(self.contest_lineups.keys())[0]
        start_time = time.time()


        print("Starting parallel processing setup...")
        start_time = time.time()

        # Pre-process player data
        print("Processing player data...")
        players = [(k, self.player_dict[k]) for k in self.player_dict.keys()
                   if self.player_dict[k].get('GameLocked', True) == False]

        ids = []
        ownership = []
        salaries = []
        projections = []
        positions = []
        teams = []
        matchups = []

        print(f"Pre-processing {len(players)} players...")
        for k, player in players:
            if "Team" not in player:
                print(f"{player['Name']} name mismatch between projections and player ids!")
                continue

            ids.append(player["UniqueKey"])
            ownership.append(player["Ownership"])
            salaries.append(player["Salary"])
            projections.append(player["fieldFpts"] if player["fieldFpts"] >= self.projection_minimum else 0)
            teams.append(player["Team"])
            matchups.append(player["Matchup"])
            pos_list = []
            for pos in self.roster_construction:
                if pos in player["Position"]:
                    pos_list.append(1)
                else:
                    pos_list.append(0)
            positions.append(np.array(pos_list))

        print(f"Completed initial data processing in {time.time() - start_time:.1f} seconds")

        # Convert to numpy arrays
        print("Converting to numpy arrays...")
        ids = np.array(ids)
        ownership = np.array(ownership)
        salaries = np.array(salaries)
        projections = np.array(projections)
        pos_matrix = np.array(positions)
        teams = np.array(teams)
        matchups = np.array(matchups)

        # Prepare problems list with lineup data
        problems = []
        for key, lineup in self.contest_lineups.items():
            lu_tuple = (
                key, lineup, ids, np.zeros(shape=len(ids)),
                pos_matrix, ownership, self.min_salary, self.max_salary,
                self.optimal_score, salaries, projections,
                self.max_pct_off_optimal, teams, matchups,
                len(self.roster_construction), self.site,
                self.roster_construction,
                {pos: i for i, pos in enumerate(self.roster_construction)},
                np.min(salaries) if len(salaries) > 0 else 0
            )
            problems.append(lu_tuple)



        total_problems = len(problems)
        print(f"\nStarting parallel processing of {total_problems:,} lineups")
        completed_lineups = 0
        results = []
        backoff_count = 0

        def update_progress(result):
            nonlocal completed_lineups, backoff_count
            completed_lineups += 1

            if isinstance(result, str) and "Backoff triggered" in result:
                backoff_count += 1
                return

            results.append(result)

            if completed_lineups % (total_problems // 10) == 0:  # Update every 10%
                elapsed = time.time() - start_time
                progress_pct = (completed_lineups / total_problems) * 100
                speed = completed_lineups / elapsed if elapsed > 0 else 0
                eta = (total_problems - completed_lineups) / speed if speed > 0 else 0

                self.print(f"\nProgress: {completed_lineups:,}/{total_problems:,} lineups ({progress_pct:.1f}%)")
                print(f"Speed: {speed:.1f} lineups/sec, ETA: {int(eta)} seconds")
                if backoff_count > 0:
                    print(f"Backoffs since last update: {backoff_count}")
                    backoff_count = 0

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            tasks = [pool.apply_async(self._generate_lineups_wrapper, args=(prob,), callback=update_progress)
                     for prob in problems]

            # Wait for all tasks to complete
            for task in tasks:
                task.wait()

        print(f"\nParallel processing complete. Total results: {len(results)}")

        # Process results
        processed_keys = [result[0] for result in results if isinstance(result, tuple)]
        print(f"Successfully processed {len(processed_keys)} lineup keys")

        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")


        # New function to get names from IDs in a lineup
        def get_names_from_ids(lineup):
            new_pos_dict = {}
            for pos in self.roster_construction:
                new_pos_dict[pos] = self.player_dict[lineup[pos]]['Name']
            return new_pos_dict

        bad_lu_count = 0
        bad_lus = []
        good_lu_count = 0
        minutes_count = {}  # Dictionary to keep track of count by UsedPlayerMinutes

        for r in results:
            for p in self.roster_construction:
                try:
                    if r[p] == 'LOCKED':
                        bad_lu_count += 1
                        if r['UsedPlayerMinutes'] in minutes_count:
                            minutes_count[r['UsedPlayerMinutes']] += 1
                        else:
                            minutes_count[r['UsedPlayerMinutes']] = 1
                except KeyError:
                    print('cant find pos in lineup')
                    print(r)
            else:
                good_lu_count += 1

        print(f'bad lineups: {bad_lu_count}, good lineups: {good_lu_count}')
        #end = time.time()
        # Assuming results is a list of tuples like the one you provided
        self.contest_lineups = {lineup['EntryId']: lineup for lineup in results}
        self.count_lineups_and_extract_fields()

    @staticmethod
    def generate_lineups(
            key,
            lineup,
            ids,
            in_lineup,
            pos_matrix,
            ownership,
            salary_floor,
            salary_ceiling,
            optimal_score,
            salaries,
            projections,
            max_pct_off_optimal,
            teams,
            matchups,
            num_players_in_roster,
            site,
            roster_construction,
            pos_index_dict,
            player_salary_floor
    ):

        rng = np.random.default_rng()  # Use default_rng for a more modern RNG
        lineup_copy = lineup.copy()
        iteration_count = 0
        total_players = num_players_in_roster
        reasonable_projection = optimal_score - (max_pct_off_optimal * optimal_score)
        max_players_per_team = None
        max_attempts = 10  # set a maximum number of attempts before reducing floors
        player_teams = []
        lineup_matchups = []
        backoff_factor = 0.98  # You can adjust this to change how much you back off each time
        min_salary_floor = 1  # Define the minimum salary floor
        min_reasonable_projection = 1  # Define the minimum reasonable projection

        def is_valid_lineup(salary, proj, player_teams):
            # Helper function to determine if the current lineup is valid
            if not (salary_floor <= salary <= salary_ceiling):
                return False
            if proj < reasonable_projection:
                return False
            if max_players_per_team is not None:
                if any(count > max_players_per_team for count in Counter(player_teams).values()):
                    return False
            return True

        def finalize_lineup(lineup, salary, proj, unlocked_proj):
            # Helper function to finalize the lineup before returning
            lineup['Salary'] = int(salary) if np.isscalar(salary) else int(salary[0])
            lineup['ProjectedFieldFpts'] = proj if np.isscalar(proj) else proj[0]
            lineup['UnlockedFieldFpts'] = unlocked_proj if np.isscalar(unlocked_proj) else unlocked_proj[0]
            return lineup

        def reset_lineup():
            # Helper function to reset lineup to the initial state
            nonlocal salary, proj, unlocked_proj, player_teams, lineup_matchups, in_lineup, players_remaining
            salary = lineup_copy['LockedSalary']
            proj = lineup_copy['ProjectedFieldFpts']
            unlocked_proj = 0
            player_teams = []
            lineup_matchups = []
            in_lineup.fill(0)
            players_remaining = lineup_copy['UnlockedPlayers']
            return lineup_copy.copy()

        def log_lineup_state(action, state):
            print(f"{action} - Lineup State for {key}: {state}")

        # Start of lineup generation logic
        if lineup['UserLu'] or lineup['EmptyLu'] or lineup['UnlockedPlayers'] == 0:
            return lineup.copy()
        while True:
            lineup = reset_lineup()
            for attempt in range(max_attempts):
                for position in roster_construction:
                    if lineup[f"{position}_is_locked"]:
                        continue  # Skip locked positions
                    k = pos_index_dict[position]
                    pos = pos_matrix[:, k]
                    remaining_salary = salary_ceiling - salary
                    max_allowable_salary = (remaining_salary - (player_salary_floor * (players_remaining - 1)))

                    valid_players = np.nonzero(
                        (pos > 0) &
                        (in_lineup == 0) &
                        (
                                salaries <= max_allowable_salary) &  # Players must be affordable within the max allowable salary
                        (salaries >= player_salary_floor)  # Players must meet the minimum salary requirement
                    )[0]
                    if players_remaining == 1:
                        valid_players = np.nonzero(
                            (pos > 0)
                            & (in_lineup == 0)
                            & (salaries <= remaining_salary)
                            & (salary + salaries >= salary_floor)
                            & (salaries <= max_allowable_salary)
                        )[0]
                    else:
                        valid_players = np.nonzero(
                            (pos > 0)
                            & (in_lineup == 0)
                            & (salaries <= remaining_salary)
                            & (salaries <= max_allowable_salary)
                        )[0]
                    if not valid_players.size:
                        lineup = reset_lineup()  # No valid players, reset lineup
                        break  # Break out of the position loop to restart
                    # grab names of players eligible
                    plyr_list = ids[valid_players]
                    # create np array of probability of being seelcted based on ownership and who is eligible at the position
                    prob_list = ownership[valid_players]
                    prob_list = prob_list / prob_list.sum()
                    if players_remaining == 1:
                        boosted_salaries = np.array(
                            [
                                salary_boost(s, salary_ceiling)
                                for s in salaries[valid_players]
                            ]
                        )
                        boosted_probabilities = prob_list * boosted_salaries
                        boosted_probabilities /= (
                            boosted_probabilities.sum()
                        )  # normalize to ensure it sums to 1
                    try:
                        if players_remaining == 1:
                            choice = rng.choice(plyr_list, p=boosted_probabilities)
                        else:
                            choice = rng.choice(plyr_list, p=prob_list)
                    except:
                        lineup = reset_lineup()
                        continue  # Skip to the next iteration of the while loop
                    players_remaining -= 1
                    choice_idx = np.nonzero(ids == choice)[0]
                    lineup[position] = str(choice)  # Adding player
                    in_lineup[choice_idx] = 1
                    salary += salaries[choice_idx]
                    proj += projections[choice_idx]
                    unlocked_proj += projections[choice_idx]
                    player_teams.append(teams[choice_idx][0])
                    lineup_matchups.append(matchups[choice_idx[0]])
                if players_remaining == 0:
                    # Check if the lineup is valid after all positions are filled
                    if is_valid_lineup(salary, proj, player_teams):
                        return finalize_lineup(lineup, salary, proj,
                                               unlocked_proj)  # If lineup is valid, finalize and return
                lineup = reset_lineup()

            # Backoff strategy if max_attempts is reached without a valid lineup
            print(f"Backoff triggered ... New salary floor: {salary_floor}, new projection: {reasonable_projection}")

            if salary_floor > min_salary_floor:
                salary_floor *= backoff_factor
            else:
                salary_floor = min_salary_floor  # Ensure not to go below min

            if reasonable_projection > min_reasonable_projection:
                reasonable_projection *= backoff_factor
            else:
                reasonable_projection = min_reasonable_projection  # Ensure not to go below min
            # Conditions to stop trying if it gets too low
            if salary_floor == min_salary_floor and reasonable_projection == min_reasonable_projection:
                print(f"Minimum thresholds reached without a valid lineup for key {key}.")
                return lineup  # Return the best attempt or a lineup indicating failure

    def calc_gamma(self, mean, sd):
        alpha = (mean / sd) ** 2
        beta = sd ** 2 / mean
        return alpha, beta

    def count_lineups_and_extract_fields(self):
        for entry_id, lineup_info in self.contest_lineups.items():
            try:
                actual_lineup_list = [lineup_info[pos] for pos in self.roster_construction]
            except:
                print(lineup_info)

            lineup_set = frozenset(actual_lineup_list)

            # If this is the first time we see this lineup, initialize it
            if lineup_set not in self.field_lineups:
                self.field_lineups[lineup_set] = {
                    'Count': 0,  # Start at 0 since we'll increment below
                    'BayesianProjectedFpts': lineup_info['BayesianProjectedFpts'],
                    'BayesianProjectedVar': lineup_info['BayesianProjectedVar'],
                    'Lineup': actual_lineup_list,
                    'EntryIds': [],
                    'ROI': 0,
                    'Wins': 0,
                    'Top1Percent': 0,
                    'Cashes': 0
                }

            # Always increment count and append entry_id, regardless if it's new or duplicate
            self.field_lineups[lineup_set]['Count'] += 1
            self.field_lineups[lineup_set]['EntryIds'].append(entry_id)


    @staticmethod
    def run_simulation_for_game(
            team1_id,
            team1,
            team2_id,
            team2,
            num_iterations,
            roster_construction,
            time_remaining_dict
    ):
        # Define correlations between positions
        if time_remaining_dict[team1_id]['Minutes Remaining'] == 0:
            game = team1 + team2
            temp_fpts_dict = {}
            for i, player in enumerate(game):
                temp_fpts_dict[player["ID"]] = np.full(num_iterations, player["BayesianProjectedFpts"])
                # If time remaining is zero, we set the player's fantasy points to their BayesianProjectedFpts
        else:
            def get_corr_value(player1, player2):
                # First, check for specific player-to-player correlations
                if player2["Name"] in player1.get("Player Correlations", {}):
                    return player1["Player Correlations"][player2["Name"]]

                # If no specific correlation is found, proceed with the general logic
                position_correlations = {
                    "PG": -0.1324,
                    "SG": -0.1324,
                    "SF": -0.0812,
                    "PF": -0.0812,
                    "C": -0.1231,
                }

                if player1["Team"] == player2["Team"] and player1["Position"][0] in [
                    "PG",
                    "SG",
                    "SF",
                    "PF",
                    "C",
                ]:
                    primary_position = player1["Position"][0]
                    return position_correlations[primary_position]

                if player1["Team"] != player2["Team"]:
                    player_2_pos = "Opp " + str(player2["Position"][0])
                else:
                    player_2_pos = player2["Position"][0]

                return player1["Correlations"].get(
                    player_2_pos, 0
                )  # Default to 0 if no correlation is found

            def build_covariance_matrix(players):
                N = len(players)
                matrix = [[0 for _ in range(N)] for _ in range(N)]
                corr_matrix = [[0 for _ in range(N)] for _ in range(N)]

                for i in range(N):
                    for j in range(N):
                        if i == j:
                            matrix[i][j] = (
                                players[i]["BayesianProjectedVar"]
                            )  # Variance on the diagonal
                            corr_matrix[i][j] = 1
                        else:
                            matrix[i][j] = (
                                    get_corr_value(players[i], players[j])
                                    * players[i]["StdDev"]
                                    * players[j]["StdDev"]
                            )
                            corr_matrix[i][j] = get_corr_value(players[i], players[j])
                return matrix, corr_matrix

            def ensure_positive_semidefinite(matrix):
                eigs = np.linalg.eigvals(matrix)
                if np.any(eigs < 0):
                    jitter = abs(min(eigs)) + 1e-6  # a small value
                    matrix += np.eye(len(matrix)) * jitter
                return matrix

            game = team1 + team2
            covariance_matrix, corr_matrix = build_covariance_matrix(game)
            corr_matrix = np.array(corr_matrix)

            covariance_matrix, corr_matrix = build_covariance_matrix(game)
            covariance_matrix = np.array(corr_matrix)

            if covariance_matrix.ndim != 2 or covariance_matrix.shape == (0,):
                print(
                    f"Simulation skipped for game between {team1_id} and {team2_id} due to invalid covariance matrix shape: {covariance_matrix.shape}")
                return {}

            # Given eigenvalues and eigenvectors from previous code
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

            # Set negative eigenvalues to zero
            eigenvalues[eigenvalues < 0] = 0

            # Reconstruct the matrix
            covariance_matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)

            try:
                samples = multivariate_normal.rvs(
                    mean=[player["BayesianProjectedFpts"] for player in game],
                    cov=covariance_matrix,
                    size=num_iterations,
                )
            except:
                print(team1_id, team2_id, "bad matrix", covariance_matrix)


            player_samples = []
            for i, player in enumerate(game):
                sample = samples[:, i]
                player_samples.append(sample)

            temp_fpts_dict = {}

            for i, player in enumerate(game):
                temp_fpts_dict[player["ID"]] = player_samples[i]

        return temp_fpts_dict

    @staticmethod
    @jit(nopython=True)
    def calculate_payouts(args):
        (
            ranks,
            payout_array,
            entry_fee,
            field_lineup_keys,
            use_contest_data,
            field_lineups_count,
        ) = args
        num_lineups = len(field_lineup_keys)
        combined_result_array = np.zeros(num_lineups)

        payout_cumsum = np.cumsum(payout_array)

        for r in range(ranks.shape[1]):
            ranks_in_sim = ranks[:, r]
            payout_index = 0
            for lineup_index in ranks_in_sim:
                lineup_count = field_lineups_count[lineup_index]

                if payout_index >= len(payout_cumsum):
                    # Beyond payout positions, just accumulate entry fee loss
                    prize_for_lineup = -entry_fee
                else:
                    # Calculate prize while avoiding division by very small numbers
                    if payout_index != 0:
                        cumsum_diff = (
                                payout_cumsum[min(payout_index + lineup_count - 1, len(payout_cumsum) - 1)] -
                                payout_cumsum[payout_index - 1]
                        )
                    else:
                        cumsum_diff = payout_cumsum[min(payout_index + lineup_count - 1, len(payout_cumsum) - 1)]

                    # Use a safe division that caps the result
                    MAX_VALUE = 1e6  # Cap at million-dollar ROI
                    if lineup_count > 0:
                        prize_for_lineup = max(min(cumsum_diff / lineup_count, MAX_VALUE), -MAX_VALUE)
                    else:
                        prize_for_lineup = 0

                combined_result_array[lineup_index] += prize_for_lineup
                payout_index += lineup_count

        return combined_result_array

    def print(self, *args, **kwargs):
        """Override to allow progress capturing"""
        if self.is_subprocess:
            # When running as subprocess, just use regular print
            print(*args, **kwargs)
        else:
            # When running in GUI, use the GUI's print method
            print(*args, **kwargs)  # Or however you want to handle GUI printing

    def run_tournament_simulation(self):
        start_time = time.time()
        print(f"\nStarting Tournament Simulation")
        print(f"Configuration:")
        print(f"- Simulations: {self.num_iterations:,}")
        print(f"- Field lineups: {len(self.field_lineups.keys()):,}")
        print(f"- Matchups: {self.matchups}")

        # Initialize lineup_to_int mapping at the start
        self.lineup_to_int = {lineup: index for index, lineup in enumerate(self.field_lineups.keys())}
        temp_fpts_dict = {}

        # Game Simulation Phase with callbacks
        if len(self.matchups) > 0:
            game_simulation_params = []
            for m in self.matchups:
                game_simulation_params.append(
                    (m[0], self.teams_dict[m[0]], m[1], self.teams_dict[m[1]],
                     self.num_iterations, self.roster_construction, self.time_remaining_dict)
                )

            total_games = len(game_simulation_params)
            self.print(f"\nPhase 1: Starting simulations for {total_games} games...")
            completed_games = 0
            results = []

            def update_progress(result):
                nonlocal completed_games
                completed_games += 1
                elapsed = time.time() - start_time
                speed = completed_games / elapsed if elapsed > 0 else 0
                eta = (total_games - completed_games) / speed if speed > 0 else 0

                print(
                    f"Games processed: {completed_games}/{total_games} ({(completed_games / total_games) * 100:.1f}%)")
                print(f"Speed: {speed:.1f} games/sec, ETA: {int(eta)} seconds")
                results.append(result)

            with multiprocessing.Pool() as pool:
                tasks = [pool.apply_async(self.run_simulation_for_game, args=params, callback=update_progress)
                         for params in game_simulation_params]

                # Wait for all tasks to complete
                for task in tasks:
                    task.wait()

            # Update temp_fpts_dict with results
            for result in results:
                temp_fpts_dict.update(result)

        # [Rest of the simulation code remains the same...]
        # Lineup Processing Phase
        self.print(f"\nPhase 2: Fantasy Points Calculation")
        field_lineups_count = np.array(
            [self.field_lineups[idx]["Count"] for idx in self.field_lineups.keys()]
        )

        fpts_array = np.zeros((len(self.field_lineups), self.num_iterations))
        total_lineups = len(self.field_lineups)
        process_start = time.time()

        for index, (keys, values) in enumerate(self.field_lineups.items()):
            if index % max(1, total_lineups // 10) == 0:  # Update every 10%
                elapsed = time.time() - process_start
                speed = (index + 1) / elapsed if elapsed > 0 else 0
                eta = (total_lineups - (index + 1)) / speed if speed > 0 else 0
                self.print(
                    f"Processing lineups: {index + 1:,}/{total_lineups:,} ({(index + 1) / total_lineups * 100:.1f}%)")
                print(f"Speed: {speed:.1f} lineups/sec, ETA: {int(eta)} seconds")

            fpts_sim = sum([temp_fpts_dict[player] for player in values["Lineup"]
                            if player in temp_fpts_dict])
            fpts_array[index] = fpts_sim

        # Rankings Computation Phase
        self.print(f"\nPhase 3: Rankings Computation")
        print("Converting array to float16...")
        fpts_array = fpts_array.astype(np.float16)

        chunk_size = 1000
        total_chunks = (self.num_iterations + chunk_size - 1) // chunk_size
        print(f"Processing {total_chunks} ranking chunks...")
        ranks_list = []
        chunk_start = time.time()

        for i in range(0, self.num_iterations, chunk_size):
            end_idx = min(i + chunk_size, self.num_iterations)
            chunk_ranks = np.argsort(-fpts_array[:, i:end_idx], axis=0).astype(np.uint32)
            ranks_list.append(chunk_ranks)

            chunk_num = (i // chunk_size) + 1
            elapsed = time.time() - chunk_start
            speed = chunk_num / elapsed if elapsed > 0 else 0
            eta = (total_chunks - chunk_num) / speed if speed > 0 else 0

            self.print(f"Ranking chunks: {chunk_num}/{total_chunks} ({(chunk_num / total_chunks) * 100:.1f}%)")
            print(f"Speed: {speed:.1f} chunks/sec, ETA: {int(eta)} seconds")

        self.print("\nPhase 4: Statistics Calculation")
        print("Combining ranking results...")
        ranks = np.concatenate(ranks_list, axis=1)

        self.print("Computing win statistics...")
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)

        self.print("Computing cash statistics...")
        cashes, cash_counts = np.unique(
            ranks[0:len(list(self.payout_structure.values()))], return_counts=True
        )

        self.print("Computing top 1% statistics...")
        top1pct_cutoff = math.ceil(0.01 * len(self.field_lineups))
        top1pct, top1pct_counts = np.unique(
            ranks[0:top1pct_cutoff, :], return_counts=True
        )

        # ROI Calculations Phase
        self.print("\nPhase 5: ROI Calculations")
        payout_array = np.array(list(self.payout_structure.values()))
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))

        field_lineups_keys_array = np.array([self.lineup_to_int[lineup] for lineup in self.field_lineups.keys()])
        chunk_size = max(1000, self.num_iterations // 16)

        simulation_chunks = []
        for i in range(0, self.num_iterations, chunk_size):
            end_idx = min(i + chunk_size, self.num_iterations)
            try:
                chunk_data = (
                    ranks[:, i:end_idx].copy(),
                    payout_array,
                    self.entry_fee,
                    field_lineups_keys_array,
                    self.use_contest_data,
                    field_lineups_count,
                )
                simulation_chunks.append(chunk_data)
            except Exception as e:
                print(f"Error creating chunk {i}-{end_idx}: {e}")
                continue


        total_roi_chunks = len(simulation_chunks)
        self.print(f"Processing {total_roi_chunks} ROI chunks...")
        roi_start = time.time()
        results = []
        failed_chunks = []

        with multiprocessing.Pool() as pool:
            # Submit all tasks
            async_results = [pool.apply_async(self.calculate_payouts, (chunk,))
                             for chunk in simulation_chunks]

            # Process results with timeout
            for i, result in enumerate(async_results, 1):
                try:
                    # Add 5-minute timeout per chunk
                    chunk_result = result.get(timeout=60)
                    results.append(chunk_result)

                    # Progress reporting
                    elapsed = time.time() - roi_start
                    speed = i / elapsed if elapsed > 0 else 0
                    eta = (total_roi_chunks - i) / speed if speed > 0 else 0

                    self.print(f"ROI chunks: {i}/{total_roi_chunks} ({(i / total_roi_chunks) * 100:.1f}%)")
                    print(f"Speed: {speed:.1f} chunks/sec, ETA: {int(eta)} seconds")

                except multiprocessing.TimeoutError:
                    print(f"Warning: Chunk {i} timed out after 300 seconds")
                    failed_chunks.append(i)
                    # Instead of None, create a zero array of the correct shape
                    if results:  # If we have at least one successful result to reference
                        chunk_result = np.zeros_like(results[0])
                    else:  # If this is the first chunk, we need to determine the shape
                        chunk_result = np.zeros(len(field_lineups_keys_array))
                    results.append(chunk_result)
                except Exception as e:
                    print(f"Error processing chunk {i}: {e}")
                    failed_chunks.append(i)
                    # Same as above for error cases
                    if results:
                        chunk_result = np.zeros_like(results[0])
                    else:
                        chunk_result = np.zeros(len(field_lineups_keys_array))
                    results.append(chunk_result)

        if failed_chunks:
            print(f"Warning: {len(failed_chunks)} chunks failed: {failed_chunks}")
            # If too many chunks failed, you might want to raise an error
            if len(failed_chunks) > total_roi_chunks // 2:  # If more than 50% failed
                raise RuntimeError(f"Too many chunks failed ({len(failed_chunks)} out of {total_roi_chunks})")

        # Now the results array should be homogeneous and safe to sum
        self.print("\nPhase 6: Final Results Processing")
        combined_result_array = np.sum(results, axis=0)

        total_sum = 0
        index_to_key = list(self.field_lineups.keys())

        print("Updating lineup statistics...")
        total_lineups = len(self.field_lineups)
        for idx, roi in enumerate(combined_result_array):
            if idx % max(1, total_lineups // 10) == 0:  # Update every 10%
                print(
                    f"Processing lineup stats: {idx + 1:,}/{total_lineups:,} ({(idx + 1) / total_lineups * 100:.1f}%)")

            lineup_key = index_to_key[idx]
            lineup_count = self.field_lineups[lineup_key]["Count"]
            total_sum += roi * lineup_count
            self.field_lineups[lineup_key]["ROI"] += roi

        for i, lineup_key in enumerate(self.field_lineups.keys()):
            if i % max(1, total_lineups // 10) == 0:  # Update every 10%
                print(f"Finalizing stats: {i + 1:,}/{total_lineups:,} ({(i + 1) / total_lineups * 100:.1f}%)")

            lineup_int_key = self.lineup_to_int[lineup_key]

            if lineup_int_key in wins:
                win_index = np.where(wins == lineup_int_key)[0][0]
                self.field_lineups[lineup_key]["Wins"] += win_counts[win_index]

            if lineup_int_key in top1pct:
                top1pct_index = np.where(top1pct == lineup_int_key)[0][0]
                self.field_lineups[lineup_key]["Top1Percent"] += top1pct_counts[top1pct_index]

            if lineup_int_key in cashes:
                cash_index = np.where(cashes == lineup_int_key)[0][0]
                self.field_lineups[lineup_key]["Cashes"] += cash_counts[cash_index]

        total_time = time.time() - start_time
        print(f"\nSimulation Complete!")
        self.print(f"Total time: {total_time:.2f} seconds")
        print("Preparing output...")

    def format_lineup_table(self, lineup_data, lineup_info):
        """Format lineup information into a readable table"""
        try:
            from tabulate import tabulate
            has_tabulate = True
        except ImportError:
            has_tabulate = False

        # Collect player data
        players_data = []
        total_salary = 0
        total_ownership = 0
        total_points = 0

        for pos in self.roster_construction:
            player_id = lineup_info[pos]
            if player_id == '':
                players_data.append(['EMPTY', pos, 'N/A', 0, 0.0])
                continue

            for v in self.player_dict.values():
                if v["ID"] == player_id:
                    name = v["DK Name"].replace('#', '-')
                    team = v["Team"]
                    salary = v["Salary"]
                    ownership = v["Ownership"]
                    points = v["BayesianProjectedFpts"]
                    total_salary += salary
                    total_ownership += ownership
                    total_points += points
                    players_data.append([name, pos, team, salary, f" {ownership:.1f}", f" {points:.1f}"] )
                    break

        # Format header information
        header_info = [
            f"User: {lineup_info['User']}",
            f"Lineup Type: {lineup_info['Type']}",
            f"Projected: {lineup_info['BayesianProjectedFpts']:.1f}",
            f"Total Salary: ${total_salary:,}",
            f"Total Ownership: {total_ownership:.1f}% "
            f"Total Points: {total_points:.1f}"
        ]

        # Create player table
        player_table = tabulate(
            players_data,
            headers=["Player", "Pos", "Team", "Salary", "Own%", "Points"],
            tablefmt="grid"
        )

        # Format simulation results
        sim_info = [
            f"Cash Rate: {lineup_data['Cashes'] / self.num_iterations * 100:.1f}%",
            #f"Top 1% Rate: {lineup_data['Top1Percent'] / self.num_iterations * 100:.1f}%",
            f"ROI: {lineup_data['ROI'] / self.num_iterations:.2f}%",
            #f"Simulated Duplicates: {lineup_data['Count']}"
        ]

        # Combine all components
        output = "\n".join([
            "=" * 60,
            "\n".join(header_info),
            #"-" * 60,
            #player_table,
            #"-" * 60,
            #"\n".join(sim_info),
            #"=" * 60,
            ""  # Empty line for spacing
        ])

        return output


    def display_lineup_details(self, index, lineup_data):
        """Display formatted lineup details during processing"""
        if lineup_data.get('Type', '').startswith('user'):
            table = self.format_lineup_table(
                self.field_lineups[index],
                lineup_data
            )
            print(table)

    def output(self):
        start_time = time.time()
        print("\nStarting Output Generation...")

        # Phase 1: Process Lineup Data
        print("\nPhase 1: Processing Lineup Data")
        unique = {}
        total_lineups = len(self.field_lineups)

        for idx, (index, y) in enumerate(self.field_lineups.items()):
            if (idx + 1) % max(1, total_lineups // 10) == 0:  # Update every 10%
                print(
                    f"Processing lineups: {idx + 1:,}/{total_lineups:,} ({((idx + 1) / total_lineups) * 100:.1f}%)")

            lu_idx = self.lineup_to_int[index]

            if lu_idx is None:
                print(f"Warning: Lineup index {index} not found in lineup_to_int.")
                continue

            for entry in y['EntryIds']:
                lineup_info = self.contest_lineups[entry]
                if lineup_info["Type"].startswith("user"):
                    # Display formatted table for this lineup
                    self.display_lineup_details(index, lineup_info)
                    # Process lineup for output file
                    lineup_str = self.process_lineup_details(lineup_info, y, lu_idx, entry)
                    if lineup_str:
                        # Use combination of index and entry as key to preserve duplicates
                        unique[f"{index}_{entry}"] = lineup_str

        # Phase 2: Sort and Rearrange Lineups
        print("\nPhase 2: Sorting and Rearranging Lineups")
        sorted_unique = sorted(
            unique.items(),
            key=lambda x: (
                -float(x[1].split(",")[13].replace("%", "")),  # Primary sort by ROI (descending)
                -float(x[1].split(",")[9]),  # Secondary sort by Ceiling (descending)
                str(x[1].split(",")[-1])[:10]  # Keep original third sort criteria
            )
        )

        # Rest of the code remains the same...


        num_sets = self.lineup_sets
        rearranged_unique = self.rearrange_lineups(sorted_unique, num_sets)

        # Phase 3: Write Main Output File
        print("\nPhase 3: Writing Main Output File")
        self.write_main_output(rearranged_unique)

        # Additional phase for all lineups
        print("\nAdditional Phase: Writing All Lineups Output")
        self.write_all_lineups_output()

        # Phase 4: Process Player Exposure
        print("\nPhase 4: Processing Player Exposure")
        self.write_player_exposure()

        # Phase 5: Process Late Swap Data
        print("\nPhase 5: Processing Late Swap Data")
        self.process_late_swap_data(rearranged_unique, num_sets)

        total_time = time.time() - start_time
        print(f"\nOutput Generation Complete!")

    def validate_entry_groups(self, entry_groups, num_sets):
        """Validate that all entry groups have the correct number of sets."""
        for base_entry_id, entries in entry_groups.items():
            if len(entries) != num_sets:
                print(f"Warning: Entry {base_entry_id} has {len(entries)} sets instead of {num_sets}")

    def rearrange_lineups(self, sorted_unique, num_sets):
        """Rearrange lineups into specified number of sets with entries grouped by ROI."""
        self.print(f"Rearranging lineups into {num_sets} sets...")

        # First, group lineups by their base entry ID (removing the _X suffix)
        entry_groups = {}
        for item in sorted_unique:
            # Extract the entry ID from the lineup string
            entry_id = item[1].split(',')[-1].strip()  # Get the full entry ID
            base_entry_id = entry_id.split('_')[0] if '_' in entry_id else entry_id

            if base_entry_id not in entry_groups:
                entry_groups[base_entry_id] = []
            entry_groups[base_entry_id].append(item)

        # Sort each group by ROI and ceiling
        for base_entry_id in entry_groups:
            entry_groups[base_entry_id].sort(
                key=lambda x: (
                    -float(x[1].split(",")[13].replace("%", "")),  # ROI (descending)
                    -float(x[1].split(",")[9])  # Ceiling (descending)
                )
            )

            # After grouping entries
            self.validate_entry_groups(entry_groups, num_sets)

        # Create the final rearranged list
        rearranged_unique = []
        for set_index in range(num_sets):
            # For each set, add the corresponding entry from each group
            for base_entry_id in sorted(entry_groups.keys()):
                entries = entry_groups[base_entry_id]
                if set_index < len(entries):
                    rearranged_unique.append(entries[set_index])

        self.print(f"Successfully rearranged {len(rearranged_unique):,} lineups")
        return rearranged_unique


    def write_main_output(self, rearranged_unique):
        """Write main output file with lineup data."""
        out_path = os.path.join(
            os.path.dirname(__file__),
            f"../dk_output/dk_gpp_sim_lineups_{self.field_size}_{self.num_iterations}.csv"
        )


        print(f"Writing main output to: {os.path.basename(out_path)}")
        total_lineups = len(rearranged_unique)

        with open(out_path, "w") as f:
            # Write header
            f.write(
                "PG,SG,SF,PF,C,G,F,UTIL,Proj,Ceiling,Salary,Cash %,Top 1%,ROI%,"
                "Wins,Own Sum,Avg Return,Stack1,Stack2,Lineup,Dupes,User,Index,Entry ID\n"
            )

            # Write lineup data with progress updates
            for idx, (_, lineup_str) in enumerate(rearranged_unique, 1):
                f.write(f"{lineup_str}\n")

                if idx % max(1, total_lineups // 10) == 0:  # Update every 10%
                    print(f"Writing lineups: {idx:,}/{total_lineups:,} ({(idx / total_lineups) * 100:.1f}%)")

        print("Main output file complete")

    def write_all_lineups_output(self):
        """Write an additional output file with all lineups sorted by ROI and ceiling."""
        out_path = os.path.join(
            os.path.dirname(__file__),
            f"../dk_output/dk_gpp_sim_all_lineups_{self.field_size}_{self.num_iterations}.csv"
        )

        print(f"Writing all lineups output to: {os.path.basename(out_path)}")

        # Process all lineups
        all_lineups = []
        total_lineups = len(self.field_lineups)

        print("\nProcessing all lineups...")
        for idx, (index, y) in enumerate(self.field_lineups.items()):
            if (idx + 1) % max(1, total_lineups // 10) == 0:  # Update every 10%
                print(f"Processing lineups: {idx + 1:,}/{total_lineups:,} ({((idx + 1) / total_lineups) * 100:.1f}%)")

            lu_idx = self.lineup_to_int[index]
            if lu_idx is None:
                print(f"Warning: Lineup index {index} not found in lineup_to_int.")
                continue

            for entry in y['EntryIds']:
                lineup_info = self.contest_lineups[entry]
                # Process lineup for output file
                lineup_str = self.process_lineup_details(lineup_info, y, lu_idx, entry)
                if lineup_str:
                    all_lineups.append(lineup_str)

        # Sort all lineups by ROI (descending) and Ceiling (descending)
        print("\nSorting all lineups...")
        sorted_lineups = sorted(
            all_lineups,
            key=lambda x: (
                -float(x.split(",")[13].replace("%", "")),  # Primary sort by ROI (descending)
                -float(x.split(",")[9])  # Secondary sort by Ceiling (descending)
            )
        )

        # Write sorted lineups to file
        print("\nWriting all lineups to file...")
        total_lineups = len(sorted_lineups)

        with open(out_path, "w") as f:
            # Write header
            f.write(
                "PG,SG,SF,PF,C,G,F,UTIL,Proj,Ceiling,Salary,Cash %,Top 1%,ROI%,"
                "Wins,Own Sum,Avg Return,Stack1,Stack2,Lineup,Dupes,User,Index,Entry ID\n"
            )

            # Write lineup data with progress updates
            for idx, lineup_str in enumerate(sorted_lineups, 1):
                f.write(f"{lineup_str}\n")

                if idx % max(1, total_lineups // 10) == 0:  # Update every 10%
                    print(f"Writing lineups: {idx:,}/{total_lineups:,} ({(idx / total_lineups) * 100:.1f}%)")

        print("All lineups output file complete")




    def write_player_exposure(self):
        """Process and write player exposure data."""
        out_path = os.path.join(
            os.path.dirname(__file__),
            f"../dk_output/dk_lateswap_sim_player_exposure_{self.field_size}_{self.num_iterations}.csv"
        )

        print(f"Processing player exposure data...")
        unique_players = {}

        # Process player data
        total_lineups = len(self.field_lineups)
        for idx, (_, val) in enumerate(self.field_lineups.items(), 1):
            if idx % max(1, total_lineups // 10) == 0:  # Update every 10%
                print(f"Processing exposure data: {idx:,}/{total_lineups:,} ({(idx / total_lineups) * 100:.1f}%)")

            roi_value = val["ROI"] / 100 if not (math.isnan(val["ROI"]) or val["ROI"] is None) else 0

            for player in val["Lineup"]:
                if player not in unique_players:
                    unique_players[player] = {
                        "Cashes": val["Cashes"],
                        "Top1Percent": val["Top1Percent"],
                        "In": val["Count"],
                        "ROI": roi_value,
                    }
                else:
                    unique_players[player]["Cashes"] += val["Cashes"]
                    unique_players[player]["Top1Percent"] += val["Top1Percent"]
                    unique_players[player]["In"] += val["Count"]
                    unique_players[player]["ROI"] += roi_value


        # Write exposure data
        print("Writing player exposure data...")
        total_players = len(unique_players)
        processed = 0

        # Calculate ROI and create sorted list
        player_data_list = []
        for player_id, data in unique_players.items():
            field_p = round(data["In"] / self.field_size * 100, 2)
            max_ranked = field_p / 100 * self.field_size * self.num_iterations

            if max_ranked == 0:
                cash_p, top10_p, roi_p = 0, 0, 0
            else:
                cash_p = round(data["Cashes"] / max_ranked * 100, 2)
                top10_p = round(data["Top1Percent"] / max_ranked * 100, 2)
                roi_p = round(data["ROI"] / max_ranked * 100, 2)

                # Clean up extremely large ROI values
                if abs(roi_p) > 1000000:
                    roi_p = 0.0

            # Find player details
            player_details = None
            for v in self.player_dict.values():
                if player_id == v["ID"]:
                    player_details = v
                    break

            if player_details:
                player_data_list.append({
                    "roi": roi_p,
                    "details": {
                        "player": player_details,
                        "cash_p": cash_p,
                        "top10_p": top10_p,
                        "field_p": field_p,
                        "roi_p": roi_p
                    }
                })

        # Sort by ROI in descending order
        player_data_list.sort(key=lambda x: x["roi"], reverse=True)
        # Write sorted exposure data
        print("Writing sorted player exposure data...")
        total_players = len(player_data_list)

        with open(out_path, "w") as f:
            f.write(
                "Player,Position,Team,UpdatedProjection,UpdatedStdDev,Cash%,Top1%,"
                "Sim. Own%,Proj. Own%,Avg. Return,Game Minutes Remaining\n"
            )

            for idx, player_data in enumerate(player_data_list, 1):
                if idx % max(1, total_players // 5) == 0:
                    print(
                        f"Writing player data: {idx:,}/{total_players:,} ({(idx / total_players) * 100:.1f}%)")

                details = player_data["details"]
                player = details["player"]
                if player.get('BayesianProjectedFpts') < 1:
                    player['BayesianProjectedFpts'] = 0
                    player['BayesianProjectedVar'] = 0


                # Format ROI with fixed notation
                roi_str = f"${details['roi_p']:.2f}" if abs(details['roi_p']) < 1000000 else "$0.00"

                f.write(
                    f"{player['Name'].replace('#', '-')},"
                    f"{'/'.join(player.get('Position'))},"
                    f"{player.get('Team')},"
                    f"{player.get('BayesianProjectedFpts')},"
                    f"{np.sqrt(player.get('BayesianProjectedVar'))},"
                    f"{details['cash_p']}%,{details['top10_p']}%,"
                    f"{details['field_p']}%,{player['Ownership']}%,"
                    f"{roi_str},{player.get('Minutes Remaining')}\n"
                )

        print("Player exposure file complete")


    def process_late_swap_data(self, rearranged_unique, num_sets):
        """Process late swap data with basic logging and proven functionality."""
        import logging
        import gc
        import os

        # Set up logging in the current working directory
        current_dir = os.getcwd()
        log_file = os.path.join(current_dir, f"late_swap_log.txt")

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True  # Force configuration to ensure it works
        )

        def log_print(message):
            """Log to both file and print to GUI."""
            #logging.info(message)
            print(message)

        try:
            log_print("Starting late swap processing...")

            # Create mapping of entry IDs to output data
            entryid_to_output = {}
            for lineup_data, lineup_obj in self.output_lineups:
                entry_id = lineup_obj["EntryId"]
                entryid_to_output[entry_id] = (lineup_data, lineup_obj)

            new_output_lineups = []
            for _, lineup_str in rearranged_unique:
                # The Entry ID should be the last column in lineup_str
                cols = lineup_str.split(',')
                entry_id = cols[-1].strip()

                if entry_id in entryid_to_output:
                    new_output_lineups.append(entryid_to_output[entry_id])
                else:
                    log_print(f"Warning: Entry ID {entry_id} not found in output_lineups mapping.")

            # Sort lineups
            sorted_lineups = []
            for lineup, old_lineup in new_output_lineups:
                if "contest_id" not in old_lineup or "EntryId" not in old_lineup:
                    continue
                sorted_lineup = self.sort_lineup(lineup)
                sorted_lineup = self.adjust_roster_for_late_swap(sorted_lineup, old_lineup)
                sorted_lineups.append((sorted_lineup, old_lineup))

            log_print(f"Processed {len(sorted_lineups)} lineups")

            if 'late_swap_path' not in self.config:
                return

            late_swap_path = os.path.join(
                os.path.dirname(__file__),
                f"../dk_data/{self.config['late_swap_path']}"
            )



            # Read the template data
            with open(late_swap_path, "r", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames[:12]
                template_rows = list(reader)

            num_entries = len(sorted_lineups) // num_sets
            first_file_columns = [
                {key: row[key] for key in fieldnames[:4]}
                for row in template_rows[:num_entries]
            ]


            # Process sets in small chunks
            chunk_size = 5
            for chunk_start in range(0, num_sets, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_sets)
                log_print(f"\nProcessing sets {chunk_start + 1} to {chunk_end}")

                for set_index in range(chunk_start, chunk_end):
                    try:
                        start_idx = set_index * num_entries
                        end_idx = start_idx + num_entries
                        log_print(f"Processing set {set_index + 1}")

                        lineups_slice = sorted_lineups[start_idx:end_idx]
                        updated_rows = []
                        base_count = set_index * num_entries

                        for idx, (new_lineup, old_lineup) in enumerate(lineups_slice):
                            count = base_count + idx
                            trimmed_row = dict(first_file_columns[count % num_entries])

                            for i, position in enumerate(["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]):
                                player_entry = new_lineup[i]
                                player_id = player_entry[-1] if isinstance(player_entry, tuple) else player_entry

                                if player_id in self.player_dict:
                                    player_data = self.player_dict[player_id]
                                    trimmed_row[position] = f"{player_data['Name']} ({player_data['ID']})"
                                else:
                                    trimmed_row[position] = "Unknown Player"

                            updated_rows.append(trimmed_row)

                        # Write the updated rows to a new CSV file for this set
                        late_swap_entries_path = os.path.join(
                            os.path.dirname(__file__),
                            f"../dk_output/late_swap_{num_entries}_entries_set_{set_index + 1}_entries.csv"
                        )

                        with open(late_swap_entries_path, "w", encoding="utf-8-sig", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(updated_rows)

                        log_print(f"Completed set {set_index + 1}")

                        # Clear rows and do garbage collection
                        updated_rows.clear()
                        gc.collect()

                    except Exception as e:
                        log_print(f"Error processing set {set_index + 1}: {str(e)}")
                        raise

            log_print("Late swap processing complete")

        except Exception as e:
            log_print(f"Error in late swap processing: {str(e)}")
            raise

    def process_lineup_details(self, x, y, lu_idx, entry):
        """Process individual lineup details and return formatted string"""
        try:
            lu_type = x["Type"]
            userName = x['User']

            # Process player details
            players_info = self.get_players_info(x)
            if not players_info:
                return None

            lu_names, lu_teams, own_p, total_salary, total_projection, total_variance = players_info

            # Calculate statistics
            ceil_p = total_projection + np.sqrt(total_variance)  # Modified this
            counter = Counter(lu_teams)
            stacks = counter.most_common()
            primaryStack = f"{stacks[0][0]} {stacks[0][1]}"
            secondaryStack = f"{stacks[1][0]} {stacks[1][1]}"

            own_s = np.sum(own_p)

            # Safely calculate ROI percentage
            if (self.entry_fee <= 0 or self.num_iterations <= 0 or
                    y.get("ROI") is None or np.isnan(y["ROI"])):
                roi_p = 0.0
            else:
                roi_p = round(y["ROI"] / self.entry_fee / self.num_iterations * 100, 2)

            # Safely calculate ROI round
            if self.num_iterations <= 0 or y.get("ROI") is None or np.isnan(y["ROI"]):
                roi_round = 0.0
            else:
                roi_round = round(y["ROI"] / self.num_iterations, 2)


            # Safely calculate cash and top1 percentages
            cash_percentage = (y.get("Cashes", 0) / self.num_iterations * 100) if self.num_iterations > 0 else 0.0
            top1_percentage = y.get("Top1Percent", 0)

            # Update contest entries
            if userName in self.contest_entries:
                self.contest_entries[userName]['ROI'] += y.get('ROI', 0)
                self.contest_entries[userName]['Cashes'] += y.get('Cashes', 0)
                self.contest_entries[userName]['Top1'] += y.get('Top1Percent', 0)

            # Format lineup string
            return self.format_lineup_string(
                lu_names, x, total_projection, ceil_p, total_salary,
                cash_percentage, top1_percentage,
                roi_p, y.get("Wins", 0), own_s, roi_round,
                primaryStack, secondaryStack, lu_type,
                y.get("Count", 0), userName, lu_idx, entry
            )
        except Exception as e:
            print(f"Warning: Error processing lineup {entry}: {str(e)}")
            return None



    def get_players_info(self, lineup):
        """Get detailed player information for a lineup"""
        lu_names = []
        lu_teams = []
        own_p = []
        total_salary = 0
        total_projection = 0
        total_variance = 0

        for p in self.roster_construction:
            p_str = f'{p}_is_locked'
            Id = lineup[p]
            if Id == '':
                lu_names.append('null')
                lu_teams.append('null')
                own_p.append(0)
                continue

            player_found = False
            for k, v in self.player_dict.items():
                if v["ID"] == Id:
                    lu_names.append(v["DK Name"])
                    lu_teams.append(v["Team"])
                    own_p.append(v["Ownership"])
                    total_salary += v["Salary"]
                    total_projection += v["BayesianProjectedFpts"]
                    total_variance += v["BayesianProjectedVar"]
                    player_found = True
                    break

            if not player_found:
                print(f"Warning: Player ID {Id} not found in player dictionary")
                return None

        return lu_names, lu_teams, own_p, total_salary, total_projection, total_variance

    def format_lineup_string(self, lu_names, x, fpts_p, ceil_p, total_salary,
                             cash_p, top10_p, roi_p, win_p, own_s, roi_round,
                             primaryStack, secondaryStack, lu_type, simDupes,
                             userName, lu_idx, entry):
        """Format lineup information into string for output"""
        players_str = ",".join(
            f"{name.replace('#', '-')} ({x[pos]})"
            for name, pos in zip(lu_names, self.roster_construction)
        )

        stats_str = f"{fpts_p},{ceil_p},{total_salary},{cash_p:.2f}%,{top10_p},{roi_p}%,{win_p},{own_s},${roi_round}"
        meta_str = f"{primaryStack},{secondaryStack},{lu_type},{simDupes},{userName},{lu_idx},{entry}"

        return f"{players_str},{stats_str},{meta_str}"


    def sort_lineup(self, lineup):
        order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        sorted_lineup = [None] * 8

        for player in lineup:
            player_key, pos, player_id = player  # Extract the ID as well
            order_idx = order.index(pos)
            if sorted_lineup[order_idx] is None:
                # Include the full tuple (key, position, ID) in the sorted lineup
                sorted_lineup[order_idx] = (player_key, pos, player_id)
            else:
                # Handle the case where the position is already filled
                sorted_lineup[order_idx + 1] = (player_key, pos, player_id)

        return sorted_lineup

    def adjust_roster_for_late_swap(self, lineup, old_lineup):

        POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

        def is_locked(position_index):
            """Check if a position is locked in the old lineup."""
            if 0 <= position_index < len(POSITIONS):
                position_name = POSITIONS[position_index]
                return old_lineup.get(f"{position_name}_is_locked", False)  # Default to False if key is missing
            raise ValueError(f"Invalid position index: {position_index}")

        # Ensure locked players remain in their positions
        for i, position in enumerate(POSITIONS):
            if is_locked(i):
                lineup[i] = old_lineup[position]  # Force the locked player into their position in the new lineup

        # Iterate over non-locked positions for adjustments
        for i, position in enumerate(POSITIONS):
            if position not in ["G", "F", "UTIL"]:
                continue  # Skip non-flexible positions

            current_player = lineup[i]
            current_player_data = self.player_dict.get(current_player, {})
            current_player_start_time = current_player_data.get("GameTime", float('inf'))

            # Skip this position if it is locked
            if is_locked(i):
                continue

            for primary_i, primary_pos in enumerate(POSITIONS[:5]):  # Iterate only over primary positions
                primary_player = lineup[primary_i]
                primary_player_data = self.player_dict.get(primary_player, {})
                primary_player_start_time = primary_player_data.get("GameTime", float('inf'))

                # Skip if the primary position is locked or involves a locked player
                if is_locked(primary_i):
                    continue

                # Ensure the current and primary players are not the same locked player
                if current_player == old_lineup.get(primary_pos) or primary_player == old_lineup.get(position):
                    print(f"Skipping swap: Locked players involved ({current_player}, {primary_player})")
                    continue

                # Check if swapping is valid based on game start times and position overlap
                if (
                        primary_player_start_time > current_player_start_time
                        and set(primary_player_data.get("Position", [])) & set(current_player_data.get("Position", []))
                ):
                    # Swap players between positions
                    print(f"Swapping {current_player} with {primary_player}")
                    lineup[i], lineup[primary_i] = lineup[primary_i], lineup[i]
                    break  # Exit the loop once a swap is made

        return lineup

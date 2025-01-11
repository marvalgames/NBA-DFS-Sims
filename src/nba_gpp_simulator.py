import collections
import csv
import datetime
import itertools
import json
import math
import multiprocessing as mp
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import pulp as plp
from numba import jit
from scipy.stats import multivariate_normal


@jit(nopython=True)
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2


class NBA_GPP_Simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    stacks_dict = {}
    gen_lineup_list = []
    roster_construction = []
    id_name_dict = {}
    salary = None
    optimal_score = None
    field_size = None
    team_list = []
    num_iterations = None
    site = None
    payout_structure = {}
    use_contest_data = False
    entry_fee = None
    use_lineup_input = None
    matchups = set()
    projection_minimum = 15
    randomness_amount = 100
    min_lineup_salary = 49000
    max_pct_off_optimal = 0.4
    overlap_limit = 7
    teams_dict = collections.defaultdict(list)  # Initialize teams_dict
    correlation_rules = {}
    game_info = {}
    seen_lineups = {}
    seen_lineups_ix = {}
    position_map = {
        0: ["PG"],
        1: ["SG"],
        2: ["SF"],
        3: ["PF"],
        4: ["C"],
        5: ["PG", "SG"],
        6: ["SF", "PF"],
        7: ["PG", "SG", "SF", "PF", "C"],
    }

    def __init__(
        self,
        site,
        field_size,
        num_iterations,
        use_contest_data,
        use_lineup_input,
        min_lineup_salary,
        projection_min,
        contest_path
    ):
        self.projection_min = int(projection_min)
        self.site = site
        self.use_lineup_input = use_lineup_input
        self.load_config()
        self.load_rules()

        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../dk_data/{}".format(self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../dk_data/{}".format(self.config["player_path"]),
        )

        print(f"Default solver: {plp.LpSolverDefault}")

        self.load_player_ids(player_path)

        self.roster_construction = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        self.salary = 50000

        # Use the provided contest_path instead of hardcoding it
        if contest_path and os.path.exists(contest_path):
            self.load_contest_data(contest_path)
            print(f"Contest data loaded from: {contest_path}")
        else:
            print("Warning: Contest path not provided or file does not exist")

        self.load_contest_data(contest_path)
        print(f"Contest payout structure loaded.")

        self.assertPlayerDict()
        self.num_iterations = int(num_iterations)
        self.min_lineup_salary = int(min_lineup_salary)
        self.get_optimal()
        if self.use_lineup_input:
            self.load_lineups_from_file()
        self.load_correlation_rules()

    # make column lookups on datafiles case-insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    def load_rules(self):
        self.projection_minimum = int(self.projection_minimum)
        self.randomness_amount = float(self.config["randomness"])#not used by sims
        self.min_lineup_salary = int(self.min_lineup_salary)
        self.max_pct_off_optimal = float(self.config["max_pct_off_optimal"])
        self.default_var = float(self.config["default_var"])
        self.correlation_rules = self.config["custom_correlations"]


    def assertPlayerDict(self):
        for p, s in list(self.player_dict.items()):
            if s["ID"] == 0 or s["ID"] == "" or s["ID"] is None:
                print(
                    s["Name"]
                    + " name mismatch between projections and player ids, excluding from player_dict"
                )
                self.player_dict.pop(p)

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `mlb_optimizer.py`
    def get_optimal(self):
        problem = plp.LpProblem("NBA", plp.LpMaximize)
        lp_variables = {
            self.player_dict[(player, pos_str, team)]["ID"]: plp.LpVariable(
                str(self.player_dict[(player, pos_str, team)]["UniqueKey"]),
                cat="Binary",
            )
            for (player, pos_str, team) in self.player_dict
        }

        # set the objective - maximize fpts
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["fieldFpts"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
            ),
            "Objective",
        )

        # Set the salary constraints
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
            )
            <= self.salary
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "PG" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            >= 1
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "PG" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            <= 3
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "SG" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            >= 1
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "SG" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            <= 3
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "SF" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            >= 1
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "SF" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            <= 3
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "PF" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            >= 1
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "PF" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            <= 3
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "C" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            >= 1
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "C" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            <= 2
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "PG" in self.player_dict[(player, pos_str, team)]["Position"]
                or "SG" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            >= 3
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "SF" in self.player_dict[(player, pos_str, team)]["Position"]
                or "PF" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            >= 3
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
                if "PG" in self.player_dict[(player, pos_str, team)]["Position"]
                or "C" in self.player_dict[(player, pos_str, team)]["Position"]
            )
            <= 4
        )

        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
            )
            == 8
        )

        # Max 8 per team in case of weird issues with stacking on short slates
        for team in self.team_list:
            problem += (
                plp.lpSum(
                    lp_variables[
                        self.player_dict[(player, pos_str, team)]["UniqueKey"]
                    ]
                    for (player, pos_str, team) in self.player_dict
                    if self.player_dict[(player, pos_str, team)]["Team"] == team
                )
                <= 7
            )

        try:
            problem.solve(plp.PULP_CBC_CMD(msg=False))
        except plp.PulpSolverError:
            print(
                "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                    len(self.num_lineups), self.num_lineups
                )
            )

        # Get the lineup and add it to our list
        player_unqiue_keys = [
            player for player in lp_variables if lp_variables[player].varValue != 0
        ]
        players = []
        for key, value in self.player_dict.items():
            if value["UniqueKey"] in player_unqiue_keys:
                players.append(key)

        fpts_proj = sum(self.player_dict[player]["fieldFpts"] for player in players)
        self.optimal_score = float(fpts_proj)

    @staticmethod
    def extract_matchup_time(game_string):
        # Extract the matchup, date, and time
        match = re.match(
            r"(\w{2,4}@\w{2,4}) (\d{2}/\d{2}/\d{4}) (\d{2}:\d{2}[APM]{2} ET)",
            game_string,
        )

        if match:
            matchup, date, time = match.groups()
            # Convert 12-hour time format to 24-hour format
            time_obj = datetime.datetime.strptime(time, "%I:%M%p ET")
            # Convert the date string to datetime.date
            date_obj = datetime.datetime.strptime(date, "%m/%d/%Y").date()
            # Combine date and time to get a full datetime object
            datetime_obj = datetime.datetime.combine(date_obj, time_obj.time())
            return matchup, datetime_obj
        return None

    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                name_key = "name" if self.site == "dk" else "nickname"
                player_name = row[name_key].replace("-", "#").lower().strip()
                # some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                position = [pos for pos in row["position"].split("/")]
                position.sort()
                if any(pos in ["PG", "SG"] for pos in position):
                    position.append("G")
                if any(pos in ["SF", "PF"] for pos in position):
                    position.append("F")
                position.append("UTIL")
                team_key = "teamabbrev" if self.site == "dk" else "team"
                team = row[team_key]
                game_info = "game info" if self.site == "dk" else "game"
                game_info_str = row["game info"] if self.site == "dk" else row["game"]
                result = self.extract_matchup_time(game_info_str)
                match = re.search(pattern=r"(\w{2,4}@\w{2,4})", string=row[game_info])
                if match:
                    opp = match.groups()[0].split("@")
                    self.matchups.add((opp[0], opp[1]))
                    for m in opp:
                        if m != team:
                            team_opp = m
                    opp = tuple(opp)
                if result:
                    matchup, game_time = result
                    self.game_info[opp] = game_time
                pos_str = str(position)
                if (player_name, pos_str, team) in self.player_dict:
                    self.player_dict[(player_name, pos_str, team)]["ID"] = str(
                        row["id"]
                    )
                    self.player_dict[(player_name, pos_str, team)]["UniqueKey"] = str(
                        row["id"]
                    )
                    self.player_dict[(player_name, pos_str, team)]["Team"] = row[
                        team_key
                    ]
                    self.player_dict[(player_name, pos_str, team)]["Opp"] = team_opp
                    self.player_dict[(player_name, pos_str, team)]["Matchup"] = opp
                self.id_name_dict[str(row["id"])] = row[name_key]

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
                    # print(indices)
                    # have to add 1 to range to get it to generate value for everything
                    for i in range(int(indices[0]), int(indices[1]) + 1):
                        # print(i)
                        # Where I'm from, we 0 index things. Thus, -1 since Payout starts at 1st place
                        if i >= self.field_size:
                            break
                        self.payout_structure[i - 1] = float(
                            row["payout"].replace(",", "")
                        )
                # single-position payouts
                else:
                    if int(row["place"]) >= self.field_size:
                        break
                    self.payout_structure[int(row["place"]) - 1] = float(
                        row["payout"].replace(",", "")
                    )

    def load_correlation_rules(self):
        if len(self.correlation_rules.keys()) > 0:
            for primary_player in self.correlation_rules.keys():
                # Convert primary_player to the consistent format
                formatted_primary_player = (
                    primary_player.replace("-", "#").lower().strip()
                )
                for (
                    player_name,
                    pos_str,
                    team,
                ), player_data in self.player_dict.items():
                    if formatted_primary_player == player_name:
                        for second_entity, correlation_value in self.correlation_rules[
                            primary_player
                        ].items():
                            # Convert second_entity to the consistent format
                            formatted_second_entity = (
                                second_entity.replace("-", "#").lower().strip()
                            )

                            # Check if the formatted_second_entity is a player name
                            found_second_entity = False
                            for (
                                se_name,
                                se_pos_str,
                                se_team,
                            ), se_data in self.player_dict.items():
                                if formatted_second_entity == se_name:
                                    player_data["Player Correlations"][
                                        formatted_second_entity
                                    ] = correlation_value
                                    se_data["Player Correlations"][
                                        formatted_primary_player
                                    ] = correlation_value
                                    found_second_entity = True
                                    break

                            # If the second_entity is not found as a player, assume it's a position and update 'Correlations'
                            if not found_second_entity:
                                player_data["Correlations"][
                                    second_entity
                                ] = correlation_value

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#").lower().strip()
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
                own = float(row["own%"].replace("%", ""))
                if own == 0:
                    own = 0.1
                pos_str = str(position)
                player_data = {
                    "Fpts": fpts,
                    "fieldFpts": fieldFpts,
                    "Position": position,
                    "Name": player_name,
                    "DK Name": row["name"],
                    "Team": team,
                    "Opp": "",
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
                }

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

    def extract_id(self, cell_value):
        if "(" in cell_value and ")" in cell_value:
            return cell_value.split("(")[1].replace(")", "")
        elif ":" in cell_value:
            return cell_value.split(":")[0]
        else:
            return cell_value

    def load_lineups_from_file(self):
        print("loading lineups")
        i = 0
        path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, "tournament_lineups.csv"),
        )
        with open(path) as file:
            reader = pd.read_csv(file)
            bad_players = []
            j = 0
            for i, row in reader.iterrows():
                if i == self.field_size:
                    break
                lineup = [
                    self.extract_id(str(row[q]))
                    for q in range(len(self.roster_construction))
                ]
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print(
                            "player id {} in lineup {} not found in player dict".format(
                                l, i
                            )
                        )
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        bad_players.append(l)
                        error = True
                if len(lineup) < len(self.roster_construction):
                    print("lineup {} doesn't match roster construction size".format(i))
                    continue
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                if not error:
                    lineup_list = sorted(lineup)
                    lineup_set = frozenset(lineup_list)

                    # Keeping track of lineup duplication counts
                    if lineup_set in self.seen_lineups:
                        self.seen_lineups[lineup_set] += 1
                    else:
                        self.field_lineups[j] = {
                            "Lineup": lineup,
                            "Wins": 0,
                            "Top1Percent": 0,
                            "ROI": 0,
                            "Cashes": 0,
                            "Type": "opto",
                            "Count": 1,
                        }

                        # Add to seen_lineups and seen_lineups_ix
                        self.seen_lineups[lineup_set] = 1
                        self.seen_lineups_ix[lineup_set] = j

                        j += 1
        print("loaded {} lineups".format(j))

    @staticmethod
    def generate_lineups(
        lu_num,
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
        opponents,
        overlap_limit,
        matchups,
        num_players_in_roster,
        site,
    ):
        # new random seed for each lineup (without this there is a ton of dupes)
        rng = np.random.Generator(np.random.PCG64())
        lus = {}
        # make sure nobody is already showing up in a lineup
        if sum(in_lineup) != 0:
            in_lineup.fill(0)
        reject = True
        iteration_count = 0
        total_players = num_players_in_roster
        reasonable_projection = optimal_score - (max_pct_off_optimal * optimal_score)
        max_players_per_team = 4 if site == "fd" else None
        while reject:
            iteration_count += 1
            salary = 0
            proj = 0
            if sum(in_lineup) != 0:
                in_lineup.fill(0)
            lineup = []
            player_teams = []
            def_opps = []
            players_opposing_def = 0
            lineup_matchups = []
            k = 0
            for pos in pos_matrix.T:
                if k < 1:
                    # check for players eligible for the position and make sure they arent in a lineup, returns a list of indices of available player
                    valid_players = np.nonzero((pos > 0) & (in_lineup == 0))[0]
                    # grab names of players eligible
                    plyr_list = ids[valid_players]
                    # create np array of probability of being seelcted based on ownership and who is eligible at the position
                    prob_list = ownership[valid_players]
                    prob_list = prob_list / prob_list.sum()
                    try:
                        choice = rng.choice(plyr_list, p=prob_list)
                    except:
                        print(plyr_list, prob_list)
                        print("find failed on nonstack and first player selection")
                    choice_idx = np.nonzero(ids == choice)[0]
                    lineup.append(str(choice))
                    in_lineup[choice_idx] = 1
                    salary += salaries[choice_idx]
                    proj += projections[choice_idx]
                    lineup_matchups.append(matchups[choice_idx[0]])
                    player_teams.append(teams[choice_idx][0])
                if k >= 1:
                    remaining_salary = salary_ceiling - salary
                    if players_opposing_def < overlap_limit:
                        if k == total_players - 1:
                            valid_players = np.nonzero(
                                (pos > 0)
                                & (in_lineup == 0)
                                & (salaries <= remaining_salary)
                                & (salary + salaries >= salary_floor)
                            )[0]
                        else:
                            valid_players = np.nonzero(
                                (pos > 0)
                                & (in_lineup == 0)
                                & (salaries <= remaining_salary)
                            )[0]
                        # grab names of players eligible
                        plyr_list = ids[valid_players]
                        # create np array of probability of being seelcted based on ownership and who is eligible at the position
                        prob_list = ownership[valid_players]
                        prob_list = prob_list / prob_list.sum()
                        if k == total_players - 1:
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
                            if k == total_players - 1:
                                choice = rng.choice(plyr_list, p=boosted_probabilities)
                            else:
                                choice = rng.choice(plyr_list, p=prob_list)
                        except:
                            salary = 0
                            proj = 0
                            lineup = []
                            player_teams = []
                            def_opps = []
                            players_opposing_def = 0
                            lineup_matchups = []
                            in_lineup.fill(0)  # Reset the in_lineup array
                            k = 0  # Reset the player index
                            continue  # Skip to the next iteration of the while loop
                        choice_idx = np.nonzero(ids == choice)[0]
                        lineup.append(str(choice))
                        in_lineup[choice_idx] = 1
                        salary += salaries[choice_idx]
                        proj += projections[choice_idx]
                        player_teams.append(teams[choice_idx][0])
                        lineup_matchups.append(matchups[choice_idx[0]])
                        if max_players_per_team is not None:
                            team_count = Counter(player_teams)
                            if any(
                                count > max_players_per_team
                                for count in team_count.values()
                            ):
                                salary = 0
                                proj = 0
                                lineup = []
                                player_teams = []
                                def_opps = []
                                players_opposing_def = 0
                                lineup_matchups = []
                                in_lineup.fill(0)  # Reset the in_lineup array
                                k = 0  # Reset the player index
                                continue  # Skip to the next iteration of the while loop
                    else:
                        if k == total_players - 1:
                            valid_players = np.nonzero(
                                (pos > 0)
                                & (in_lineup == 0)
                                & (salaries <= remaining_salary)
                                & (salary + salaries >= salary_floor)
                            )[0]
                        else:
                            valid_players = np.nonzero(
                                (pos > 0)
                                & (in_lineup == 0)
                                & (salaries <= remaining_salary)
                            )[0]
                        # grab names of players eligible
                        plyr_list = ids[valid_players]
                        # create np array of probability of being seelcted based on ownership and who is eligible at the position
                        prob_list = ownership[valid_players]
                        prob_list = prob_list / prob_list.sum()
                        if k == total_players - 1:
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
                            if k == total_players - 1:
                                choice = rng.choice(plyr_list, p=boosted_probabilities)
                            else:
                                choice = rng.choice(plyr_list, p=prob_list)
                        except:
                            salary = 0
                            proj = 0
                            lineup = []
                            player_teams = []
                            players_opposing_def = 0
                            lineup_matchups = []
                            in_lineup.fill(0)  # Reset the in_lineup array
                            k = 0  # Reset the player index
                            continue  # Skip to the next iteration of the while loop
                            # if remaining_salary <= np.min(salaries):
                            #     reject_counters["salary_too_high"] += 1
                            # else:
                            #     reject_counters["salary_too_low"]
                        choice_idx = np.nonzero(ids == choice)[0]
                        lineup.append(str(choice))
                        in_lineup[choice_idx] = 1
                        salary += salaries[choice_idx]
                        proj += projections[choice_idx]
                        player_teams.append(teams[choice_idx][0])
                        lineup_matchups.append(matchups[choice_idx[0]])
                        if max_players_per_team is not None:
                            team_count = Counter(player_teams)
                            if any(
                                count > max_players_per_team
                                for count in team_count.values()
                            ):
                                salary = 0
                                proj = 0
                                lineup = []
                                player_teams = []
                                def_opps = []
                                players_opposing_def = 0
                                lineup_matchups = []
                                in_lineup.fill(0)  # Reset the in_lineup array
                                k = 0  # Reset the player index
                                continue  # Skip to the next iteration of the while loop
                k += 1
            # Must have a reasonable salary
            # if salary > salary_ceiling:
            #     reject_counters["salary_too_high"] += 1
            # elif salary < salary_floor:
            #     reject_counters["salary_too_low"] += 1
            if salary >= salary_floor and salary <= salary_ceiling:
                # Must have a reasonable projection (within 60% of optimal) **people make a lot of bad lineups
                if proj >= reasonable_projection:
                    if len(set(lineup_matchups)) > 1:
                        if max_players_per_team is not None:
                            team_count = Counter(player_teams)
                            if all(
                                count <= max_players_per_team
                                for count in team_count.values()
                            ):
                                reject = False
                                lus[lu_num] = {
                                    "Lineup": lineup,
                                    "Wins": 0,
                                    "Top1Percent": 0,
                                    "ROI": 0,
                                    "Cashes": 0,
                                    "Type": "generated",
                                    "Count": 0,
                                }
                        else:
                            reject = False
                            lus[lu_num] = {
                                "Lineup": lineup,
                                "Wins": 0,
                                "Top1Percent": 0,
                                "ROI": 0,
                                "Cashes": 0,
                                "Type": "generated",
                                "Count": 0,
                            }
        return lus

    def generate_field_lineups(self):
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print(
                "supplied lineups >= contest field size. only retrieving the first "
                + str(self.field_size)
                + " lineups"
            )
        else:
            print("Generating " + str(diff) + " lineups.")
            ids = []
            ownership = []
            salaries = []
            projections = []
            positions = []
            teams = []
            opponents = []
            matchups = []
            # put def first to make it easier to avoid overlap
            for k in self.player_dict.keys():
                if "Team" not in self.player_dict[k].keys():
                    print(
                        self.player_dict[k]["Name"],
                        " name mismatch between projections and player ids!",
                    )
                ids.append(self.player_dict[k]["UniqueKey"])
                ownership.append(self.player_dict[k]["Ownership"])
                salaries.append(self.player_dict[k]["Salary"])
                if self.player_dict[k]["fieldFpts"] >= self.projection_minimum:
                    projections.append(self.player_dict[k]["fieldFpts"])
                else:
                    projections.append(0)
                teams.append(self.player_dict[k]["Team"])
                opponents.append(self.player_dict[k]["Opp"])
                matchups.append(self.player_dict[k]["Matchup"])
                pos_list = []
                for pos in self.roster_construction:
                    if pos in self.player_dict[k]["Position"]:
                        pos_list.append(1)
                    else:
                        pos_list.append(0)
                positions.append(np.array(pos_list))
            in_lineup = np.zeros(shape=len(ids))
            ownership = np.array(ownership)
            salaries = np.array(salaries)
            projections = np.array(projections)
            pos_matrix = np.array(positions)
            ids = np.array(ids)
            optimal_score = self.optimal_score
            salary_floor = self.min_lineup_salary
            salary_ceiling = self.salary
            max_pct_off_optimal = self.max_pct_off_optimal
            teams = np.array(teams)
            opponents = np.array(opponents)
            overlap_limit = self.overlap_limit
            problems = []
            num_players_in_roster = len(self.roster_construction)
            # creating tuples of the above np arrays plus which lineup number we are going to create
            for i in range(diff):
                lu_tuple = (
                    i,
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
                    opponents,
                    overlap_limit,
                    matchups,
                    num_players_in_roster,
                    self.site,
                )
                problems.append(lu_tuple)
            start_time = time.time()
            with mp.Pool() as pool:
                output = pool.starmap(self.generate_lineups, problems)
                print(
                    "number of running processes =",
                    (
                        pool.__dict__["_processes"]
                        if (pool.__dict__["_state"]).upper() == "RUN"
                        else None
                    ),
                )
                pool.close()
                pool.join()
            print("pool closed")
            self.update_field_lineups(output, diff)
            end_time = time.time()
            print("lineups took " + str(end_time - start_time) + " seconds")
            print(str(diff) + " field lineups successfully generated")

    def get_start_time(self, player_id):
        for _, player in self.player_dict.items():
            if player["ID"] == player_id:
                matchup = player["Matchup"]
                return self.game_info[matchup]
        return None

    def get_player_attribute(self, player_id, attribute):
        for _, player in self.player_dict.items():
            if player["ID"] == player_id:
                return player.get(attribute, None)
        return None

    def is_valid_for_position(self, player, position_idx):
        return any(
            pos in self.position_map[position_idx]
            for pos in self.get_player_attribute(player, "Position")
        )

    def sort_lineup_by_start_time(self, lineup):
        # Iterate over the entire roster construction
        for i, position in enumerate(
            self.roster_construction
        ):  # ['PG','SG','SF','PF','C','G','F','UTIL']
            # Only check G, F, and UTIL positions
            if position in ["G", "F", "UTIL"]:
                current_player = lineup[i]
                current_player_start_time = self.get_start_time(current_player)

                # Look for a swap candidate among primary positions
                for primary_i, primary_pos in enumerate(
                    self.roster_construction[:5]
                ):  # Only the primary positions (0 to 4)
                    primary_player = lineup[primary_i]
                    primary_player_start_time = self.get_start_time(primary_player)

                    # Check the conditions for the swap
                    if (
                        primary_player_start_time > current_player_start_time
                        and self.is_valid_for_position(primary_player, i)
                        and self.is_valid_for_position(current_player, primary_i)
                    ):
                        # Perform the swap
                        lineup[i], lineup[primary_i] = lineup[primary_i], lineup[i]
                        break  # Break out of the inner loop once a swap is made
        return lineup

    def update_field_lineups(self, output, diff):
        if len(self.field_lineups) == 0:
            new_keys = list(range(0, self.field_size))
        else:
            new_keys = list(
                range(
                    max(self.field_lineups.keys()) + 1,
                    max(self.field_lineups.keys()) + 1 + diff,
                )
            )

        nk = new_keys[0]
        for i, o in enumerate(output):
            lineup_list = sorted(next(iter(o.values()))["Lineup"])
            lineup_set = frozenset(lineup_list)

            # Keeping track of lineup duplication counts
            if lineup_set in self.seen_lineups:
                self.seen_lineups[lineup_set] += 1

                # Increase the count in field_lineups using the index stored in seen_lineups_ix
                self.field_lineups[self.seen_lineups_ix[lineup_set]]["Count"] += 1
            else:
                self.seen_lineups[lineup_set] = 1

                # Updating the field lineups dictionary
                if nk in self.field_lineups.keys():
                    print("bad lineups dict, please check dk_data files")
                else:
                    if self.site == "dk":
                        sorted_lineup = self.sort_lineup_by_start_time(
                            next(iter(o.values()))["Lineup"]
                        )
                    else:
                        sorted_lineup = next(iter(o.values()))["Lineup"]

                    self.field_lineups[nk] = next(iter(o.values()))
                    self.field_lineups[nk]["Lineup"] = sorted_lineup
                    self.field_lineups[nk]["Count"] += self.seen_lineups[lineup_set]
                    # Store the new nk in seen_lineups_ix for quick access in the future
                    self.seen_lineups_ix[lineup_set] = nk
                    nk += 1

    def calc_gamma(self, mean, sd):
        alpha = (mean / sd) ** 2
        beta = sd**2 / mean
        return alpha, beta

    @staticmethod
    def run_simulation_for_game(
        team1_id,
        team1,
        team2_id,
        team2,
        num_iterations,
        roster_construction,
    ):
        # Define correlations between positions

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

            if (
                player1["Team"] == player2["Team"]
                and player1["Position"][0] == player2["Position"][0]
            ):
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
                            players[i]["StdDev"] ** 2
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

        # Given eigenvalues and eigenvectors from previous code
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Set negative eigenvalues to zero
        eigenvalues[eigenvalues < 0] = 0

        # Reconstruct the matrix
        covariance_matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)

        try:
            samples = multivariate_normal.rvs(
                mean=[player["Fpts"] for player in game],
                cov=covariance_matrix,
                size=num_iterations,
            )
        except:
            print(team1_id, team2_id, "bad matrix")

        player_samples = []
        for i, player in enumerate(game):
            if "QB" in player["Position"]:
                sample = samples[:, i]
            else:
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
                prize_for_lineup = (
                    (
                        payout_cumsum[payout_index + lineup_count - 1]
                        - payout_cumsum[payout_index - 1]
                    )
                    / lineup_count
                    if payout_index != 0
                    else payout_cumsum[payout_index + lineup_count - 1] / lineup_count
                )
                combined_result_array[lineup_index] += prize_for_lineup
                payout_index += lineup_count
        return combined_result_array

    def run_tournament_simulation(self):
        self.print(f"Running {self.num_iterations} simulations")
        self.print(f"Number of unique field lineups: {len(self.field_lineups.keys())}")

        start_time = time.time()
        temp_fpts_dict = {}

        game_simulation_params = []
        self.print("Setting up game simulations...")
        for m in self.matchups:
            game_simulation_params.append(
                (
                    m[0],
                    self.teams_dict[m[0]],
                    m[1],
                    self.teams_dict[m[1]],
                    self.num_iterations,
                    self.roster_construction,
                )
            )

        total_games = len(game_simulation_params)
        self.print(f"Starting simulations for {total_games} games...")

        with mp.Pool() as pool:
            results = pool.starmap(self.run_simulation_for_game, game_simulation_params)
            completed_games = 0

            def update_progress(result):
                nonlocal completed_games
                completed_games += 1
                self.print(f"Completed game simulations: {completed_games}/{total_games}")
                results.append(result)

            # Start all tasks
            tasks = [pool.apply_async(self.run_simulation_for_game, args=params, callback=update_progress)
                     for params in game_simulation_params]

            # Wait for all tasks to complete
            for task in tasks:
                task.wait()


        self.print("Processing simulation results...")
        for res in results:
            temp_fpts_dict.update(res)

        # generate arrays for every sim result for each player in the lineup and sum
        self.print("Generating fantasy points arrays...")
        fpts_array = np.zeros(shape=(len(self.field_lineups), self.num_iterations))
        # converting payout structure into an np friendly format, could probably just do this in the load contest function
        field_lineups_count = np.array(
            [self.field_lineups[idx]["Count"] for idx in self.field_lineups.keys()]
        )

        total_lineups = len(self.field_lineups)
        processed_lineups = 0

        for index, values in self.field_lineups.items():
            try:
                fpts_sim = sum([temp_fpts_dict[player] for player in values["Lineup"]])
                fpts_array[index] = fpts_sim
                processed_lineups += 1
                if processed_lineups % max(1, total_lineups // 20) == 0:  # Update every 5%
                    self.print(f"Processed lineups: {processed_lineups}/{total_lineups}")
            except KeyError as e:
                for player in values["Lineup"]:
                    if player not in temp_fpts_dict.keys():
                        self.print(f"Missing player: {player}")

        self.print("Starting ranking computations...")
        self.print("Converting array to float16...")
        fpts_array = fpts_array.astype(np.float16)

        self.print(f"Computing rankings for {self.num_iterations} iterations...")
        # Break this into chunks for progress reporting
        chunk_size = 1000  # Process 1000 iterations at a time
        total_chunks = (self.num_iterations + chunk_size - 1) // chunk_size
        ranks_list = []

        for i in range(0, self.num_iterations, chunk_size):
            end_idx = min(i + chunk_size, self.num_iterations)
            chunk_ranks = np.argsort(-fpts_array[:, i:end_idx], axis=0).astype(np.uint32)
            ranks_list.append(chunk_ranks)
            self.print(f"Processed rankings chunk {(i // chunk_size) + 1}/{total_chunks}")

        self.print("Combining ranking results...")
        ranks = np.concatenate(ranks_list, axis=1)

        self.print("Calculating win statistics...")
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)

        self.print("Calculating cash statistics...")
        cashes, cash_counts = np.unique(
            ranks[0: len(list(self.payout_structure.values()))], return_counts=True
        )

        self.print("Calculating top 1% statistics...")
        top1pct_cutoff = math.ceil(0.01 * len(self.field_lineups))
        top1pct, top1pct_counts = np.unique(
            ranks[0:top1pct_cutoff, :], return_counts=True
        )

        self.print("Setting up payout structure...")
        payout_array = np.array(list(self.payout_structure.values()))
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))

        self.print("Starting ROI calculations...")
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))
        chunk_size = max(1000, self.num_iterations // 16)  # Ensure reasonable chunk size

        simulation_chunks = []
        for i in range(0, self.num_iterations, chunk_size):
            end_idx = min(i + chunk_size, self.num_iterations)
            simulation_chunks.append(
                (
                    ranks[:, i:end_idx].copy(),
                    payout_array,
                    self.entry_fee,
                    field_lineups_keys_array,
                    self.use_contest_data,
                    field_lineups_count,
                )
            )

        total_chunks = len(simulation_chunks)
        completed_chunks = 0
        results = []

        self.print(f"Processing {total_chunks} ROI chunks...")
        with mp.Pool() as pool:
            def chunk_callback(result):
                nonlocal completed_chunks
                completed_chunks += 1
                self.print(f"Processed ROI chunk: {completed_chunks}/{total_chunks}")
                results.append(result)

            chunk_tasks = [pool.apply_async(self.calculate_payouts, args=(chunk,), callback=chunk_callback)
                           for chunk in simulation_chunks]

            for task in chunk_tasks:
                task.wait()





        combined_result_array = np.sum(results, axis=0)

        total_sum = 0
        index_to_key = list(self.field_lineups.keys())
        for idx, roi in enumerate(combined_result_array):
            lineup_key = index_to_key[idx]
            lineup_count = self.field_lineups[lineup_key][
                "Count"
            ]  # Assuming "Count" holds the count of the lineups
            total_sum += roi * lineup_count
            self.field_lineups[lineup_key]["ROI"] += roi

        self.print("Updating final statistics...")
        for idx in self.field_lineups.keys():
            if idx in wins:
                self.field_lineups[idx]["Wins"] += win_counts[np.where(wins == idx)][0]
            if idx in top1pct:
                self.field_lineups[idx]["Top1Percent"] += top1pct_counts[
                    np.where(top1pct == idx)
                ][0]
            if idx in cashes:
                self.field_lineups[idx]["Cashes"] += cash_counts[
                    np.where(cashes == idx)
                ][0]

        end_time = time.time()
        diff = end_time - start_time
        self.print(
            f"{self.num_iterations} tournament simulations finished in {diff} seconds. Outputting."
        )

    def output(self):
        unique = {}
        for index, x in self.field_lineups.items():
            if len(x["Lineup"]) < 8:
                print(f"invalid lineup found: {len(x['Lineup'])} players.")
                continue
            lu_type = x["Type"]
            salary = 0
            fpts_p = 0
            fieldFpts_p = 0
            ceil_p = 0
            own_p = []
            lu_names = []
            lu_teams = []
            players_vs_def = 0
            def_opps = []
            for id in x["Lineup"]:
                for k, v in self.player_dict.items():
                    if v["ID"] == id:
                        salary += v["Salary"]
                        fpts_p += v["Fpts"]
                        fieldFpts_p += v["fieldFpts"]
                        ceil_p += v["Ceiling"]
                        own_p.append(v["Ownership"])
                        lu_names.append(v["DK Name"])
                        lu_teams.append(v["Team"])
                        continue
            counter = collections.Counter(lu_teams)
            stacks = counter.most_common()

            primaryStack = str(stacks[0][0]) + " " + str(stacks[0][1])
            secondaryStack = str(stacks[1][0]) + " " + str(stacks[1][1])

            own_s = np.sum(own_p)
            #own_p = np.prod(own_p) / 1000000
            win_p = round(x["Wins"] / self.num_iterations * 100, 2)
            top10_p = round((x["Top1Percent"] / self.num_iterations) * 100, 2)
            cash_p = round(x["Cashes"] / self.num_iterations * 100, 2)
            simDupes = x["Count"]

            roi_p = round(
                x["ROI"] / self.entry_fee / self.num_iterations * 1, 2
            )
            roi_round = round(x["ROI"] / self.num_iterations, 2)
            lineup_str = (
                f"{lu_names[0].replace('#', '-')}"
                f" ({x['Lineup'][0]}),"
                f"{lu_names[1].replace('#', '-')}"
                f" ({x['Lineup'][1]}),"
                f"{lu_names[2].replace('#', '-')}"
                f" ({x['Lineup'][2]}),"
                f"{lu_names[3].replace('#', '-')}"
                f" ({x['Lineup'][3]}),"
                f"{lu_names[4].replace('#', '-')}"
                f" ({x['Lineup'][4]}),"
                f"{lu_names[5].replace('#', '-')}"
                f" ({x['Lineup'][5]}),"
                f"{lu_names[6].replace('#', '-')}"
                f" ({x['Lineup'][6]}),"
                f"{lu_names[7].replace('#', '-')}"
                f" ({x['Lineup'][7]}),"
                f"{fpts_p},{fieldFpts_p},{ceil_p},{salary},{win_p}%,{top10_p}%,{roi_p}%,{cash_p}%,{own_s},${roi_round},{primaryStack},{secondaryStack},{lu_type},{simDupes}"
            )

            unique[index] = lineup_str

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../dk_output/dk_gpp_sim_lineups_{}_{}.csv".format(
                self.field_size, self.num_iterations
            ),
        )

        # Sort the `unique` dictionary by a criterion, e.g., ROI
        # Assuming `lineup_str` contains ROI as a percentage at a specific position
        sorted_unique = sorted(
            unique.items(),  # Convert dictionary to list of tuples
            key=lambda item: float(item[1].split(",")[14].replace("%", "")),  # Adjust index based on ROI position
            reverse=True,  # Sort descending if higher ROI is better
        )

        # Open the file and write the sorted data
        with open(out_path, "w") as f:
            # Write headers based on site and contest data

            f.write(
                "PG,SG,SF,PF,C,G,F,UTIL,Fpts Proj,Field Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Cash %,Own. Sum,Avg. Return,Stack1 Type,Stack2 Type,Lineup Type,Sim Dupes\n"
            )

            # Write each sorted lineup
            for _, lineup_str in sorted_unique:
                f.write(f"{lineup_str}\n")

        # input_csv_path = os.path.join(
        #     os.path.dirname(__file__),
        #     "../{}_data/{}".format(self.site, self.config["late_swap_path"]),


        input_csv_path = os.path.join(
            os.path.dirname(__file__),
            "../dk_data/{}".format(self.config["entries_path"]),
        )
        output_csv_path = os.path.join(
            os.path.dirname(__file__),
            "../dk_output/{}".format(self.config["entries_path"]), "_lineups",
        )
        # Read the input CSV file
        print(output_csv_path)
        with open(input_csv_path, "r") as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)
            print('------------------------------------------------------------------')
            for row in reader:
                print(row)
                print('------------------------------------------------------------------')

        output_csv_path = os.path.join(
            os.path.dirname(__file__),
            "../dk_output/{}".format(self.config["entries_path"]),
        )

        # Combine data from sorted_unique with the read CSV
        combined_data = []
        for i, row in enumerate(rows):
            if not row["Entry ID"] or row["Entry ID"] == "0":
                continue
            combined_row = {
                "Entry ID": row["Entry ID"],
                "Contest Name": row["Contest Name"],
                "Contest ID": row["Contest ID"],
                "Entry Fee": row["Entry Fee"],
            }
            # Split lineup_str from sorted_unique and map to columns
            _, lineup_str = sorted_unique[i]
            lineup_data = lineup_str.split(",")
            combined_row.update({
                "PG": lineup_data[0],
                "SG": lineup_data[1],
                "SF": lineup_data[2],
                "PF": lineup_data[3],
                "C": lineup_data[4],
                "G": lineup_data[5],
                "F": lineup_data[6],
                "UTIL": lineup_data[7],
            })
            combined_data.append(combined_row)
            print('------------------------------------')
            print(combined_row)

        # Write combined data to a new CSV file
        with open(output_csv_path, "w", newline="") as outfile:
            fieldnames = [
                "Entry ID", "Contest Name", "Contest ID", "Entry Fee",
                "PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_data)

        print(f"Combined data written to {output_csv_path}")

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../dk_output/dk_gpp_sim_player_exposure_{}_{}.csv".format(
                self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            f.write(
                "Player,Position,Team,Cash%,Win%,Top1%,Sim. Own%,Proj. Own%,Avg. Return\n"
            )
            unique_players = {}

            # Process lineups and aggregate data
            for val in self.field_lineups.values():
                for player in val["Lineup"]:
                    if player not in unique_players:
                        unique_players[player] = {
                            "Wins": val["Wins"],
                            "Cashes": val["Cashes"],
                            "Top1Percent": val["Top1Percent"],
                            "In": val["Count"],
                            "ROI": val["ROI"],
                        }
                    else:
                        unique_players[player]["Wins"] += val["Wins"]
                        unique_players[player]["Cashes"] += val["Cashes"]
                        unique_players[player]["Top1Percent"] += val["Top1Percent"]
                        unique_players[player]["In"] += val["Count"]
                        unique_players[player]["ROI"] += val["ROI"]

            # Create a list to hold sorted player data
            sorted_player_data = []

            # Process each player and calculate required percentages
            for player, data in unique_players.items():
                field_p = round(data["In"] / self.field_size * 100, 2)
                max_ranked = field_p / 100 * self.field_size * self.num_iterations

                # Avoid divide by zero
                if max_ranked == 0:
                    cash_p = 0
                    top10_p = 0
                    win_p = 0
                else:
                    cash_p = round(data["Cashes"] / max_ranked * 100, 2)
                    top10_p = round(data["Top1Percent"] / max_ranked * 100, 2)
                    win_p = round(data["Wins"] / max_ranked * 100, 2)

                roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 2)

                for k, v in self.player_dict.items():
                    if player == v["ID"]:
                        proj_own = v["Ownership"]
                        p_name = v["Name"]
                        position = "/".join(v.get("Position"))
                        team = v.get("Team")
                        break

                # Add to sorted list
                sorted_player_data.append(
                    {
                        "Player": p_name.replace("#", "-"),
                        "Position": position,
                        "Team": team,
                        "Cash%": cash_p,
                        "Win%": win_p,
                        "Top1%": top10_p,
                        "SimOwn%": field_p,
                        "ProjOwn%": proj_own,
                        "AvgReturn": roi_p,
                    }
                )

            # Sort players by ROI (descending order)
            sorted_player_data.sort(key=lambda x: x["AvgReturn"], reverse=True)

            # Write the sorted data to the CSV
            for player_data in sorted_player_data:
                f.write(
                    "{Player},{Position},{Team},{Cash%}%,{Win%}%,{Top1%}%,{SimOwn%}%,{ProjOwn%}%,${AvgReturn}\n".format(
                        **player_data
                    )
                )











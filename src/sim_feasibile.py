import pandas as pd
import numpy as np
import random

def simulate_feasibility_with_progress(data, max_salary=50000, lineup_size=8, num_samples=500000, print_every=10000):
    #Simulate valid lineups with random sampling and progress updates.
    min_salary = 0
    min_score = 0
    min_proj_points = 0
    #players = data[['DK Name', 'Salary', 'Points Proj', 'Ceiling']].to_dict(orient='records')
    eligible_players = data[data['Points Proj'] >= min_proj_points][
        ['DK Name', 'Salary', 'Points Proj']].to_dict(orient='records')
    feasibility = {player['DK Name']: 0 for player in eligible_players}  # Initialize feasibility counts

    print(f"Simulating {num_samples} random lineups...")

    i = 1  # Initialize i outside the loop
    while i <= num_samples:  # Use while instead of for to control incrementing i manually
        lineup = random.sample(eligible_players, lineup_size)  # Randomly select 8 players
        total_salary = sum(player['Salary'] for player in lineup)
        total_points = sum(player['Points Proj'] for player in lineup)

        # Only process lineups with total_salary > 49000
        if min_salary < total_salary <= max_salary and total_points > min_score:  # Check salary range
            for player in lineup:
                feasibility[player['DK Name']] += 1
            i += 1  # Increment i only if the lineup meets the salary criteria

        # Print progress every N samples
        if i % print_every == 0:
            print(f"Processed {i} / {num_samples} lineups...")

    # Normalize feasibility scores
    for player in feasibility:
        feasibility[player] /= num_samples

    print("Feasibility calculation completed.")
    return feasibility


def generate_random_scores(players, num_samples, sd_multiplier=1):
    random_scores = {}
    for player in players:
        proj_points = player['Points Proj']
        ceiling = player['Ceiling']

        # Assuming ceiling represents `sd_multiplier` standard deviations
        std_dev = ceiling / sd_multiplier

        # Generate random scores with Gaussian distribution
        scores = np.random.normal(loc=proj_points, scale=std_dev, size=num_samples)
        random_scores[player['DK Name']] = scores

    return random_scores



def simulate_weighted_feasibility_with_progress(data, max_salary=50000, lineup_size=8, num_samples=50000, print_every=10000):
    """
    Simulate valid lineups with position-based eligibility using the slot chart.
    """
    # Slot chart mapping positions to slots
    slot_map = {
        1: ['PG', 'PG/SG', 'PG/SF','G'],
        2: ['SG', 'PG/SG', 'G'],
        3: ['SF', 'SG/SF', 'PG/SF', 'SF/PF', 'F'],
        4: ['PF', 'SF/PF', 'PF/C', 'F'],
        5: ['C', 'PF/C'],
        6: ['PG', 'SG', 'PG/SG', 'SG/SF', 'PG/SF', 'G'],
        7: ['SF', 'PF', 'SF/PF', 'SG/SF', 'PG/SF', 'F'],
        8: ['PG', 'SG', 'SF', 'PF', 'C', 'PG/SG', 'SG/SF', 'SF/PF', 'PG/SF', 'PF/C', 'G', 'F', 'UTIL'],
    }

    min_salary =  49000
    min_score = 200
    min_proj_points = 12

    # Filter eligible players based on minimum projected points
    eligible_players = data[data['Points Proj'] >= min_proj_points].to_dict(orient='records')
    weighted_feasibility = {player['DK Name']: 0 for player in eligible_players}  # Initialize feasibility counts
    tournament_feasibility = {player['DK Name']: 0 for player in eligible_players}  # Initialize feasibility counts
    count = len(eligible_players)

    print(f"Simulating {num_samples} random lineups...{count} players...")
    lineups = []  # Store all generated lineups
    random_scores = generate_random_scores(eligible_players, num_samples)

    i = 1  # Initialize iteration counter
    #for _ in range(num_samples):
    while i <= num_samples:  # Use while instead of for to control incrementing i manually
        lineup = []
        selected_players = set()  # Track selected players to avoid duplicates

        for slot in range(1, lineup_size + 1):
            # Filter players eligible for the current slot
            slot_positions = slot_map[slot]
            slot_eligible_players = [
                player for player in eligible_players
                if player['DK Name'] not in selected_players and player['Position'] in slot_positions
            ]

            # Ensure there are eligible players for the slot
            if not slot_eligible_players:
                break  # Skip this lineup if no eligible players for the slot

            # Randomly select a player for the slot
            selected_player = random.choice(slot_eligible_players)
            lineup.append(selected_player)
            selected_players.add(selected_player['DK Name'])

        # Validate lineup
        if len(lineup) == lineup_size:
            total_salary = sum(player['Salary'] for player in lineup)
            total_points = sum(random.choice(random_scores[player['DK Name']]) for player in lineup)

            if min_salary < total_salary <= max_salary and total_points > min_score:
                i += 1  # Increment iteration only for valid lineups
                for player in lineup:
                    weighted_feasibility[player['DK Name']] += 1
                lineups.append({
                    'lineup': lineup,
                    'total_points': total_points
                })

                # Print progress every N samples
                if i % print_every == 0:
                    print(f"Processed {i} / {num_samples} lineups...")
                    print(f"Salary: {total_salary}  Points: {total_points}")


        # Sort lineups by total_points in descending order
    lineups.sort(key=lambda x: x['total_points'], reverse=True)

    top_lineups = int(num_samples * 0.20)
    lineups = lineups[:top_lineups]

    # Assign feasibility scores based on ranking
    for rank, lineup_info in enumerate(lineups, start=1):
        lineup = lineup_info['lineup']
        for player in lineup:
            tournament_feasibility[player['DK Name']] += 1 / rank

    # Normalize feasibility scores
    for player in tournament_feasibility:
        tournament_feasibility[player] /= num_samples

    #for player in weighted_feasibility:
      #  weighted_feasibility[player] /= num_samples / 1000

    # Ensure all players have a feasibility score (fill missing with zero)
    for player_name in data['DK Name']:
        if player_name not in tournament_feasibility:
            tournament_feasibility[player_name] = 0

    for player_name in data['DK Name']:
        if player_name not in weighted_feasibility:
            weighted_feasibility[player_name] = 0

    print("Tournament Feasibility calculation completed.")
    return weighted_feasibility, tournament_feasibility


# Load the dataset
file_path = 'dataset_23.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Inspect the dataset
print("Dataset Overview:")
print(data.head())

# Add Feasibility Scores
data['Feasibility'] = 0.0  # Initialize feasibility column

# Calculate feasibility for each contest
for contest_id, group in data.groupby('Contest ID'):
    print(f"Processing Contest ID: {contest_id}")
    feasibility = simulate_feasibility_with_progress(
        group, max_salary=50000, lineup_size=8, num_samples=200000, print_every=50000
    )
    # Map feasibility back to the dataset
    data.loc[group.index, 'Feasibility'] = group['DK Name'].map(feasibility)

# Verify the updated dataset
print("\nDataset with Feasibility Scores:")
print(data[['Contest ID', 'DK Name', 'Feasibility']].head())
for contest_id, group in data.groupby('Contest ID'):
    print(f"Feasibility Score Sum for Contest {contest_id}: {group['Feasibility'].sum()}")


# Add Feasibility Scores
data['Weighted Feasibility'] = 0.0  # Initialize feasibility column
data['Tournament Feasibility'] = 0.0

# Calculate feasibility for each contest
for contest_id, group in data.groupby('Contest ID'):
    print(f"Processing Contest ID: {contest_id}")
    weighted_feasibility, tournament_feasibility = simulate_weighted_feasibility_with_progress(
        group, max_salary=50000, lineup_size=8, num_samples=10000, print_every=200
    )
    # Map feasibility back to the dataset
    data.loc[group.index, 'Weighted Feasibility'] = group['DK Name'].map(weighted_feasibility)
    data.loc[group.index, 'Tournament Feasibility'] = group['DK Name'].map(tournament_feasibility)

# Verify the updated dataset
print("\nDataset with Weighted Feasibility Scores:")
print(data[['Contest ID', 'DK Name', 'Weighted Feasibility']].head())
for contest_id, group in data.groupby('Contest ID'):
    print(f"Weighted Feasibility Score Sum for Contest {contest_id}: {group['Weighted Feasibility'].sum()}")

print("\nDataset with Tournament Feasibility Scores:")
print(data[['Contest ID', 'DK Name', 'Tournament Feasibility']].head())
for contest_id, group in data.groupby('Contest ID'):
    print(f"Tournament Feasibility Score Sum for Contest {contest_id}: {group['Tournament Feasibility'].sum()}")

# Save the updated dataset (optional)
output_file = 'dataset_with_feasibility_weighted.csv'
data.to_csv(output_file, index=False)
print(f"Updated dataset saved to {output_file}")

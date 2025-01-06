import requests
import pandas as pd
import os

# The Odds API Key and URL
api_key = 'd4237a37fb55c03282af5de33235e1d6'
url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={api_key}&regions=us&markets=spreads,totals'


# Fetch data from The Odds API
def fetch_odds_data():
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        games = []
        for event in data:
            if "bookmakers" in event and event["bookmakers"]:
                for bookmaker in event["bookmakers"]:
                    # Only process DraftKings data
                    if bookmaker["title"] != "DraftKings":
                        continue

                    spread_market = None
                    total_market = None
                    for market in bookmaker.get("markets", []):
                        if market["key"] == "spreads":
                            spread_market = market
                        elif market["key"] == "totals":
                            total_market = market

                    # Extract spreads and totals
                    spread_data = {}
                    if spread_market:
                        for outcome in spread_market.get("outcomes", []):
                            spread_data[outcome["name"]] = outcome.get("point", "N/A")

                    # Extract game totals
                    total = None
                    if total_market:
                        for outcome in total_market.get("outcomes", []):
                            total = outcome.get("point", "N/A")

                    # Create rows with only required columns
                    for team, spread in spread_data.items():
                        games.append({
                            "Team": team,
                            "Spread": spread,
                            "Total": total
                        })
        return pd.DataFrame(games)
    else:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")


# Save to CSV
def save_to_csv(dataframe):
    # Set the path to save the CSV in the same folder as the script
    csv_path = os.path.join(os.path.dirname(__file__), "odds_data.csv")

    # Save the DataFrame to a CSV file
    dataframe.to_csv(csv_path, index=False)
    print(f"Data successfully saved to {csv_path}")


# Main function
def main():
    try:
        # Fetch data
        odds_data = fetch_odds_data()

        # Print the DataFrame to the console
        print("Fetched Odds Data from DraftKings:")
        print(odds_data)

        # Save to CSV
        save_to_csv(odds_data)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

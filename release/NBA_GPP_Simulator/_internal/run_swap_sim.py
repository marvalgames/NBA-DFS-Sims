import sys
import nba_swap_sims


def main():
    print(f"run_swap_sim.py has started.")  # Add this line
    if len(sys.argv) < 7:  # Ensure at least four arguments are passed
        print(f"Usage: {sys.argv[0]} <num_iterations> <site> <num_uniques>")
        sys.exit(1)

    # Parse command-line arguments
    try:
        num_iterations = int(sys.argv[1])
        site = sys.argv[2]
        num_uniques = int(sys.argv[3])
        num_lineup_sets = int(sys.argv[4])
        min_salary = int(sys.argv[5])
        projection_minimum = int(sys.argv[6])
        contest_path = sys.argv[7]
    except ValueError:
        print("Error: num_iterations and num_uniques must be integers")
        sys.exit(1)

    # Run the simulation
    sim_to = nba_swap_sims.NBA_Swaptimizer_Sims(num_iterations, site, num_uniques, num_lineup_sets, min_salary, projection_minimum, contest_path)
    sim_to.swaptimize()
    sim_to.compute_best_guesses_parallel()
    sim_to.run_tournament_simulation()
    sim_to.output()

if __name__ == '__main__':
    main()


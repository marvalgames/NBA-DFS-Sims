import sys
import nba_swap_sims


def main():
    print("run_swap_sim.py has started.")  # Add this line
    if len(sys.argv) < 4:  # Ensure at least three arguments are passed
        print(f"Usage: {sys.argv[0]} <num_iterations> <site> <num_uniques>")
        sys.exit(1)

    # Parse command-line arguments
    try:
        num_iterations = int(sys.argv[1])
        site = sys.argv[2]
        num_uniques = int(sys.argv[3])
    except ValueError:
        print("Error: num_iterations and num_uniques must be integers")
        sys.exit(1)

    print(f"Arguments received: num_iterations={num_iterations}, site={site}, num_uniques={num_uniques}")

    print('Swap Sim Module Started')
    print(f"Arguments: num_iterations={num_iterations}, site={site}, num_uniques={num_uniques}")


    # Run the simulation
    sim_to = nba_swap_sims.NBA_Swaptimizer_Sims(num_iterations, site, num_uniques)
    sim_to.swaptimize()
    sim_to.compute_best_guesses_parallel()
    sim_to.run_tournament_simulation()
    sim_to.output()

if __name__ == '__main__':
    main()


# import sys
# import nba_swap_sims
#
#
# def main():
#     print(f"run_swap_sim.py has started.")
#     if len(sys.argv) < 7:
#         print(
#             f"Usage: {sys.argv[0]} <num_iterations> <site> <num_uniques> <num_lineup_sets> <min_salary> <projection_minimum> <contest_path>")
#         sys.exit(1)
#
#     try:
#         num_iterations = int(sys.argv[1])
#         site = sys.argv[2]
#         num_uniques = int(sys.argv[3])
#         num_lineup_sets = int(sys.argv[4])
#         min_salary = int(sys.argv[5])
#         projection_minimum = int(sys.argv[6])
#         contest_path = sys.argv[7]
#
#         # Create simulation instance with subprocess flag
#         sim_to = nba_swap_sims.NBA_Swaptimizer_Sims(
#             num_iterations=num_iterations,
#             site=site,
#             num_uniques=num_uniques,
#             num_lineup_sets=num_lineup_sets,
#             min_salary=min_salary,
#             projection_minimum=projection_minimum,
#             contest_path=contest_path,
#             is_subprocess=True  # Add this flag
#         )
#
#         # Run simulation steps
#         sim_to.swaptimize()
#         sim_to.compute_best_guesses_parallel()
#         sim_to.run_tournament_simulation()
#         sim_to.output()
#
#         print("Simulation completed successfully")
#         sys.exit(0)
#
#     except Exception as e:
#         print(f"Error during simulation: {str(e)}", file=sys.stderr)
#         sys.exit(1)
#
#
# if __name__ == '__main__':
#     main()
#

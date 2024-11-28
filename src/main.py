import sys
from nba_optimizer import *
from nba_late_swaptimizer import NBA_Late_Swaptimizer


def main():
    site = "dk"
    process = "opto"
    process = "swap_sim"
    #process = "sim"

    num_lineups = 20
    num_uniques = 1
    use_contest_data = True
    field_size = 1000
    use_file_upload = False
    num_iterations = 500

    if process == 'opto':
        opto = NBA_Optimizer(site, num_lineups, num_uniques)
        opto.optimize()
        opto.output()

    elif process == 'sim':
        import nba_gpp_simulator
        sim = nba_gpp_simulator.NBA_GPP_Simulator(site, field_size, num_iterations, use_contest_data,
                                                  use_file_upload)
        sim.generate_field_lineups()
        sim.run_tournament_simulation()
        sim.output()

    elif process == 'swap':
        num_uniques = 1
        swap_to = NBA_Late_Swaptimizer(site, num_uniques)
        swap_to.swaptimize()
        swap_to.output()

    elif process == 'swap_sim':
        import nba_swap_sims
        sim_to = nba_swap_sims.NBA_Swaptimizer_Sims(num_iterations, site, num_uniques)
        sim_to.swaptimize()
        sim_to.compute_best_guesses_parallel()
        sim_to.run_tournament_simulation()
        sim_to.output()


if __name__ == "__main__":
    main()



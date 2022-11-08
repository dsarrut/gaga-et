#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from helpers_cmd_line import *

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("json_param", nargs=-1)
@click.option("--russian_roulette", "--rr", default=None, help="Russian Roulette")
@common_simu_options
def go(json_param, output_folder, visu, activity_bq, threads, rad, russian_roulette):
    # get the param dic structure
    param = read_json_param(json_param)

    # update param from command line
    update_param(param, visu, threads, activity_bq, rad)

    # start transcript
    param.output_folder = Transcript.start(output_folder)

    # rr ?
    if russian_roulette:
        param.russian_roulette = russian_roulette
    if 'russian_roulette' not in param or param.russian_roulette is None:
        param.russian_roulette = 100

    # print param
    print(json.dumps(param, indent=4, sort_keys=False))

    # create the simu
    sim = make_arf_training_dataset_simulation(param)

    # run
    sim.initialize()
    sim.start()

    # print stats at the end
    stats = sim.get_actor("Stats")
    print(stats)
    print(stats.user_info.output)

    # stop transcript
    Transcript.stop()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()

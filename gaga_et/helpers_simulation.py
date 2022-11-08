#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def set_default_ui(sim, param):
    ui = sim.user_info
    ui.g4_verbose = False
    ui.g4_verbose_level = 1
    ui.visu = param.visu
    ui.number_of_threads = param.number_of_threads

    # special case for visu
    if ui.visu:
        print("Visualization is ON, set very low activity, one single thread and no CT")
        ui.number_of_threads = 1
        param.activity_bq = 100
    return ui

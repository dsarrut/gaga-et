#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gaga_et
import opengate as gate


def make_training_dataset_pet_simulation(sim, param):
    # main options
    gaga_et.set_default_ui(sim, param)

    # compute largest dimension
    gaga_et.set_default_world(sim, param)

    # spherical phase space
    gaga_et.add_sphere_phsp(sim, param)

    # CT phantom
    gaga_et.add_ct_image(sim, param)

    # activity source
    if "source_image" not in param or param["source_image"] == "":
        gaga_et.create_uniform_activity_source(param)
    gaga_et.print_activity(param)
    gaga_et.add_positron_vox_activity_source(sim, param)

    # add stat actor
    gaga_et.add_stat_actor(sim, param)

    # add phsp actor
    gaga_et.add_pet_phsp_actor(sim, param)

    # physics
    gaga_et.set_default_physics(sim)

    # param
    Bq = gate.g4_units('Bq')
    MBq = 1e6 * Bq
    print()
    print(f'CT image       : {param.ct_image}')
    print(f'Activity image : {param.source_image}')
    print(f'Activity       : {param.activity_bq * Bq / MBq} MBq')
    print(f'Radionuclide   : {param.radionuclide}')
    print(f'Threads        : {sim.user_info.number_of_threads}')

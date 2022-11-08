#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate


def set_default_physics(sim):
    p = sim.get_physics_user_info()
    p.physics_list_name = 'G4EmStandardPhysics_option4'
    p.enable_decay = False
    p.apply_cuts = True
    cuts = p.production_cuts
    m = gate.g4_units('m')
    cuts.world.gamma = 1 * m
    cuts.world.positron = 1 * m
    cuts.world.electron = 1 * m
    mm = gate.g4_units('mm')
    cuts.ct.gamma = 1 * mm
    cuts.ct.positron = 1 * mm
    cuts.ct.electron = 1 * mm

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


def add_stat_actor(sim):
    stats = sim.add_actor('SimulationStatisticsActor', 'stats')
    stats.track_types_flag = True
    return stats


def add_pet_phsp_actor(sim, param):
    # PHSP Actor
    phsp = sim.add_actor('PhaseSpaceActor', 'phase_space')
    phsp.mother = 'spherical_phase_space'

    # we use PrePosition because this is the first step in the volume
    phsp.attributes = [
        # output attributes:
        'KineticEnergy',
        'PrePosition',
        'PreDirection',
        'TimeFromBeginOfEvent',
        # input attributes:
        'EventID',
        'EventKineticEnergy',
        'EventPosition',
        # 'EventDirection',
        'EventTrackVertexMomentumDirection'  # not really used
    ]
    phsp.output = str(Path(param.output_folder) / 'pet_ct.root')

    # the following option is required to store also event that do not exit the phantom
    phsp.store_absorbed_event = True

    # filter gamma only
    f = sim.add_filter('ParticleFilter', 'f')
    f.particle = 'gamma'
    phsp.filters.append(f)

    return phsp

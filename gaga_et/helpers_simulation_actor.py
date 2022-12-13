#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import opengate as gate
import opengate.contrib.spect_ge_nm670 as gate_spect


def add_stat_actor(sim, param):
    stats = sim.add_actor('SimulationStatisticsActor', 'stats')
    stats.track_types_flag = True
    stats.output = Path(param.output_folder) / "stats.txt"
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
        'EventDirection',  # not really used
    ]
    phsp.output = str(Path(param.output_folder) / 'pet_ct.root')

    # the following option is required to store also event that do not exit the phantom
    phsp.store_absorbed_event = True

    # filter gamma only
    f = sim.add_filter('ParticleFilter', 'f')
    f.particle = 'gamma'
    phsp.filters.append(f)

    return phsp


def add_spect_phsp_actor(sim, param):
    """
    For the moment, this is almost equivalent to the pet counterpart. It may change in the future.
    """
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
        'EventDirection',
    ]
    phsp.output = str(Path(param.output_folder) / 'spect_ct.root')

    # the following option is required to store also event that do not exit the phantom
    phsp.store_absorbed_event = True

    # filter gamma only
    f = sim.add_filter('ParticleFilter', 'f')
    f.particle = 'gamma'
    phsp.filters.append(f)

    return phsp


def add_digitizer(sim, param, add_fake_channel=False):
    mm = gate.g4_units("mm")
    keV = gate.g4_units("keV")
    for i in range(param.spect_heads):
        fn = f'{Path(param.output_folder) / "ref"}_projection_{i}.mhd'
        channels = gate_spect.get_simplified_digitizer_channels_rad(
            f"spect_{i}", param.radionuclide, True
        )
        # add a "fake" first channel to have as many channel as the ARF
        if add_fake_channel:
            spect_name = f"spect_{i}"
            channels.insert(
                0,
                {"name": f"spectrum_{spect_name}", "min": 0 * keV, "max": 10000 * keV},
            )
        proj = gate_spect.add_digitizer(sim, f"spect_{i}_crystal", channels)
        proj.spacing = [4.41806 * mm, 4.41806 * mm]
        proj.output = fn


def rotate_one_angle(sim, heads, param):
    n = 1
    angle = float(param.angle)
    for head in heads:
        motion = sim.add_actor("MotionVolumeActor", f"Move_{head.name}")
        motion.mother = head.name
        motion.translations, motion.rotations = gate.volume_orbiting_transform(
            "z", angle, angle, n, head.translation, head.rotation
        )
        motion.priority = 5
    return n


def rotate_gantry(sim, heads, param):
    n = param.spect_angles
    for head in heads:
        motion = sim.add_actor("MotionVolumeActor", f"Move_{head.name}")
        motion.mother = head.name
        motion.translations, motion.rotations = gate.volume_orbiting_transform(
            "z", 0, 360 / param.spect_heads, n, head.translation, head.rotation
        )
        motion.priority = 5
    return n

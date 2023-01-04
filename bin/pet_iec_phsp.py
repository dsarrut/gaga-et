#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import random
import string

import opengate as gate
import opengate.contrib.phantom_nema_iec_body as gate_iec
import numpy as np
import click
import os
import sys


class Transcript(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    @staticmethod
    def start(filename):
        """Start transcript, appending print output to given filename"""
        sys.stdout = Transcript(filename)

    @staticmethod
    def stop():
        """Stop transcript and return print functionality to normal"""
        sys.stdout.logfile.close()
        sys.stdout = sys.stdout.terminal


# -----------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--output_folder', '-o', default='auto', help='Output folder; if "auto", random folder name.')
@click.option('--spheres', '-s', type=(int, float), multiple=True,
              help='Activity concentration (kBq/mL) in spheres 10 13 17 22 28 37')
@click.option('--bg', default=0.0, help='Background activity concentration (kBq/mL)')
@click.option('--insert', default=0.0, help='Central insert activity (kBq/mL)')
@click.option('--rad', default='F18', help='Beta+ emitters F18, Ga68')
@click.option('--tung', default=False, is_flag=True, help='Add a tungsten highZ element')
@click.option('--tungs', default=0.0, help='Add source in tungsten also')
@click.option('--threads', '-t', default=1, help='Number of threads')
@click.option('--uniform', '-u', default=0.0, help='Uniform source, activity concentration')
def simulation(output_folder, spheres, threads, bg, insert, rad, tung, tungs, uniform):
    # output folder # FIXME TO PUT IN gate_HELPER (?)
    if output_folder == "auto":
        r = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        r = 'run.' + r
        if not os.path.exists(r):
            os.mkdir(r)
        if not os.path.isdir(r):
            gate.fatal(f'Error {r} is not a folder')
        output_folder = r

    # Start log to file (every 'print' will be also put in the log file)
    print(f'Output folder: {output_folder}')
    Transcript.start(f'{output_folder}/simu.log')
    current_time = datetime.datetime.now()
    print(f'Now: {current_time}')
    print(f'Output folder: {output_folder}')
    print(f'Activity spheres:        {spheres} kBq')
    print(f'Activity background:     {bg} kBq')
    print(f'Activity central insert: {insert} kBq')
    print(f'Params activity sp={spheres}')
    print(f'Params activity bg={bg} ins={insert} unif={uniform}')
    print(f'Params threads={threads} {rad} tung={tung} ')

    # create the simulation
    sim = gate.Simulation()

    # main options
    ui = sim.user_info
    ui.g4_verbose = False
    ui.g4_verbose_level = 1
    ui.visu = False
    ui.number_of_threads = threads

    # define units
    mm = gate.g4_units('mm')
    nm = gate.g4_units('nm')
    m = gate.g4_units('m')
    cm = gate.g4_units('cm')
    cm3 = gate.g4_units('cm3')
    KeV = gate.g4_units('keV')
    Bq = gate.g4_units('Bq')
    kBq = Bq * 1000
    sec = gate.g4_units('second')
    mL = gate.g4_units('mL')

    # change world size
    world = sim.world
    world.size = [3 * m, 3 * m, 3 * m]
    world.material = 'G4_AIR'
    # world.color = [0, 0, 0, 0]

    # cylinder
    '''cyl = sim.add_volume('Tubs', 'phase_space_cylinder')
    cyl.rmin = 300 * mm
    cyl.rmax = cyl.rmin + 1 * nm
    cyl.dz = 100 * cm  # Warning dZ is half-length !
    cyl.color = [1, 1, 1, 1]
    cyl.material = 'G4_AIR'''''

    cyl = sim.add_volume('Sphere', 'phase_space_cylinder') # not a cylinder, but a sphere
    cyl.rmin = 210 * mm
    cyl.rmax = 211 * mm
    cyl.color = [0, 1, 0, 1]
    cyl.material = 'G4_AIR'

    # add a iec phantom
    iec_phantom = gate_iec.add_phantom(sim)

    # additional element ?
    if tung:
        vint = sim.get_volume_user_info('iec_interior')
        print(vint)
        t = sim.add_volume('Box', 'tung')
        t.mother = vint.name
        t.size = [3 * cm, 8 * cm, 10 * cm]
        t.translation = [-9 * cm, 5 * cm, 5 * cm]
        t.material = 'G4_CADMIUM_TUNGSTATE'
        t.color = [0, 0, 1, 1]

    # spheres diameters
    diameters = [10, 13, 17, 22, 28, 37]

    # uniform source ?
    if uniform != 0:
        print('error')
        exit()
        uniform = float(uniform)
        print('Uniform activity concentration', uniform)
        # set all other sources to zero
        # spheres = []
        # for d in diameters:
        #    spheres.append([int(d), 0])
        # insert = 0
        # bg = 0
        # create a uniform source
        suni = sim.add_source('Generic', 'suni')
        suni.mother = iec_phantom.name
        print(f'Uniform source volume in a global box larger than iec')
        suni.position.type = 'box'
        suni.position.size = gate.get_volume_bounding_size(sim, suni.mother)
        print(suni.position.size)
        # suni.position.size = [a * 1.2 for a in suni.position.size]
        # print(suni.position.size)
        volume = suni.position.size[0] * suni.position.size[1] * suni.position.size[2] / cm3
        print('Volume of the box source in cm3', volume)
        suni.particle = 'e+'
        suni.energy.type = rad
        suni.activity = uniform * kBq / ui.number_of_threads * volume
        print(f'Activity of {suni.activity / kBq} kBq. '
              f'Concentration: {suni.activity / volume / kBq} kBq/mL')
        print(suni)

    # simple source (gamma, 511 keV) activity
    activity_concentrations = np.zeros(6)
    for s in spheres:
        print(s)
        i = diameters.index(s[0])
        # vol = 4 / 3 * np.pi * np.power(s[0] / mm / 2, 3) * 0.001
        activity_concentrations[i] = s[1] * kBq / ui.number_of_threads
        # print(f'Activity in sphere {s[0]} : {activity_concentrations[i] * vol / kBq} kBq')

    sources = gate_iec.add_spheres_sources(sim, 'iec', 'iec_sources',
                                          diameters, activity_concentrations)
    for s in sources:
        s.particle = 'e+'
        s.energy.type = rad
        print(f'Source activity {s.name} : {s.activity / Bq} Bq {s.energy}')

    # Background source #1
    if insert != 0:
        sbg1 = sim.add_source('Generic', 'sbg1')
        sbg1.mother = f'iec_center_cylinder_hole'
        v = sim.get_volume_user_info(sbg1.mother)
        s = sim.get_solid_info(v)
        bg_volume = s.cubic_volume / cm3
        sbg1.position.type = 'box'
        sbg1.position.size = gate.get_volume_bounding_size(sim, sbg1.mother)
        sbg1.position.confine = sbg1.mother
        sbg1.particle = 'e+'
        sbg1.energy.type = rad
        sbg1.activity = insert * kBq / ui.number_of_threads * bg_volume
        print(f'Activity of {sbg1.mother} {sbg1.activity / kBq} kBq. '
              f'Concentration: {sbg1.activity / bg_volume / kBq} kBq/mL')

    # background source #2
    if bg != 0:
        sbg2 = sim.add_source('Generic', 'sbg2')
        sbg2.mother = f'iec_interior'
        v = sim.get_volume_user_info(sbg2.mother)
        s = sim.get_solid_info(v)
        bg_volume = s.cubic_volume / cm3
        sbg2.position.type = 'box'
        sbg2.position.size = gate.get_volume_bounding_size(sim, sbg2.mother)
        sbg2.position.confine = sbg2.mother
        sbg2.particle = 'e+'
        sbg2.energy.type = rad
        sbg2.activity = bg * kBq / ui.number_of_threads * bg_volume
        print(f'Volume of   {sbg2.mother} {bg_volume} cm3')
        print(f'Activity of {sbg2.mother} {sbg2.activity / kBq} kBq. '
              f'Concentration: {sbg2.activity / bg_volume / kBq} kBq/mL')

    # background source #2
    if tungs != 0:
        sbg3 = sim.add_source('Generic', 'sbg3')
        sbg3.mother = f'tung'
        v = sim.get_volume_user_info(sbg3.mother)
        s = sim.get_solid_info(v)
        bg_volume = s.cubic_volume / cm3
        sbg3.position.type = 'box'
        sbg3.position.size = gate.get_volume_bounding_size(sim, sbg3.mother)
        sbg3.position.confine = sbg3.mother
        sbg3.particle = 'e+'
        sbg3.energy.type = rad
        sbg3.activity = bg * kBq / ui.number_of_threads * bg_volume
        print(f'Volume of   {sbg3.mother} {bg_volume} cm3 tungs')
        print(f'Activity of {sbg3.mother} {sbg3.activity / kBq} kBq. '
              f'Concentration: {sbg3.activity / bg_volume / kBq} kBq/mL')

    # add stat actor
    stats = sim.add_actor('SimulationStatisticsActor', 'stats')
    stats.track_types_flag = True

    # filter gamma only
    f = sim.add_filter('ParticleFilter', 'f')
    f.particle = 'gamma'

    # Hits tree Actor
    phsp = sim.add_actor('PhaseSpaceActor', 'phase_space')
    phsp.mother = 'phase_space_cylinder'
    # we use PrePosition because this is the first step in the volume
    phsp.attributes = ['KineticEnergy', 'PrePosition', 'PreDirection',
                       'TimeFromBeginOfEvent', 'EventID',
                       'EventKineticEnergy',
                       # 'TrackVertexKineticEnergy',
                       # 'TrackVertexMomentumDirection',
                       'EventDirection', 'EventPosition',
                       ]
    phsp.output = f'{output_folder}/pet_iec.root'
    phsp.store_absorbed_event = True  ## FIXME
    phsp.filters.append(f)
    print(phsp)

    # physics
    p = sim.get_physics_user_info()
    p.physics_list_name = 'G4EmStandardPhysics_option4'
    p.enable_decay = False
    p.apply_cuts = True
    cuts = p.production_cuts
    mm = gate.g4_units('mm')
    cuts.world.gamma = 1 * mm
    cuts.world.positron = 1 * mm
    cuts.world.electron = 1 * mm
    cuts.iec.gamma = 0.1 * mm
    cuts.iec.positron = 0.1 * mm
    cuts.iec.electron = 0.1 * mm

    # initialize & start
    # ui.running_verbose_level = gam.EVENT
    output = sim.start()

    # output stat
    stats = output.get_actor('stats')
    print(stats)
    stats.write(f'{output_folder}/pet_iec_stat.txt')

    # end log to file
    end_time = datetime.datetime.now()
    print(f'Now:      {end_time}')
    print(f'Duration: {end_time - current_time}')
    Transcript.stop()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    simulation()

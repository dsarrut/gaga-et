#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gaga_et
import opengate as gate
import pathlib
import opengate.contrib.spect_ge_nm670 as gate_spect


def make_arf_training_dataset_simulation(sim, param):
    # main options
    ui = sim.user_info
    ui.g4_verbose = False
    ui.visu = param.visu
    ui.number_of_threads = param.number_of_threads
    ui.verbose_level = gate.INFO

    # convert str to pathlib
    param.output_folder = pathlib.Path(param.output_folder)

    # units
    m = gate.g4_units("m")
    mm = gate.g4_units("mm")
    nm = gate.g4_units("nm")
    cm = gate.g4_units("cm")
    Bq = gate.g4_units("Bq")
    MeV = gate.g4_units("MeV")

    # activity
    activity = param.activity_bq * Bq / ui.number_of_threads
    if ui.visu:
        activity = 1e2 * Bq
        ui.number_of_threads = 1

    # world size
    world = sim.world
    world.size = [1 * m, 1 * m, 1 * m]
    world.material = "G4_Galactic"

    # spect head
    colli_name = gate_spect.get_collimator(param.radionuclide)
    print(f"Collimator : {colli_name}")
    spect = gate_spect.add_ge_nm67_spect_head(
        sim, "spect", collimator_type=colli_name, debug=ui.visu
    )
    crystal_name = f"{spect.name}_crystal"

    # physic list
    p = sim.get_physics_user_info()
    p.physics_list_name = "G4EmStandardPhysics_option4"
    sim.set_cut("world", "all", 1 * mm)

    # detector input plane
    detector_plane = sim.add_volume("Box", "detPlane")
    detector_plane.mother = "spect"
    detector_plane.size = [57.6 * cm, 44.6 * cm, 1 * nm]
    pos, crystal_distance, psd = gate_spect.get_plane_position_and_distance_to_crystal(
        colli_name
    )
    pos += 1 * nm
    detector_plane.translation = [0, 0, pos]
    detector_plane.material = "G4_Galactic"
    detector_plane.color = [1, 0, 0, 1]

    # source
    s1 = sim.add_source("Generic", "s1")
    s1.particle = "gamma"
    s1.activity = activity
    s1.position.type = "disc"
    s1.position.radius = 57.6 * cm / 4  # divide by 4, arbitrarily
    s1.position.translation = [0, 0, 12 * cm]
    s1.direction.type = "iso"
    s1.energy.type = "range"
    s1.energy.min_energy = 0.01 * MeV
    w, ene = gate.get_rad_gamma_energy_spectrum(param.radionuclide)
    s1.energy.max_energy = max(ene) * 1.001
    print(f"Energy spectrum {ene}")
    print(f"Max energy  {s1.energy.max_energy}")
    s1.direction.acceptance_angle.volumes = [detector_plane.name]
    s1.direction.acceptance_angle.intersection_flag = True

    # digitizer
    channels = gate_spect.get_simplified_digitizer_channels_rad(
        "spect", param.radionuclide, scatter_flag=True
    )
    cc = gate_spect.add_digitizer_energy_windows(sim, crystal_name, channels)

    # arf actor for building the training dataset
    arf = sim.add_actor("ARFTrainingDatasetActor", "ARF (training)")
    arf.mother = detector_plane.name
    arf.output = param.output_folder / "arf_training_dataset.root"
    arf.energy_windows_actor = cc.name
    arf.russian_roulette = param.russian_roulette

    # add stat actor
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.track_types_flag = True
    stats.output = param.output_folder / "arf_training_dataset_stats.txt"


def make_training_dataset_simulation(sim, param):
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
    gaga_et.add_gamma_vox_activity_source(sim, param)

    # add stat actor
    gaga_et.add_stat_actor(sim, param)

    # add phsp actor
    gaga_et.add_spect_phsp_actor(sim, param)

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


def make_spect_simulation(sim, param):
    # main options
    gaga_et.set_default_ui(sim, param)

    # compute largest dimension
    gaga_et.set_default_world(sim, param)

    # CT phantom ?
    if param.gaga_pth == "":
        gaga_et.add_ct_image(sim, param)

    # arf ?
    use_arf = False
    if "garf_pth" in param and param.garf_pth != "":
        use_arf = True

    # spect head (debug mode = very small collimator)
    if use_arf:
        heads = gaga_et.add_spect_arf(sim, param)
    else:
        heads = gaga_et.add_spect_heads(sim, param)

    # physics
    gaga_et.set_default_physics(sim)

    # add the voxelized activity source
    if param.gaga_pth == "":
        source = gaga_et.add_voxelized_source(sim, param.ct, param)
    else:
        source = gaga_et.add_gaga_source(sim, param)

    # AA Angular acceptance
    source.direction.acceptance_angle.volumes = []
    for i in range(param.spect_heads):
        source.direction.acceptance_angle.volumes.append(f"spect_{i}")
    source.direction.acceptance_angle.intersection_flag = param["angular_acceptance"]
    source.direction.acceptance_angle.skip_policy = param["skip_policy"]
    source.skip_policy = param["skip_policy"]

    # add stat actor
    gaga_et.add_stat_actor(sim, param)

    # digitizer actor
    if not use_arf:
        gaga_et.add_digitizer(sim, param, add_fake_channel=True)

    # rotation
    if param.angle is not None:
        n = gaga_et.rotate_one_angle(sim, heads, param)
    else:
        n = gaga_et.rotate_gantry(sim, heads, param)


    # consider n runs
    sec = gate.g4_units("s")
    sim.run_timing_intervals = gate.range_timing(0, 1 * sec, n)

    return sim

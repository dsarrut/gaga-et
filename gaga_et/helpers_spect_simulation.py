#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from helpers_simulation import *
import opengate.contrib.spect_ge_nm670 as gate_spect


def max_length(ct_filename):
    info = gate.read_image_info(ct_filename)
    length = info.size * info.spacing
    max_len = np.max(length)
    return length, max_len


def make_arf_training_dataset_simulation(param, sim=None):
    # create the simulation
    if not sim:
        sim = gate.Simulation()

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
    w, ene = gate.get_rad_energy_spectrum(param.radionuclide)
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

    return sim


def make_training_dataset_simulation(param, sim=None):
    # create the simulation
    if not sim:
        sim = gate.Simulation()

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
    Bq = gate.g4_units("Bq")

    # compute largest dimension
    if param["ct_image"] != "":
        length, max_len = max_length(param.ct_image)
    else:
        max_len = 755.0
    print(f"Maximum CT length: {max_len / mm} mm")

    # CT phantom
    ct = None
    if param["ct_image"] != "":
        ct = add_ct_image(sim, param)

    # world size
    world = sim.world
    world.size = [2 * max_len + 0.5 * m, 2 * max_len + 0.5 * m, 2 * max_len + 0.5 * m]
    world.material = "G4_Galactic"

    # cylinder for output phsp
    sph_surface = sim.add_volume("Sphere", "phase_space_sphere")
    sph_surface.rmin = max_len
    sph_surface.rmax = max_len + 1 * mm
    sph_surface.color = [0, 1, 0, 1]
    sph_surface.material = "G4_AIR"

    # physic list
    p = sim.get_physics_user_info()
    p.physics_list_name = "G4EmStandardPhysics_option4"
    sim.set_cut("world", "all", 1 * mm)

    # create a mask source
    if "source_image" not in param or param["source_image"] == "":
        mask = create_mask_uniform_source(param.ct_image, param.min_z, param.max_z)
        mask_filename = str(param.output_folder / "source_uniform.mhd")
        itk.imwrite(mask, mask_filename)
    else:
        mask_filename = param["source_image"]
    print(mask_filename)

    # to compute activity concentration, need the volume
    m = itk.imread(mask_filename)
    info = gate.get_info_from_image(m)
    m = itk.array_view_from_image(m)
    vol = m[m == 1].sum() * np.prod(info.spacing) * 0.001
    print(f"Volume of the activity map: {vol} cc")
    print(f"Total activity            : {param.activity_bq} Bq")
    print(f"Activity concentration    : {param.activity_bq / vol} Bq/cc")

    # add the uniform voxelized source
    source = sim.add_source("Voxels", "source")
    if param["ct_image"] != "":
        source.mother = ct.name
    source.particle = "gamma"
    source.activity = param.activity_bq * Bq / ui.number_of_threads
    source.image = mask_filename
    source.direction.type = "iso"
    source.energy.type = "spectrum"
    gate.set_source_rad_energy_spectrum(source, param.radionuclide)

    # add stat actor
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.track_types_flag = True
    stats.output = param.output_folder / "training_dataset_stats.txt"

    # filter gamma only
    f = sim.add_filter("ParticleFilter", "f")
    f.particle = "gamma"

    # phsp
    phsp = sim.add_actor("PhaseSpaceActor", "phase_space")
    phsp.mother = "phase_space_sphere"
    phsp.attributes = [
        "KineticEnergy",
        "PrePosition",  # why pre and not post ? because first time in vol
        "PreDirection",
        "TimeFromBeginOfEvent",  # not needed ; only to test with ideal reconstruction
        # needed for gan_flag
        "EventID",
        "EventKineticEnergy",
        # for conditional :
        "EventPosition",
        "EventDirection",  # is it normalized ?
    ]
    phsp.output = param.output_folder / "training_dataset.root"
    # this option allow to store all events even if absorbed
    phsp.store_absorbed_event = True
    # filter to keep only the gammas
    phsp.filters.append(f)

    return sim


def make_spect_simulation(param, sim=None):
    # create the simulation
    if not sim:
        sim = gate.Simulation()

    # main options
    ui = sim.user_info
    ui.g4_verbose = False
    ui.visu = param.visu
    ui.number_of_threads = param.number_of_threads
    ui.verbose_level = gate.INFO
    if ui.visu:
        ui.number_of_threads = 1
        param.activity_bq = 1000
    print(ui)

    # convert str to pathlib
    param.output_folder = pathlib.Path(param.output_folder)

    # units
    m = gate.g4_units("m")
    cm = gate.g4_units("cm")
    mm = gate.g4_units("mm")
    sec = gate.g4_units("second")

    # compute largest dimension
    if param["ct_image"] != "":
        length, max_len = max_length(param.ct_image)
    else:
        max_len = 755.0

    # CT phantom (only if gaga is not used)
    ct = None
    if param.gaga_pth == "" or ui.visu:
        if param.ct_image != "":
            ct = add_ct_image(sim, param)

    # world size
    world = sim.world
    s = 2 * max_len + 0.5 * m + param.spect_radius_mm * mm
    print(f"world size {s / cm} cm")
    world.size = [s, s, s]
    world.material = "G4_Galactic"

    # arf ?
    use_arf = False
    if "garf_pth" in param and param.garf_pth != "":
        use_arf = True

        # spect head (debug mode = very small collimator)
    if use_arf:
        heads = add_spect_arf(sim, param)
    else:
        heads = add_spect_heads(sim, param)

    # physic list
    p = sim.get_physics_user_info()
    p.physics_list_name = "G4EmStandardPhysics_option4"
    sim.set_cut("world", "all", 1 * mm)

    # add the voxelized activity source
    if param.gaga_pth == "":
        source = add_voxelized_source(sim, ct, param)
    else:
        source = add_gaga_source(sim, param)
    print("Source translation", source.position.translation)

    # add stat actor
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.track_types_flag = True
    stats.output = param.output_folder / "ref_stats.txt"

    # digitizer actor
    if not use_arf:
        add_digitizer(sim, param, add_fake_channel=True)

    # rotation
    if param.angle is not None:
        n = rotate_one_angle(sim, heads, param)
    else:
        n = rotate_gantry(sim, heads, param)

    # consider n runs
    sim.run_timing_intervals = gate.range_timing(0, 1 * sec, n)

    return sim


def add_spect_heads(sim, param):
    mm = gate.g4_units("mm")
    # spect head (debug mode = very small collimator)
    nb_heads = param.spect_heads
    if nb_heads > 4:
        gate.fatal(f"Cannot have more than 4 heads, sorry")
    colli_type = gate_spect.get_collimator(param.radionuclide)
    itr, irot = gate_spect.get_orientation_for_CT(
        colli_type, param.spect_table_shift_mm * mm, param.spect_radius_mm * mm
    )
    tr, rot = gate.volume_orbiting_transform("z", 0, 360, nb_heads, itr, irot)
    heads = []
    for i in range(nb_heads):
        spect = gate_spect.add_ge_nm67_spect_head(
            sim, f"spect_{i}", colli_type, debug=sim.user_info.visu
        )
        spect.translation = tr[i]
        spect.rotation = gate.rot_g4_as_np(rot[i])
        heads.append(spect)
    return heads


def add_spect_arf(sim, param):
    cm = gate.g4_units("cm")
    nm = gate.g4_units("nm")
    mm = gate.g4_units("mm")

    # head position, rotation
    nb_heads = param.spect_heads
    colli_type = gate_spect.get_collimator(param.radionuclide)
    pos, crystal_distance, _ = gate_spect.get_plane_position_and_distance_to_crystal(
        colli_type
    )
    itr, irot = gate_spect.get_orientation_for_CT(
        colli_type, param.spect_table_shift_mm * mm, param.spect_radius_mm * mm
    )
    tr, rot = gate.volume_orbiting_transform("z", 0, 360, nb_heads, itr, irot)

    print("Plane distance according to head center", pos)

    heads = []
    for i in range(param.spect_heads):
        # fake spect head
        head = gate_spect.add_ge_nm67_fake_spect_head(sim, f"spect_{i}")
        head.translation = tr[i]
        print("head translation", i, head.translation)
        head.rotation = gate.rot_g4_as_np(rot[i])
        heads.append(head)

        # detector plane
        detector_plane = sim.add_volume("Box", f"detPlane_{i}")
        detector_plane.mother = head.name
        detector_plane.size = [57.6 * cm, 44.6 * cm, 1 * nm]
        detector_plane.translation = [0, 0, pos]
        detector_plane.material = "G4_Galactic"
        detector_plane.color = [1, 0, 0, 1]

        # arf actor
        arf = sim.add_actor("ARFActor", f"arf_{i}")
        arf.mother = detector_plane.name
        fn = f'{param.output_folder / "ref"}_projection_{i}.mhd'
        arf.output = fn
        arf.batch_size = param.garf_batch_size
        arf.image_size = [128, 128]
        arf.image_spacing = [4.41806 * mm, 4.41806 * mm]
        arf.verbose_batch = True
        arf.distance_to_crystal = crystal_distance
        arf.pth_filename = param.garf_pth

    return heads


def add_voxelized_source(sim, ct, param):
    Bq = gate.g4_units("Bq")
    source = sim.add_source("Voxels", "source")
    if param["ct_image"] != "":
        source.mother = ct.name
        source.position.translation = gate.get_translation_between_images_center(
            param.ct_image, param.activity_source
        )
    source.particle = "gamma"
    source.activity = param.activity_bq * Bq / sim.user_info.number_of_threads
    source.image = param.activity_source
    source.direction.type = "iso"
    source.energy.type = "spectrum"
    gate.set_source_rad_energy_spectrum(source, param.radionuclide)
    return source


def add_gaga_source(sim, param):
    Bq = gate.g4_units("Bq")
    keV = gate.g4_units("keV")
    mm = gate.g4_units("mm")
    gsource = sim.add_source("GAN", "gaga")
    gsource.particle = "gamma"
    w, en = gate.get_rad_energy_spectrum(param.radionuclide)
    rad_yield = np.sum(w)
    print(f'Rad "{param.radionuclide}" yield is {rad_yield}')
    gsource.activity = (
            param.activity_bq * Bq / sim.user_info.number_of_threads * rad_yield
    )
    gsource.pth_filename = param.gaga_pth
    gsource.position_keys = ["PrePosition_X", "PrePosition_Y", "PrePosition_Z"]
    gsource.backward_distance = param.gaga_backward_distance_mm * mm
    gsource.direction_keys = ["PreDirection_X", "PreDirection_Y", "PreDirection_Z"]
    gsource.energy_key = "KineticEnergy"
    gsource.energy_threshold = param.gaga_energy_threshold_keV * keV
    gsource.weight_key = None
    gsource.time_key = "TimeFromBeginOfEvent"
    gsource.time_relative = True
    gsource.batch_size = param.gaga_batch_size
    gsource.verbose_generator = True

    # set the generator and the condition generator
    voxelized_cond_generator = gate.VoxelizedSourceConditionGenerator(
        param.activity_source
    )
    if "gaga_cond_direction" not in param:
        param.gaga_cond_direction = True
    voxelized_cond_generator.compute_directions = param.gaga_cond_direction

    # FIXME here change the move_backward if need to un-parameterize
    """
    2 functions to change : get_output_keys_with_lock and move_backward 
    
    warning in get_output_keys_with_lock
    need g.params.keys_output
    """

    gen = gate.GANSourceConditionalGenerator(
        gsource,
        voxelized_cond_generator.generate_condition,
    )

    # FIXME test
    if "ideal" in param.gaga_pth:
        print("ideal")
        mb = gen.move_backward
        gen.keys_output = [
            "KineticEnergy",
            "PrePosition_X",
            "PrePosition_Y",
            "PrePosition_Z",
            "PreDirection_X",
            "PreDirection_Y",
            "PreDirection_Z",
            "TimeFromBeginOfEvent",
            "EventPosition_X",
            "EventPosition_Y",
            "EventPosition_Z",
            "EventDirection_X",
            "EventDirection_Y",
            "EventDirection_Z",
        ]

        import gaga_phsp as gaga

        def reparametrization(g, fake):
            print("reparametrization")
            params = {"keys_list": g.params.keys_list}
            print(params)
            fake, keys_out = gaga.from_ideal_pos_to_exit_pos(fake, params)
            print("k out", keys_out)
            mb(g, fake)

        gen.move_backward = reparametrization

    gsource.generator = gen
    return gsource


def add_digitizer(sim, param, add_fake_channel=False):
    mm = gate.g4_units("mm")
    keV = gate.g4_units("keV")
    for i in range(param.spect_heads):
        fn = f'{param.output_folder / "ref"}_projection_{i}.mhd'
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

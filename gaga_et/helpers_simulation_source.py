#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import itk
import numpy as np
from pathlib import Path


def create_uniform_activity_source(param):
    mask = create_mask_uniform_source(param.ct_image, param.min_z, param.max_z, param.threshold)
    mask_filename = str(Path(param.output_folder) / "source_uniform.mhd")
    itk.imwrite(mask, mask_filename)
    param["source_image"] = mask_filename


def print_activity(param):
    # to compute activity concentration, we need the volume
    m = itk.imread(param.source_image)
    info = gate.get_info_from_image(m)
    m = itk.array_view_from_image(m)
    vol = m[m == 1].sum() * np.prod(info.spacing) * 0.001
    print(f"Volume of the activity map: {vol} cc")
    print(f"Total activity            : {param.activity_bq} Bq")
    print(f"Activity concentration    : {param.activity_bq / vol} Bq/cc")


def create_mask_uniform_source(ct_filename, min_z=None, max_z=None, threshold=-1000):
    # read ct
    img = itk.imread(ct_filename)
    imga = itk.array_view_from_image(img)
    info = gate.get_info_from_image(img)

    # create a same size empty (mask) image
    mask = gate.create_3d_image(info.size, info.spacing, pixel_type="unsigned char")
    mask.SetOrigin(info.origin)
    mask.SetDirection(info.dir)
    maska = itk.array_view_from_image(mask)

    # threshold
    maska[imga > threshold] = 1

    # min max slices
    if min_z:
        mins = int((min_z - info.origin[2]) / info.spacing[2])
        maska[0:mins, :, :] = 0
    # min max slices
    if max_z:
        maxs = int((max_z - info.origin[2]) / info.spacing[2])
        maska[maxs:, :, :] = 0

    return mask


def add_positron_vox_activity_source(sim, param):
    Bq = gate.g4_units('Bq')
    ui = sim.user_info
    # add the uniform voxelized source
    source = sim.add_source("Voxels", "source")
    source.image = param.source_image
    source.particle = "e+"
    source.energy.type = param.radionuclide
    source.activity = param.activity_bq * Bq / ui.number_of_threads
    source.direction.type = "iso"
    if param.visu is False:
        source.mother = param.ct.name
        source.position.translation = gate.get_translation_between_images_center(param.ct.image, source.image)
        print("Source translation: ", source.position.translation)


def add_gamma_vox_activity_source(sim, param):
    Bq = gate.g4_units('Bq')
    ui = sim.user_info
    # add the uniform voxelized source
    source = sim.add_source("Voxels", "source")
    source.image = param.source_image
    source.particle = "gamma"
    source.energy.type = param.radionuclide
    source.activity = param.activity_bq * Bq / ui.number_of_threads
    source.direction.type = "iso"
    source.energy.type = "spectrum"
    gate.set_source_rad_energy_spectrum(source, param.radionuclide)
    if param.visu is False:
        source.mother = param.ct.name
        source.position.translation = gate.get_translation_between_images_center(param.ct.image, source.image)
        print("Source translation: ", source.position.translation)


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
    gsource = sim.add_source("GAN", "source")
    gsource.particle = "gamma"
    w, en = gate.get_rad_gamma_energy_spectrum(param.radionuclide)
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

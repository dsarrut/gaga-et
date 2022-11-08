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

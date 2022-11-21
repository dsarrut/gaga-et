#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import gaga_et
import numpy as np
from pathlib import Path

def max_length(ct_filename):
    info = gate.read_image_info(ct_filename)
    length = info.size * info.spacing
    max_len = np.max(length)
    return length, max_len


def set_default_world(sim, param, length=None):
    mm = gate.g4_units('mm')
    m = gate.g4_units('m')
    if length is None:
        if param["ct_image"] != "":
            length, max_len = gaga_et.max_length(param.ct_image)
        else:
            max_len = 755.0
    else:
        max_len = length
    print(f"Maximum CT length: {max_len / mm} mm")
    param.max_len = max_len
    # set the world size
    world = sim.world
    world.size = [2 * max_len + 0.5 * m, 2 * max_len + 0.5 * m, 2 * max_len + 0.5 * m]
    world.material = "G4_AIR"
    print(f'World size : {world.size}')
    return world


def add_sphere_phsp(sim, param):
    mm = gate.g4_units('mm')
    sphere = sim.add_volume('Sphere', 'spherical_phase_space')
    sphere.rmin = param.max_len
    sphere.rmax = param.max_len + 1 * mm
    sphere.color = [0, 1, 0, 1]
    sphere.material = 'G4_AIR'
    return sphere


def add_ct_image(sim, param):
    ui = sim.user_info
    gcm3 = gate.g4_units("g/cm3")
    print(ui)
    if ui.visu:
        info = gate.read_image_info(param.ct_image)
        length = info.size * info.spacing
        ct = sim.add_volume("Box", "ct")
        ct.size = length
        ct.material = "G4_AIR"
        ct.color = [0, 0, 1, 1]
    else:
        ct = sim.add_volume("Image", "ct")
        ct.image = param.ct_image
        ct.material = "G4_AIR"  # material used by default
        tol = param.density_tolerance_gcm3 * gcm3
        ct.voxel_materials, materials = gate.HounsfieldUnit_to_material(
            tol, param.table_mat, param.table_density
        )
        if param.verbose:
            print(f"Density tolerance = {gate.g4_best_unit(tol, 'Volumic Mass')}")
            print(f"Nb of materials in the CT : {len(ct.voxel_materials)} materials")
            print(f"Materials: {ct.voxel_materials}")
        ct.dump_label_image = Path(param.output_folder) / "labels.mhd"
    param.ct = ct
    return ct

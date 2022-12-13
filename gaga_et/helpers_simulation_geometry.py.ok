#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import gaga_et
import numpy as np
from pathlib import Path
import opengate.contrib.spect_ge_nm670 as gate_spect


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
        # specific material for the Ti insert
        gate.new_material_weights(f"Ti_insert", 9000 * gcm3, "Pb")
        ct.voxel_materials.append([1000, 200000, 'Ti_insert'])
        if param.verbose:
            print(f"Density tolerance = {gate.g4_best_unit(tol, 'Volumic Mass')}")
            print(f"Nb of materials in the CT : {len(ct.voxel_materials)} materials")
            print(f"Materials: {ct.voxel_materials}")
        ct.dump_label_image = Path(param.output_folder) / "labels.mhd"
    param.ct = ct
    return ct


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
        fn = f'{Path(param.output_folder) / "ref"}_projection_{i}.mhd'
        arf.output = fn
        arf.batch_size = param.garf_batch_size
        arf.image_size = [128, 128]
        arf.image_spacing = [4.41806 * mm, 4.41806 * mm]
        arf.verbose_batch = True
        arf.distance_to_crystal = crystal_distance
        arf.pth_filename = param.garf_pth

    return heads

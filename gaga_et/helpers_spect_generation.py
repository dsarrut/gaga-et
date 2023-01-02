#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gaga_phsp as gaga
import garf
import numpy as np
import opengate as gate
from box import Box
import itk
import time


def print_timing(t1, n, s):
    t = time.time() - t1
    pps = float(n) / t
    print(s, f"{t:.3f} sec ; PPS = {pps:.0f}")


def spect_generate_images(param):
    """
    input param:
    - gaga_pth
    - garf_pth
    - n (number of samples per angles)
    - batch_size
    - radius
    """

    # check param FIXME
    param.n = int(param.n)
    param.batch_size = int(param.batch_size)

    # load gaga pth and param
    param_gaga = Box(param)
    print(f'Loading GAN {param.gaga_pth}')
    param_gaga.gan_params, param_gaga.G, param_gaga.D, optim, dtypef = gaga.load(param.gaga_pth)

    # load garf pth
    param_garf = Box(param)
    print(f'Loading GARF {param.garf_pth}')
    param_garf.nn, param_garf.model = garf.load_nn(param.garf_pth, verbose=False)
    param_garf.gpu_batch_size = param.batch_size
    param_garf.size = np.array(param.image_size)
    param_garf.spacing = np.array(param.image_spacing)
    param_garf.length = param.arf_length_mm
    param_garf.N_dataset = param_gaga.batch_size
    param_garf.N_scale = 1.0
    param_garf.image_plane_size_mm = np.array([size * spacing for size, spacing in
                                               zip(param.image_size, param.image_spacing)])
    param_garf.nb_energy_windows = param_garf.nn['model_data']['n_ene_win']
    print(f'Number of energy windows : {param_garf.nb_energy_windows}')

    # activity source condition
    print(f'Loading cond generator (voxelized activity source) {param.activity_source}')
    cond_generator = gate.VoxelizedSourceConditionGenerator(param.activity_source)
    cond_generator.compute_directions = True
    param_gaga.cond_generator = cond_generator

    # generate the projection
    angles = [angle for angle in np.linspace(start=0, stop=360, num=param.spect_angles, endpoint=False)]
    '''
    output_image = None
    i = 0
    for angle in angles:
        image = spect_generate_one_projection(angle, param_gaga, param_garf)
        if output_image is None:
            s = list(image.shape)
            s[0] *= len(angles)
            output_image = np.zeros(s)
        nw = param_garf.nb_energy_windows
        output_image[nw * i:nw * (i + 1)] = image
        i += 1
    '''
    image = spect_generate_projections(angles, param_gaga, param_garf)

    # save or append output images
    spacing = [param.image_spacing[0], param.image_spacing[1], 1]
    img = itk.GetImageFromArray(output_image)
    img.SetSpacing(spacing)
    origin = -param_garf.image_plane_size_mm / 2.0 + param_garf.spacing / 2.0
    origin = np.array([origin[0], origin[1], 0])
    img.SetOrigin(origin)

    return img


def spect_generate_one_projection(angle, param_gaga, param_garf):
    # loop on batch
    n = param_gaga.n
    total_n = int(0)
    b = param_gaga.batch_size
    # init image
    output_img_size = [param_garf.nb_energy_windows, param_garf.image_size[0], param_garf.image_size[1]]
    image = np.zeros(output_img_size, dtype=np.float64)
    # init arf
    param_garf.current_px = None
    t1 = time.time()
    is_last_batch = total_n + b >= param_gaga.n
    while total_n < n:
        print(f'Generate batch={b}  --- {total_n}/{n}')
        spect_generate_one_batch(b, angle, param_gaga, param_garf, image, is_last_batch)
        total_n += b
        # image = image + p
        if total_n + b >= param_gaga.n:
            b = param_gaga.n - total_n
            is_last_batch = True

    print_timing(t1, n, f"TOTAL Timing for angle {angle}")
    print()
    return image


def spect_generate_one_batch(n, angle, param_gaga, param_garf, image, is_last_batch=False):
    # load or generate angles/planes
    # FIXME several planes
    param_planes = gaga.init_plane2(n,
                                    angle=angle,
                                    radius=param_gaga.radius,
                                    spect_table_shift_mm=param_gaga.spect_table_shift_mm)
    param_planes = Box(param_planes)

    # sample conditions cond = 6D position ; direction
    cond = param_gaga.cond_generator.generate_condition(n)

    # sample GAN
    t1 = time.time()
    x = gaga.generate_samples3(param_gaga.gan_params, param_gaga.G, n, cond)
    print_timing(t1, n, f"Timing GAN generation {x.shape}")

    # print keys
    # kl = param_gaga.gan_params.keys_list
    # print(kl)

    # move backward
    position = x[:, 1:4]  # FIXME indices of the position
    direction = x[:, 4:7]  # FIXME indices of the position
    back = param_gaga.gaga_backward_distance_mm
    x[:, 1:4] = position - back * direction

    # project on plane
    px = gaga.project_on_plane2(x, param_planes, param_garf.image_plane_size_mm)

    # is_last_batch = True
    if param_garf.current_px is None:
        param_garf.current_px = px
    else:
        param_garf.current_px = np.concatenate((param_garf.current_px, px), axis=0)

    # apply garf
    if len(param_garf.current_px) > n or is_last_batch:
        t1 = time.time()
        garf.build_arf_image_with_nn2(
            param_garf.nn,
            param_garf.model,
            param_garf.current_px,
            param_garf,
            image,
            verbose=False
        )
        # img = spect_apply_garf(px, param_garf, image)
        param_garf.current_px = None
        print_timing(t1, n, f"Timing ARF {px.shape}")


def spect_generate_projections(angles, param_gaga, param_garf):
    # loop on batch
    n = param_gaga.n
    total_n = int(0)
    b = param_gaga.batch_size
    # init image
    output_img_size = [param_garf.nb_energy_windows, param_garf.image_size[0], param_garf.image_size[1]]
    image = np.zeros(output_img_size, dtype=np.float64)
    # init arf
    param_garf.current_px = None
    t1 = time.time()
    is_last_batch = total_n + b >= param_gaga.n
    while total_n < n:
        print(f'Generate batch={b}  --- {total_n}/{n}')
        spect_generate_batch(b, angles, param_gaga, param_garf, image, is_last_batch)
        total_n += b
        # image = image + p
        if total_n + b >= param_gaga.n:
            b = param_gaga.n - total_n
            is_last_batch = True

    print_timing(t1, n, f"TOTAL Timing ")
    print()
    return image


def spect_generate_batch(n, angles, param_gaga, param_garf, image, is_last_batch=False):
    # sample conditions cond = 6D position ; direction
    cond = param_gaga.cond_generator.generate_condition(n)

    # sample GAN
    t1 = time.time()
    x = gaga.generate_samples3(param_gaga.gan_params, param_gaga.G, n, cond)
    print_timing(t1, n, f"Timing GAN generation {x.shape}")

    # print keys
    # kl = param_gaga.gan_params.keys_list
    # print(kl)

    i = 0
FIXME     output_image = None # FIXME NOT  must be initialized before
    for angle in angles:
        image = spect_generate_batch_one_angle(x, angle, param_gaga, param_garf)
        if output_image is None:
            s = list(image.shape)
            s[0] *= len(angles)
            output_image = np.zeros(s)
        nw = param_garf.nb_energy_windows
        output_image[nw * i:nw * (i + 1)] = image
        i += 1


def spect_generate_batch_one_angle(x, angle, param_gaga, param_garf):
    # load or generate angles/planes
    # FIXME several planes
    param_planes = gaga.init_plane2(n,
                                    angle=angle,
                                    radius=param_gaga.radius,
                                    spect_table_shift_mm=param_gaga.spect_table_shift_mm)
    param_planes = Box(param_planes)



    # move backward


position = x[:, 1:4]  # FIXME indices of the position
direction = x[:, 4:7]  # FIXME indices of the position
back = param_gaga.gaga_backward_distance_mm
x[:, 1:4] = position - back * direction

# project on plane
px = gaga.project_on_plane2(x, param_planes, param_garf.image_plane_size_mm)

# is_last_batch = True
if param_garf.current_px is None:
    param_garf.current_px = px
else:
    param_garf.current_px = np.concatenate((param_garf.current_px, px), axis=0)

# apply garf
if len(param_garf.current_px) > n or is_last_batch:
    t1 = time.time()
    garf.build_arf_image_with_nn2(
        param_garf.nn,
        param_garf.model,
        param_garf.current_px,
        param_garf,
        image,
        verbose=False
    )
    # img = spect_apply_garf(px, param_garf, image)
    param_garf.current_px = None
    print_timing(t1, n, f"Timing ARF {px.shape}")

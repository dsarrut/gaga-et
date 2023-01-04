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
    v = param.verb

    # load gaga pth and param
    param_gaga = Box(param)
    v.print(5, f'Loading GAN {param.gaga_pth}')
    param_gaga.gan_params, param_gaga.G, param_gaga.D, optim, dtypef = gaga.load(param.gaga_pth)

    # load garf pth
    param_garf = Box(param)
    v.print(5, f'Loading GARF {param.garf_pth}')
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
    v.print(5, f'Number of energy windows : {param_garf.nb_energy_windows}')

    # activity source condition
    v.print(5, f'Loading cond generator (voxelized activity source) {param.activity_source}')
    cond_generator = gate.VoxelizedSourceConditionGenerator(param.activity_source)
    cond_generator.compute_directions = True
    param_gaga.cond_generator = cond_generator

    # generate the angles
    angles = [angle for angle in np.linspace(start=0, stop=360, num=param.spect_angles, endpoint=False)]

    # generate the projections
    param_gaga.verb = v
    output_image = spect_generate_projections(angles, param_gaga, param_garf)

    # save or append output images
    spacing = [param.image_spacing[0], param.image_spacing[1], 1]
    img = itk.GetImageFromArray(output_image)
    img.SetSpacing(spacing)
    origin = -param_garf.image_plane_size_mm / 2.0 + param_garf.spacing / 2.0
    origin = np.array([origin[0], origin[1], 0])
    img.SetOrigin(origin)

    return img


def spect_generate_one_projection_OLD(angle, param_gaga, param_garf):
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
        spect_generate_one_batch_OLD(b, angle, param_gaga, param_garf, image, is_last_batch)
        total_n += b
        # image = image + p
        if total_n + b >= param_gaga.n:
            b = param_gaga.n - total_n
            is_last_batch = True

    print_timing(t1, n, f"TOTAL Timing for angle {angle}")
    print()
    return image


def spect_generate_one_batch_OLD(n, angle, param_gaga, param_garf, image, is_last_batch=False):
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
    v = param_gaga.verb
    # init image
    output_img_size = [param_garf.nb_energy_windows * len(angles), param_garf.image_size[0], param_garf.image_size[1]]
    output_image = np.zeros(output_img_size, dtype=np.float64)
    v.print(5, f'Allocate output images: {output_image.shape} = angles {len(angles)} '
               f'x energy windows {param_garf.nb_energy_windows}')
    # init arf
    param_garf.current_px = None
    v.t(5)
    is_last_batch = total_n + b >= param_gaga.n
    while total_n < n:
        v.print(8, f'Generate batch={b}  --- {total_n}/{n}')
        spect_generate_batch(b, angles, param_gaga, param_garf, output_image, is_last_batch)
        total_n += b
        # image = image + p
        if total_n + b >= param_gaga.n:
            b = param_gaga.n - total_n
            is_last_batch = True

    v.print_timing(5, n, f"TOTAL Timing ")
    return output_image


def spect_generate_batch(n, angles, param_gaga, param_garf, output_image, is_last_batch):
    v = param_gaga.verb
    # sample conditions cond = 6D position ; direction
    cond = param_gaga.cond_generator.generate_condition(n)

    # sample GAN
    v.t(10)
    x = gaga.generate_samples3(param_gaga.gan_params, param_gaga.G, n, cond)
    v.print_gaga_timing(10, n, f"Timing GAN generation {x.shape}")

    # move backward
    position = x[:, 1:4]  # FIXME indices of the position
    direction = x[:, 4:7]  # FIXME indices of the position
    back = param_gaga.gaga_backward_distance_mm
    x[:, 1:4] = position - back * direction

    i = 0
    nw = param_garf.nb_energy_windows
    for angle in angles:
        image = output_image[nw * i:nw * (i + 1)]
        spect_generate_batch_one_angle(x, n, angle, param_gaga, param_garf, image, is_last_batch)
        i += 1


def spect_generate_batch_one_angle(x, n, angle, param_gaga, param_garf, image, is_last_batch):
    v = param_gaga.verb
    # load or generate angles/planes
    param_planes = gaga.init_plane2(n,
                                    angle=angle,
                                    radius=param_gaga.radius,
                                    spect_table_shift_mm=param_gaga.spect_table_shift_mm)
    param_planes = Box(param_planes)

    # project on plane
    px = gaga.project_on_plane2(x, param_planes, param_garf.image_plane_size_mm)

    # store px per angle until the ARF batch is full
    if param_garf.current_px is None:
        param_garf.current_px = {}

    if angle not in param_garf.current_px or param_garf.current_px[angle] is None:
        param_garf.current_px[angle] = px
    else:
        param_garf.current_px[angle] = np.concatenate((param_garf.current_px[angle], px), axis=0)
    v.print(15, f'Current px for angle {angle} : {param_garf.current_px[angle].shape}')

    # apply garf
    if len(param_garf.current_px[angle]) > n or is_last_batch:
        v.print(15, f'last batch, create image, {param_garf.current_px[angle].shape}')
        v.t(10)
        garf.build_arf_image_with_nn2(
            param_garf.nn,
            param_garf.model,
            param_garf.current_px[angle],
            param_garf,
            image,
            verbose=False
        )
        # img = spect_apply_garf(px, param_garf, image)
        param_garf.current_px[angle] = None
        v.print_garf_timing(10, n, f"Timing ARF {px.shape}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gaga_et
import gaga_phsp as gaga
import garf
import numpy as np
import opengate as gate
import pathlib
import opengate.contrib.spect_ge_nm670 as gate_spect
from box import Box
import itk
import time


def spect_generate_images(param):
    '''
    input param:
    - gaga_pth
    - garf_pth
    - n (number of samples per angles)
    - batch_size
    - radius
    '''

    # check param FIXME
    param.n = int(param.n)
    param.batch_size = int(param.batch_size)

    # load gaga pth and param
    param_gaga = Box(param)
    print(f'Loading GAN {param.gaga_pth}')
    param_gaga.gan_params, param_gaga.G, param_gaga.D, optim, dtypef = gaga.load(param.gaga_pth)
    # print(param_gaga)

    # load garf pth
    param_garf = Box(param)
    print(f'Loading GARF {param.garf_pth}')
    param_garf.nn, param_garf.model = garf.load_nn(param.garf_pth, verbose=False)
    param_garf.gpu_batch_size = param.batch_size
    param_garf.size = param.image_size
    param_garf.spacing = param.image_spacing
    param_garf.length = param.arf_length_mm
    param_garf.N_dataset = param_gaga.batch_size
    param_garf.N_scale = 1.0

    # activity source condition
    print(f'Loading cond generator (voxelized activity source) {param.activity_source}')
    cond_generator = gate.VoxelizedSourceConditionGenerator(param.activity_source)
    cond_generator.compute_directions = True
    param_gaga.cond_generator = cond_generator

    # loop on angles/planes <---- LATER

    # load or generate angles/planes
    # FIXME several planes
    param_planes = gaga.init_plane2(param.batch_size, angle=0, radius=param.radius)
    param_planes = Box(param_planes)
    param_planes.image_size = param.image_size
    param_planes.image_plane_size_mm = [size * spacing for size, spacing in
                                        zip(param.image_size, param.image_spacing)]  # FIXME
    print(param_planes)

    # generate the projection
    image = spect_generate_one_projection(param_gaga, param_garf, param_planes)
    print(f'image: {image.shape}')

    # save or append output images
    spacing = [param.image_spacing[0], param.image_spacing[1], 1]
    img = itk.GetImageFromArray(image)
    img.SetSpacing(spacing)
    origin = -np.array(param_planes.image_plane_size_mm) / 2.0 + np.array(param.image_spacing) / 2.0
    origin = np.array([origin[0], origin[1], 0])
    img.SetOrigin(origin)

    return img


def spect_generate_one_projection(param_gaga, param_garf, param_plane):
    # loop on batch
    n = param_gaga.n
    total_n = int(0)
    b = param_gaga.batch_size
    image = np.zeros(param_plane.image_size, dtype=np.float64)
    # print('image shape', image.shape)
    t1 = time.time()
    while total_n < n:
        print(f'Generate batch={b}  --- {total_n}/{n}')
        p = spect_generate_one_batch(b, param_gaga, param_garf, param_plane)
        total_n += b
        image = image + p
        if total_n + b > param_gaga.n:
            b = param_gaga.n - total_n

    # print(f'end one projection {total_n} {image.shape}')
    t = time.time() - t1
    pps = float(n) / t
    print(f"Timing: {t:.3f} sec ; PPS = {pps:.0f}")
    return image


def spect_generate_one_batch(n, param_gaga, param_garf, param_plane):
    # sample conditions cond = 6D position ; direction
    cond = param_gaga.cond_generator.generate_condition(n)
    print(f'Generated conditions: {cond.shape}')

    # sample GAN
    x = spect_gaga_generate_samples(cond, n, param_gaga)
    kl = param_gaga.gan_params.keys_list
    print(kl)

    # FIXME move backward
    position = x[:, 1:4]  # FIXME indices of the position
    direction = x[:, 4:7]  # FIXME indices of the position
    back = 700
    x[:, 1:4] = position - back * direction

    # project on plane
    px = spect_project_on_plane(x, param_plane)
    print('px shape', px.shape)

    # apply garf
    # todo add flog to avoid computing squared image
    img = spect_apply_garf(px, param_garf)
    print('img shape', img.shape)

    return img


def spect_apply_garf(px, param_garf):
    img = garf.build_arf_image_with_nn2(
        param_garf.nn,
        param_garf.model,
        px,
        param_garf,
        verbose=True,
        debug=False
    )
    return img


def spect_project_on_plane(x, param):
    px = gaga.project_on_plane2(
        x, param,
        image_plane_size_mm=param.image_plane_size_mm,
        debug=False
    )
    return px


def spect_gaga_generate_samples(cond, n, param):
    # FIXME timing ?
    x = gaga.generate_samples2(
        param.gan_params,
        param.G,
        param.D,
        n=n,  # total n
        batch_size=n,  # batch size
        normalize=False,
        to_numpy=True,  # FIXME unsure
        cond=cond,  # FIXME todo
        silence=False
    )

    return x

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import os
import torch
import gaga_phsp as gaga
import gatetools
from gatetools import phsp
import itk
import opengate as gate
from torch.autograd import Variable
import scipy


def phsp_get_AB(phsp, keys):
    if 'X1' in keys:
        k1 = keys.index('X1')
        k2 = keys.index('Z1') + 1
        A = phsp[:, k1:k2]
        k1 = keys.index('X2')
        k2 = keys.index('Z2') + 1
        B = phsp[:, k1:k2]
    else:
        k1 = keys.index('Ax')
        k2 = keys.index('Az') + 1
        A = phsp[:, k1:k2]
        k1 = keys.index('Bx')
        k2 = keys.index('Bz') + 1
        B = phsp[:, k1:k2]
    return A, B


def read_pair_phsp_or_pth_generate(param):
    b, extension = os.path.splitext(param.file_input)
    # if pth, generate samples
    if extension == '.pth':
        if param.verbose:
            print(f'Reading gan parameters {param.file_input}')
        params, G, D, optim, dtypef = gaga.load(param.file_input, 'auto', False, param.epoch,
                                                fatal_on_unknown_keys=False)
        if param.cond_phsp is not None:
            if param.verbose:
                print(f'Reading conditional phsp {param.cond_phsp}')
            cond_keys = params['cond_keys']
            cond_data, cond_read_keys, m = gatetools.phsp.load(param.cond_phsp, nmax=param.n)
            cond_data = gatetools.phsp.select_keys(cond_data, cond_read_keys, cond_keys)
            if param.verbose:
                print(f'Conditional keys {cond_keys} {cond_data.shape}')
        else:
            cond_data = None

        if param.verbose:
            print('Generate samples', param.file_input, param.n)
        phsp = gaga.generate_samples2(params, G, D,
                                      param.n, param.batch_size,
                                      normalize=False,
                                      to_numpy=param.to_numpy,
                                      cond=cond_data,
                                      silence=not param.verbose)
        keys = params['keys_list']
        if param.verbose:
            print(keys)
            print(phsp.shape)
    else:
        # read the phsp
        phsp, keys, m = gatetools.phsp.load_npy(param.file_input, nmax=param.n, shuffle=False)
        if param.verbose:
            print(f'Load {param.file_input} {param.n}/{m}')
        if not param.to_numpy:
            # this trigger a warning (non writable numpy array) # FIXME
            dtypef, device = gaga.init_pytorch_cuda('auto', verbose=param.verbose)
            phsp = Variable(torch.FloatTensor(phsp)).type(dtypef)

    return phsp, keys


def ideal_pet_reconstruction3(phsp, keys, param):
    """
    Expected param
    - tlor_radius
    - move_forward
    - emin emax
    - img_size
    - img_spacing
    - verbose
    """

    # input is either 1) pairs or 2) lor to convert to pairs
    if 'Cx' in keys:
        # phsp, keys = from_phsp_lor_to_pairs(phsp, keys, radius1, method, ignore_direction)
        if param.tlor_radius < 0:
            gate.fatal(f'Error, tlor_radius option is needed to convert from tlor to pairs')
        params = {
            'keys_list': keys,
            'radius': param.tlor_radius,
            'ignore_directions': param.move_forward is None
        }
        if param.verbose:
            print(f'Convert tlor to pair parametrisation', keys, param.tlor_radius)
        phsp = gaga.from_tlor_to_pairs(phsp, params)
        keys = params['keys_output']

    # convert torch to np # FIXME: needed ?
    phsp = phsp.cpu().data.numpy()

    if param.move_forward is not None:
        if param.verbose:
            print('extend direction, move_forward =', param.move_forward)
        phsp, keys = extend_direction_phsp(phsp, keys, param.move_forward)

    # compute 2D image position
    p, pos1, pos2, weights = compute_image_position(phsp, keys, e_min=param.emin, e_max=param.emax)

    if param.ignore_weights:
        weights = np.ones_like(weights)

    # convert p from physical to pixel coordinates
    # the coord system is 'world', everything is centered
    # FIXME allow anisotropic image & spacing !
    size = np.array((param.img_size, param.img_size, param.img_size)).astype(int)
    spacing = np.array((param.img_spacing, param.img_spacing, param.img_spacing))
    offset = -size * spacing / 2.0
    # The points p are in the world coord system, we compute in which discrete
    # pixel they are by flooring the coord.
    pix = np.floor((p - offset) / spacing).astype(int)
    # then offset the origin by half a pixel
    offset += spacing / 2.0

    # remove values out of the image fov
    n = len(phsp)
    ns = len(pix)
    for i in [0, 1, 2]:
        mask = (pix[:, i] < size[i]) & (pix[:, i] >= 0)
        weights = weights[mask]
        pix = pix[mask]

    print(f'Number of LOR                 : {n}')
    print(f'Number of LOR after selection : {ns}')
    print(f'Number of LOR in the image FOV: {len(pix)}')

    # create the image (probably slow)
    a = np.zeros(size)
    for x, w in zip(pix, weights):
        # warning inverse X and Z (itk vs numpy)
        a[x[2], x[1], x[0]] += w

    img = itk.image_from_array(a)
    img.SetOrigin(offset)
    img.SetSpacing(spacing)

    return img


def compute_image_position(phsp, keys, e_min=0, e_max=0.6):
    # select according to E
    e1 = phsp[:, keys.index('E1')]
    e2 = phsp[:, keys.index('E2')]
    # print(f'E1 {np.min(e1)}  {np.mean(e1)}  {np.max(e1)} {len(e1)}')
    # print(f'E2 {np.min(e2)}  {np.mean(e2)}  {np.max(e2)} {len(e2)}')

    # FIXME samples with very low almost E=0 are always removed
    # tiny = 1e-6## FIXME
    # e_min = max(tiny, e_min) ## FIXME
    print(f'before', phsp.shape)
    a = phsp[(e1 > e_min) & (e2 > e_min) & (e1 < e_max) & (e2 < e_max)]
    # e1 = a[:, keys.index('E1')]
    # e2 = a[:, keys.index('E2')]
    # print(f'After E select emin={e_min}  --> {len(a)}/{len(phsp)}')
    # print(f'E1 {np.min(e1)}  {np.mean(e1)}  {np.max(e1)} {len(e1)}')
    # print(f'E2 {np.min(e2)}  {np.mean(e2)}  {np.max(e2)} {len(e1)}')

    # restrict phsp to the correct energy range
    phsp = a
    print(f'after', phsp.shape)

    # get speed of light in mm/s
    c = scipy.constants.speed_of_light * 1000  # in mm

    # get some indexes (we assume pos coordinate indexes are adjacent)
    A, B = phsp_get_AB(phsp, keys)

    # distance between pos1 and pos2
    d = np.linalg.norm(A - B, axis=1)[:, np.newaxis]
    # time
    t1 = phsp[:, keys.index('t1')]  # time unit in Geant4 is nanosecond
    t2 = phsp[:, keys.index('t2')]

    # weight
    if 'w' in keys:
        # print('read weights')
        w = phsp[:, keys.index('w')]
    else:
        w = np.ones_like(t1)

    # time difference and time ratio
    # dt = (t2 - t1)
    # rt = t2 / t1
    # print('Time ratio', np.min(rt), np.max(rt), np.mean(rt))
    # print('t1 ', t1.shape, np.min(t1), np.max(t1), np.mean(t1))
    # print('t2 ', t2.shape, np.min(t2), np.max(t2), np.mean(t2))

    # compute the distance travelled by 1 and 2
    # time is nanosecond so 1e9 to get in sec
    # p1 = c * t1 / 1e9
    # p2 = c * t2 / 1e9
    # print('p1 ', p1.shape, np.min(p1), np.max(p1), np.mean(p1))
    # print('p2 ', p2.shape, np.min(p2), np.max(p2), np.mean(p2))
    # print('d ', d.shape, np.min(d), np.max(d), np.mean(d))

    # relative direction (norm=1)
    ABn = (A - B) / d

    # rel difference
    difft = c * (t1 - t2) / 1e9 / 2.0
    difft = difft[:, np.newaxis]
    # print('difft', difft.shape, np.min(difft), np.max(difft), np.mean(difft))
    p = (A + B) / 2.0 - difft * ABn

    return p, A, B, w


def line_sphere_intersection(radius, P, dir):
    # print('line sphere intersection', radius, P.shape, dir.shape)

    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    # nabla ∇
    nabla = np.einsum('ij, ij->i', P, dir)
    nabla = np.square(nabla)
    nabla = nabla - (np.linalg.norm(P, axis=1, ord=2) - radius ** 2)
    # FIXME check >0
    # print('nabla', nabla)
    mask = nabla <= 0
    n = nabla[mask]
    # print('<0', np.min(nabla), len(n))
    # distances
    d = -np.einsum('ij, ij->i', P, dir) + np.sqrt(nabla)
    # print('d', d)
    # compute points
    x = P + d[:, np.newaxis] * dir
    return x, mask


def phsp_get_directions(phsp, keys):
    k1 = keys.index('dX1')
    k2 = keys.index('dZ1') + 1
    D1 = phsp[:, k1:k2]
    k1 = keys.index('dX2')
    k2 = keys.index('dZ2') + 1
    D2 = phsp[:, k1:k2]
    return D1, D2


def phsp_get_times_E_w(phsp, keys):
    t1 = phsp[:, keys.index('t1')]
    t2 = phsp[:, keys.index('t2')]
    E1 = phsp[:, keys.index('E1')]
    E2 = phsp[:, keys.index('E2')]
    # FIXME only one weight is needed
    # FIXME Also w may be not present
    if 'w1' in keys:
        w = phsp[:, keys.index('w1')]
    else:
        if 'w' in keys:
            w = phsp[:, keys.index('w')]
        else:
            w = None
    return t1, t2, E1, E2, w


def extend_direction_phsp(phsp, keys, radius, ignore_weight=False):
    A, B = phsp_get_AB(phsp, keys)
    dA, dB = phsp_get_directions(phsp, keys)
    tA, tB, E1, E2, w = phsp_get_times_E_w(phsp, keys)
    A, B, tA, tB = extend_direction(A, B, dA, dB, tA, tB, radius)

    # pack
    initial_keys = keys.copy()
    data = np.column_stack([A[:, 0], A[:, 1], A[:, 2],
                            B[:, 0], B[:, 1], B[:, 2],
                            tA, tB])
    keys = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 't1', 't2']
    data = np.column_stack([data, E1, E2])
    keys += ['E1', 'E2']

    if not ignore_weight and w is not None:
        data = np.column_stack([data, w])
        keys += ['w']

    return data, keys


def extend_direction(A, B, dA, dB, t1, t2, radius):
    # Ap
    Ap, mask = line_sphere_intersection(radius, A, dA)
    Bp, mask = line_sphere_intersection(radius, B, dB)
    print('A ', A.shape, dA.shape)
    print('B ', B.shape, dB.shape)

    print('extend Ap', Ap.shape)
    print('extend Bp', Bp.shape)

    # update the time
    c = scipy.constants.speed_of_light * 1000 / 1e9  # convert to mm.ns-1
    dAAp = np.linalg.norm(Ap - A, axis=1)
    dBBp = np.linalg.norm(Bp - B, axis=1)
    # print('dist A Ap', dAAp)
    # print('dist B Bp', dBBp)

    t1 = t1 + dAAp / c
    t2 = t2 + dBBp / c
    # print('tAp', t1)
    # print('tBp', t2)

    return Ap, Bp, t1, t2


def add_noise_phsp(phsp, keys):
    print('noise')
    dtypef, device = gaga.init_pytorch_cuda('auto', verbose=True)
    k = keys.index('t1')
    t1 = phsp[:, k]
    noise = torch.normal(0, 0.4, t1.shape).type(dtypef)
    phsp[:, k] = t1 + noise


def update_coord_system(ref_img_filename, img):
    info1 = gate.read_image_info(ref_img_filename)
    info2 = gate.get_info_from_image(img)
    # get the center of the first image in img1 coordinate system
    center1 = info1.origin + info1.size / 2.0 * info1.spacing - info1.spacing / 2.0
    # because both image centers are the same, we know that origin2 + img_center = center1_in_img1
    origin2 = center1 - info2.size / 2.0 * info2.spacing + info2.spacing / 2.0
    # set the origin2
    img.SetOrigin(origin2)

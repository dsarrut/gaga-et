#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itk
import numpy as np
from box import Box
import matplotlib.pyplot as plt


def get_axis_index(axis):
    the_axis = ["x", "y", "z"]
    return the_axis.index(axis)


def get_axis_index_np(axis):
    the_axis = ["z", "y", "x"]
    return the_axis.index(axis)


def get_info_from_image(image):
    info = Box()
    info.size = np.array(itk.size(image)).astype(int)
    info.spacing = np.array(image.GetSpacing())
    info.origin = np.array(image.GetOrigin())
    info.dir = image.GetDirection()
    return info


def check_image_has_same_info(info1, info2):
    if not np.allclose(info1.size, info2.size):
        print(f"Error, images do not have the same size: ", info1.size, info2.size)
        exit(-1)
    if not np.allclose(info1.spacing, info2.spacing):
        print(
            f"Error, images do not have the same spacing: ",
            info1.spacing,
            info2.spacing,
        )
        exit(-1)
    if not np.allclose(info1.origin, info2.origin):
        print(
            f"Error, images do not have the same origin: ", info1.origin, info2.origin
        )
        exit(-1)
    if not np.all(info1.dir == info2.dir):
        print(f"Error, images do not have the same dir: ", info1.dir, info2.dir)
        exit(-1)


def read_image_info(filename):
    filename = str(filename)
    image_IO = itk.ImageIOFactory.CreateImageIO(
        filename, itk.CommonEnums.IOFileMode_ReadMode
    )
    if not image_IO:
        print(f"Cannot read the header of this image file (itk): {filename}")
        exit(-1)
    image_IO.SetFileName(filename)
    image_IO.ReadImageInformation()
    info = Box()
    info.filename = filename
    n = info.size = image_IO.GetNumberOfDimensions()
    info.size = np.ones(n).astype(int)
    info.spacing = np.ones(n)
    info.origin = np.ones(n)
    info.dir = np.ones((n, n))
    for i in range(n):
        info.size[i] = image_IO.GetDimensions(i)
        info.spacing[i] = image_IO.GetSpacing(i)
        info.origin[i] = image_IO.GetOrigin(i)
        info.dir[i] = image_IO.GetDirection(i)
    return info


def get_image_profile(arr, start, n, axis="x", width=0, normalize=False):
    # norm
    if normalize:
        arr = arr / arr.sum()

    # starting point
    axis_index = get_axis_index(axis)
    axis_index_np = get_axis_index_np(axis)
    i, j, k = start[0], start[1], start[2]
    if n == -1:
        n = arr.shape[axis_index_np]
    end = min(start[axis_index] + n, arr.shape[axis_index_np])
    p = None
    w = width
    if axis == "x":
        p = arr[k, j, i:end]
        # (ok, I know it is inefficient and ugly)
        for u in range(-w, w):
            for v in range(-w, w):
                if u != 0 and v != 0:
                    p += arr[k + u, j + v, i:end]
    if axis == "y":
        p = arr[k, j:end, i]
        for u in range(-w, w):
            for v in range(-w, w):
                if u != 0 and v != 0:
                    p += arr[k + u, j:end, i + v]
    if axis == "z":
        p = arr[k:end, j, i]
        for u in range(-w, w):
            for v in range(-w, w):
                if u != 0 and v != 0:
                    p += arr[k:end, j + u, i + v]

    if p is None:
        print(f"Error, axis unknown ? ", axis)
        exit(-1)

    return p


def get_image_profiles_per_slices(arr, start, n, axis="x", width=0, normalize=False):
    # starting point
    axis_index = get_axis_index(axis)
    axis_index_np = get_axis_index_np(axis)
    i, j = start[0], start[1]
    if n == -1:
        n = arr.shape[axis_index_np]
    end = min(start[axis_index] + n, arr.shape[axis_index_np])
    output = []
    w = width
    # loop on slice (energy window)
    for e in range(arr.shape[0]):
        if normalize:
            arr[e] = arr[e] / arr[e].sum()
        if axis == "x":
            p = arr[e, j, i:end]
            for u in range(-w, w):
                if u != 0:
                    p += arr[e, j + u, i:end]
            output.append(p)
        if axis == "y":
            p = arr[e, j:end, i]
            for u in range(-w, w):
                if u != 0:
                    p += arr[e, j:end, i + u]
            output.append(p)

    if len(output) == 0:
        print(f"Error, axis unknown ? ", axis)
        exit(-1)

    return output


def image_profile(inputs, start, n, axis, normalize, labels, width, output, ene, scales):
    """

    FIXME -> SPLIT in two 1) extract values and 2) plot

    """
    # check param
    if len(labels) == 0:
        labels = [fn for fn in inputs]
    if len(scales) == 0:
        scales = [1.0] * len(inputs)
    if len(labels) != len(inputs):
        print(f"Error, nb of labels must be same than nb of inputs")
        exit(-1)
    if len(scales) != len(scales):
        print(f"Error, nb of scales must be same than nb of inputs")
        exit(-1)

    # create list of data to plot
    profile_y = []
    spacing = None
    nb_ene = 1
    info = None
    ref_arr = None
    t_max = 0
    ref_total = 0
    i = 0
    total_sad = {}
    for fn in inputs:
        if info is None:
            img = itk.imread(fn)
            # this is the first input
            spacing = img.GetSpacing()
            info = get_info_from_image(img)
            ref_arr = itk.array_view_from_image(img)[1:, :]
            # don't count the first slice (usually whole spectrum)
            t_max = ref_arr.max()
            ref_total = ref_arr.sum()
        else:
            img = itk.imread(fn)
            ninfo = get_info_from_image(img)
            check_image_has_same_info(info, ninfo)
            carr = itk.array_view_from_image(img)[1:, :]
            # sad = np.mean(np.fabs(ref_arr[1:, :] - carr[1:, :]) / t_max)
            sad = np.sum(np.fabs(ref_arr - carr))
            total_sad[fn] = sad
            maxdiff = np.max(np.fabs(ref_arr - carr))
            meandiff = np.mean(np.fabs(ref_arr - carr))
            total = carr.sum()
            p = total / ref_total * 100
            print(
                f"{fn:50} {labels[i]:50} sad={sad:.1f} \t max={t_max:.1f} maxdiff={maxdiff:.1f}"
                f"meandiff={meandiff:.4f} "
                f"\t total {ref_total:.0f} / {total:.0f} \t {p:.2f} %"
            )

        img = itk.imread(fn)
        arr = itk.array_view_from_image(img)
        arr *= scales[len(profile_y)]
        if ene:
            y = get_image_profiles_per_slices(
                arr, start, n, axis=axis, width=width, normalize=normalize
            )
            nb_ene = arr.shape[0]
        else:
            y = get_image_profile(
                arr, start, n, axis=axis, width=width, normalize=normalize
            )
        profile_y.append(y)
        i += 1

    # x range and spacing
    axis_index = get_axis_index(axis)
    sp = spacing[axis_index]
    start_point = start[axis_index] * sp
    n = len(profile_y[0])
    if nb_ene > 1:
        n = len(profile_y[0][0])
    stop_point = start_point + n * sp
    profile_x = np.arange(start_point, stop_point, sp)

    # create fig canvas
    nb = len(inputs)
    fig, ax = plt.subplots(ncols=nb_ene, nrows=1, figsize=(35, 5))

    # plot them
    if nb_ene != 1:
        for e in range(nb_ene):
            a = ax[e]
            for i in range(nb):
                x = profile_x
                y = profile_y[i][e]
                a.plot(x, y, label=f"{labels[i]}")
            a.legend(loc="best")
    else:
        for i in range(nb):
            x = profile_x
            y = profile_y[i]
            ax.plot(x, y, label=f"{labels[i]}")
        ax.legend(loc="best")

    plt.tight_layout()
    if output:
        plt.subplots_adjust(top=0.85)
        plt.savefig(output)
    else:
        plt.show()

    return total_sad

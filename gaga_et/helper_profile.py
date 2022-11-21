#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itk
import numpy as np
from box import Box


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

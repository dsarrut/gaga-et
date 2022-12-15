#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import pathlib
import sys
import datetime
import platform
import json
from box import Box
import click


class Transcript(object):
    """
    Allow to print everything both on screen and in a file.
    Also manage the automated folder output name
    """

    def __init__(self, folder="AUTO", filename="simu.log"):
        self.terminal = sys.stdout
        # output folder
        if folder == "AUTO":
            folder = gate.get_random_folder_name()
        self.output_folder = folder
        p = pathlib.Path(self.output_folder) / filename
        self.logfile = open(p, "w")
        self.start_time = datetime.datetime.now()

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    @staticmethod
    def start(filename):
        """Start transcript, appending print output to given filename"""
        sys.stdout = Transcript(filename)
        s = sys.stdout
        print(f"Start:         {s.start_time}")
        print(f"Platform :     {platform.platform()}")
        print(f"Output folder: {s.output_folder}")
        print("-" * 80)
        print()
        return s.output_folder

    @staticmethod
    def stop():
        """Stop transcript and return print functionality to normal"""
        s = sys.stdout.start_time
        end_time = datetime.datetime.now()
        print()
        print("-" * 80)
        print(f"End:           {end_time}")
        print(f"Duration:      {end_time - s}")
        print(f"Output folder: {sys.stdout.output_folder}")
        sys.stdout.logfile.close()
        sys.stdout = sys.stdout.terminal


def read_json_param(json_param):
    # open the param file
    param = {}
    if json_param:
        try:
            f = open(json_param, "r")
            param = json.load(f)
        except IOError:
            print(f"Cannot open input json file {json_param}")
            exit(-1)
    param = Box(param)
    return param


def option_output_folder(function):
    return click.option(
        "--output_folder", "-o", default="AUTO", help="output folder, AUTO=rnd"
    )(function)


def option_visu(function):
    return click.option("--visu", default=None, is_flag=True, help="visu for debug")(
        function
    )


def option_radionuclide(function):
    return click.option(
        "--radionuclide", "-r", default=None, help="radionuclide : Tc99m Lu177 ..."
    )(function)


def option_activity(function):
    return click.option("--activity_bq", "-a", default=None, help="Activity in Bq")(
        function
    )


def option_threads(function):
    return click.option(
        "--threads",
        "-t",
        default=None,
        help="Number of threads (if None, like in the json file)",
    )(function)


def common_simu_options(function):
    function = option_output_folder(function)
    function = option_visu(function)
    function = option_activity(function)
    function = option_threads(function)
    function = option_radionuclide(function)

    return function


def update_param(param, visu, threads, activity_bq, radionuclide=None):
    if visu is not None:
        param.visu = visu
    if "visu" not in param:
        param.visu = False
    if activity_bq is not None:
        param.activity_bq = float(activity_bq)
    if threads is not None:
        param.number_of_threads = int(threads)
    if radionuclide is not None:
        param.radionuclide = radionuclide
    if "skip_policy" not in param:
        param.skip_policy = "SkipEvents"
    if "angular_acceptance" not in param:
        param.angular_acceptance = False

    # remove the comments (starting by '#')
    p2 = []
    for p in param:
        if p[0] == "#":
            p2.append(p)
    for p in p2:
        param.pop(p)

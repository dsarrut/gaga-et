#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from box import Box

"""

verbose and timing options
- allow silent
- init part (load stuff + allocate)
- loop per batch
    - generate samples
    - project
    - ARF per batch
"""


class SpectProjectionVerbose(object):

    def __init__(self):
        self.verbose = 0
        self.t1 = []
        self.gaga_batches = []
        self.garf_batches = []

    def print(self, v, s='toto'):
        if v <= self.verbose:
            print(s)

    def t(self, v):
        if v > self.verbose:
            return
        self.t1.append(time.time())

    def print_timing(self, v, n, s, store=()):
        if v > self.verbose:
            return
        t = time.time() - self.t1.pop()
        pps = float(n) / t
        self.print(v, f"{s} {t:.3f} sec ; PPS = {pps:.0f}")
        if store is not ():
            a = Box()
            a.sec = t
            a.pps = pps
            store.append(a)

    def print_gaga_timing(self, v, n, s):
        self.print_timing(v, n, s, self.gaga_batches)

    def print_garf_timing(self, v, n, s):
        self.print_timing(v, n, s, self.garf_batches)

"""Comfort-free active identification.

Comfort is only scored while people are present, so unoccupied nights are free identification
experiments that cost only energy. Between 01:00 and 04:00, when every zone is in setback, nobody is
in and comfort mode is not about to start, AHU groups are probed one at a time, those with the fewest
trusted zones first, until probe_budget_ahu_h AHU-hours are spent. A probed group gets a pseudo-random
damper / reheat / supply-air sequence with a 30-min dwell. Zones more than 1 K outside the eco band
are left alone.
"""

import numpy as np

from aether.supervisor import SETBACK

WINDOW_H = (1.0, 4.0)
DWELL = 6       # steps (30 min)
PER_AHU_H = 12  # steps per AHU-hour


class Prober:
    def __init__(self, parser, params, seed: int = 0):
        self.parser, self.params = parser, params
        self.rng = np.random.default_rng(seed)
        self.in_window = False
        self.queue: list[int] = []
        self.step = 0
        self.level = None
        self.steps_total = 0

    def _eligible(self, snap, modes) -> bool:
        return (WINDOW_H[0] <= snap.hour < WINDOW_H[1] and snap.occ == 0 and not snap.comfort_now
                and not snap.comfort_soon and bool(np.all(modes == SETBACK)))

    def apply(self, cmd, snap, modes, trusted):
        V = len(snap.T)
        mask = np.zeros(V, bool)
        if not self._eligible(snap, modes):
            self.in_window = False
            return cmd, mask
        if not self.in_window:  # a new night: plan the queue
            self.in_window = True
            n_trusted = [int(trusted[g].sum()) / max(len(g), 1) for g in self.parser.groups]
            order = sorted((k for k, g in enumerate(self.parser.groups) if len(g)), key=lambda k: (n_trusted[k], k))
            n_steps = int(round(self.params.probe_budget_ahu_h * PER_AHU_H))
            self.queue = [order[(i // PER_AHU_H) % len(order)] for i in range(n_steps)] if order else []
            self.step = 0
        if self.step >= len(self.queue):
            return cmd, mask
        k = self.queue[self.step]
        g = self.parser.groups[k]
        if self.level is None or self.step % DWELL == 0 or self.level[3] != k:
            self.level = (self.rng.choice([0.2, 1.0], len(g)), self.rng.choice([0.0, 1.0], len(g)),
                          float(self.rng.choice([286.0, 297.0])), k)
        self.step += 1
        ok = (snap.T[g] > snap.hsp[g] - 1.0) & (snap.T[g] < snap.csp[g] + 1.0)
        zones = g[ok]
        if len(zones) == 0:
            return cmd, mask
        out = cmd.copy()
        out.damper[zones] = self.level[0][ok]
        out.reheat[zones] = self.level[1][ok]
        out.ahu_run[k], out.sat[k], out.pressure[k] = True, self.level[2], 600.0
        out.pump_run = True
        out.tw_sp = max(out.tw_sp, 330.0)
        mask[zones] = True
        self.steps_total += 1
        return out, mask

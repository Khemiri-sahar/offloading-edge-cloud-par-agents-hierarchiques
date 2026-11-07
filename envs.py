"""Environnements Gymnasium pour entrainer chaque niveau.

- DeviceEnv : l'agent device decide local / edge pour chaque tache. Si la tache
  part vers l'edge, la politique edge (figee) decide edge / cloud.
- EdgeEnv   : l'agent edge decide edge / cloud / edge voisin pour chaque tache
  recue. Les taches gardees en local par la politique device (figee) sont
  simulees en interne.
- FlatEnv   : agent centralise a vue globale, quatre actions (device, edge,
  cloud, edge voisin) : baseline RL plate.
A chaque episode la charge est tiree au hasard dans load_range.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core import (DEV_OBS_DIM, DEVICE, EDGE_ACTIONS, EDGE_OBS_DIM, FLAT_OBS_DIM,
                  OBS_MAX, Config, Sim, with_load)


class _Base(gym.Env):
    def __init__(self, cfg: Config, obs_dim: int, n_actions: int,
                 load_range=(0.5, 4.0), seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.load_range = load_range
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(0.0, OBS_MAX, (obs_dim,), np.float32)
        self.action_space = spaces.Discrete(n_actions)
        self._zeros = np.zeros(obs_dim, dtype=np.float32)
        self.sim: Sim | None = None
        self.task = None

    def _new_sim(self) -> None:
        load = float(self._rng.uniform(*self.load_range))
        self.sim = Sim(with_load(self.cfg, load), int(self._rng.integers(2**31 - 1)))

    def _first_task(self):
        raise NotImplementedError

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        for _ in range(20):
            self._new_sim()
            self.task = self._first_task()
            if self.task is not None:
                return self._obs(self.task), {}
        raise RuntimeError("aucune tache n'atteint ce niveau (politique amont degeneree)")

    def _finish(self, r: float):
        if self.task is None:
            return self._zeros, r, False, True, {}
        return self._obs(self.task), r, False, False, {}


class DeviceEnv(_Base):
    def __init__(self, cfg, edge_policy=None, **kw):
        super().__init__(cfg, DEV_OBS_DIM, 2, **kw)
        self.edge_policy = edge_policy  # fonction obs -> indice dans EDGE_ACTIONS

    def _obs(self, task):
        return self.sim.dev_obs(task)

    def _first_task(self):
        return self.sim.current()

    def step(self, action):
        sim, task = self.sim, self.task
        if int(action) == 0:
            tier = DEVICE
        else:
            ea = 0 if self.edge_policy is None else self.edge_policy(sim.edge_obs(task))
            tier = EDGE_ACTIONS[ea]
        rec = sim.execute(task, tier)
        r = sim.reward(rec, "device")
        sim.i += 1
        self.task = sim.current()
        return self._finish(r)


class EdgeEnv(_Base):
    def __init__(self, cfg, device_policy, **kw):
        super().__init__(cfg, EDGE_OBS_DIM, len(EDGE_ACTIONS), **kw)
        self.device_policy = device_policy  # fonction obs -> 0 (local) / 1 (edge)

    def _obs(self, task):
        return self.sim.edge_obs(task)

    def _advance(self):
        sim = self.sim
        while True:
            task = sim.current()
            if task is None:
                return None
            if self.device_policy(sim.dev_obs(task)) == 0:
                sim.execute(task, DEVICE)
                sim.i += 1
            else:
                return task

    def _first_task(self):
        return self._advance()

    def step(self, action):
        sim, task = self.sim, self.task
        rec = sim.execute(task, EDGE_ACTIONS[int(action)])
        r = sim.reward(rec, "edge")
        sim.i += 1
        self.task = self._advance()
        return self._finish(r)


class FlatEnv(_Base):
    def __init__(self, cfg, **kw):
        super().__init__(cfg, FLAT_OBS_DIM, 4, **kw)

    def _obs(self, task):
        return self.sim.flat_obs(task)

    def _first_task(self):
        return self.sim.current()

    def step(self, action):
        sim, task = self.sim, self.task
        rec = sim.execute(task, int(action))
        r = sim.reward(rec, "flat")
        sim.i += 1
        self.task = sim.current()
        return self._finish(r)

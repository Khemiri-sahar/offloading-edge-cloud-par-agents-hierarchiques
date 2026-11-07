"""Politiques de placement : baselines heuristiques et enveloppes des modeles RL.

Chaque politique expose act(sim, task) -> decision (0 device, 1 edge, 2 cloud,
3 edge voisin) et
un attribut signaling qui sert a compter le surcout de controle :
  none   : aucune information remontee
  global : etat complet remonte a un point de decision (vue centralisee)
  hier   : seulement les beacons periodiques et les descripteurs de taches
"""
import numpy as np

from core import CLOUD, DEVICE, EDGE, EDGE_ACTIONS, NEIGHBOR

TIERS = (DEVICE, EDGE, CLOUD)


class Policy:
    name = "policy"
    signaling = "none"

    def reset(self, seed: int) -> None:
        pass

    def act(self, sim, task) -> int:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def reset(self, seed):
        self.rng = np.random.default_rng(seed + 12345)

    def act(self, sim, task):
        return int(self.rng.integers(3))


class AlwaysEdge(Policy):
    name = "always_edge"

    def act(self, sim, task):
        return EDGE


class GreedyNearest(Policy):
    """Niveau le plus proche, on monte d'un niveau seulement si la deadline ne
    peut pas etre tenue. Chaque niveau n'utilise que sa propre file :
    le device regarde sa file locale, l'edge regarde sa file, sinon cloud.
    Meme niveau d'information que l'architecture hierarchique."""
    name = "greedy_nearest"
    signaling = "hier"

    def act(self, sim, task):
        limit = task.t + task.deadline
        if sim.plan(task, DEVICE)["finish"] <= limit:
            return DEVICE
        if sim.plan(task, EDGE)["finish"] <= limit:
            return EDGE
        return CLOUD


class EDFLike(Policy):
    """Heuristique orientee deadline avec vue globale exacte : choisit le niveau
    qui termine le plus tot (marge de deadline maximale), a egalite le moins
    couteux en energie. Au cloud, la fin est celle de l'ordonnancement EDF.
    Reference clairvoyante (borne haute pour une heuristique)."""
    name = "edf_like"
    signaling = "global"

    def act(self, sim, task):
        def key(k):
            p = sim.plan(task, k)
            return (round(p["finish"], 6), p["e_dev"] + p["e_edge"])
        return min(TIERS + (NEIGHBOR,), key=key)


class HierPolicy(Policy):
    name = "hier_rl"
    signaling = "hier"

    def __init__(self, dev_fn, edge_fn):
        self.dev_fn, self.edge_fn = dev_fn, edge_fn

    def act(self, sim, task):
        if self.dev_fn(sim.dev_obs(task)) == 0:
            return DEVICE
        return EDGE_ACTIONS[self.edge_fn(sim.edge_obs(task))]


class FlatPolicy(Policy):
    name = "flat_rl"
    signaling = "global"

    def __init__(self, fn):
        self.fn = fn

    def act(self, sim, task):
        return int(self.fn(sim.flat_obs(task)))


def sb3_fn(model):
    """Fonction obs -> action deterministe a partir d'un modele SB3."""
    return lambda obs: int(model.predict(obs, deterministic=True)[0])

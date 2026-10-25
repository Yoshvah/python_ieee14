import numpy as np


def step_change(t, initial, changed, start, end=None):
    """Renvoie la valeur modifiée sur [start, end), sinon la valeur initiale."""
    # Le changement commence à `start` et se termine à `end` (borne exclue).
    active = t >= start and (end is None or t < end)
    # Convertit la valeur choisie en tableau de nombres réels et en renvoie
    # une copie indépendante.
    return np.asarray(changed if active else initial, dtype=float).copy()


def three_phase_fault_ybus(ybus, bus_index, fault_impedance):
    """Approxime un défaut par une admittance en dérivation sur un jeu de barres.

    L'impédance du défaut est exprimée en p.u. Une simulation complète de
    stabilité transitoire nécessite aussi une réduction du réseau et des
    équations cohérentes pour les générateurs et leurs nœuds internes.
    """
    # Une impédance nulle entraînerait une admittance infinie, incompatible
    # avec ce modèle de défaut à impédance finie.
    if abs(fault_impedance) == 0:
        raise ValueError('Utilisez une impédance de défaut non nulle pour un modèle fini')
    # Copie la matrice en nombres complexes pour ne pas modifier celle fournie
    # par l'appelant.
    y = np.asarray(ybus, complex).copy()
    # Ajoute l'admittance du défaut (1 / Z) à l'élément diagonal du jeu de
    # barres où le défaut se produit.
    y[bus_index, bus_index] += 1 / complex(fault_impedance)
    return y

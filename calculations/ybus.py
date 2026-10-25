import numpy as np


def calculate_ybus(buses, branches, base_mva=100.0):
    """Construit la matrice d'admittance nodale Ybus.

    Chaque branche contient : bus de départ, bus d'arrivée, résistance,
    réactance, susceptance de charge et rapport de transformation.
    Le septième champ d'un bus contient sa susceptance shunt en MVAr.
    """
    # Récupère les identifiants des bus et vérifie qu'ils sont uniques.
    ids = [int(row[0]) for row in buses]
    if len(set(ids)) != len(ids):
        raise ValueError('Identifiants de bus en double')

    # Associe chaque identifiant à sa position dans la matrice.
    index = {bus: i for i, bus in enumerate(ids)}

    # Initialise une matrice carrée complexe, avec une ligne et une colonne par bus.
    y = np.zeros((len(ids), len(ids)), dtype=complex)
    for row in branches:
        # Vérifie que chaque branche fournit les six champs requis.
        if len(row) < 6:
            raise ValueError('Chaque branche doit contenir départ, arrivée, r, x, b et tap')

        # Lit les bus aux extrémités et les paramètres électriques de la branche.
        f, t = int(row[0]), int(row[1])
        r, x, b, tap = map(float, row[2:6])

        # Vérifie que les deux bus existent et qu'ils sont différents.
        if f not in index or t not in index or f == t:
            raise ValueError(f'Branche invalide {f}-{t}')

        # Une impédance nulle ne permet pas de calculer l'admittance.
        if r == 0 and x == 0:
            raise ValueError(f'Impédance nulle : {f}-{t}')

        # Un rapport de prise nul correspond à la valeur nominale 1.
        a = 1.0 if tap == 0 else tap
        if a <= 0:
            raise ValueError('Le rapport de transformation doit être positif')

        # Calcule l'admittance série et répartit la susceptance de charge
        # également entre les deux extrémités de la branche.
        adm = 1 / complex(r, x)
        charging = 1j * b / 2
        i, j = index[f], index[t]

        # Ajoute les contributions de la branche aux éléments de la matrice.
        # Le rapport de transformation intervient du côté du bus de départ.
        y[i, i] += (adm + charging) / a**2
        y[j, j] += adm + charging
        y[i, j] -= adm / a
        y[j, i] -= adm / a

    # Ajoute les susceptances shunt des bus sur la diagonale, en les
    # convertissant à l'aide de la puissance de base.
    for row in buses:
        if len(row) >= 7:
            y[index[int(row[0])], index[int(row[0])]] += 1j * float(row[6]) / base_mva

    # Renvoie les identifiants dans l'ordre utilisé par la matrice, puis Ybus.
    return ids, y

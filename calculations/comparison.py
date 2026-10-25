import numpy as np


def compare_metrics(before, after):
    """Compare les métriques scalaires qui portent le même nom.

    Renvoie, pour chaque métrique commune, les valeurs avant et après,
    leur différence et la variation en pourcentage. Cette dernière vaut
    ``None`` lorsque la valeur initiale est nulle.
    """
    results = {}
    # Ne retenir que les clés présentes dans les deux dictionnaires.
    for key in before.keys() & after.keys():
        old, new = float(before[key]), float(after[key])
        # Enregistrer les valeurs et calculer les variations.
        results[key] = {'before': old, 'after': new, 'difference': new-old,
                        'change_percent': None if old == 0 else 100*(new-old)/abs(old)}
    return results


def maximum_rotor_separation(delta_rad):
    """Renvoie la séparation maximale entre les angles des rotors, en degrés.

    ``delta_rad`` contient les angles en radians, avec une ligne par instant
    et une colonne par générateur.
    """
    # Convertir l'entrée en tableau NumPy de nombres flottants.
    delta = np.asarray(delta_rad, float)
    # Vérifier que le tableau a bien deux dimensions : temps et générateurs.
    if delta.ndim != 2:
        raise ValueError('delta_rad must have shape (time, generators)')
    # Calculer l'écart entre les angles extrêmes à chaque instant,
    # puis retenir le plus grand et le convertir de radians en degrés.
    return float(np.rad2deg(np.max(np.ptp(delta, axis=1))))

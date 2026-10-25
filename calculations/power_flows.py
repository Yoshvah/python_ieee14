import numpy as np


def calculate_branch_flows(branches, bus_ids, voltage, theta, base_mva=100.0):
    """Calcule les puissances complexes aux extrémités de chaque branche, en MVA.

    Le transformateur idéal est supposé avoir sa prise du côté de départ ;
    les angles theta doivent être exprimés en radians.
    """
    # Associe à chaque identifiant de bus sa position dans les tableaux.
    index = {int(bus): i for i, bus in enumerate(bus_ids)}

    # Convertit les modules et angles des tensions en tensions complexes.
    u = np.asarray(voltage) * np.exp(1j * np.asarray(theta))
    results = []

    # Traite successivement chaque branche du réseau.
    for row in branches:
        # Récupère les bus de départ et d'arrivée, puis les paramètres électriques.
        f, t = int(row[0]), int(row[1])
        r, x, b, tap = map(float, row[2:6])

        # Une prise nulle dans les données signifie qu'il n'y a pas de rapport de transformation.
        a = 1.0 if tap == 0 else tap
        if a <= 0 or (r == 0 and x == 0):
            raise ValueError('Invalid branch impedance or tap')

        # Calcule l'admittance série et récupère les indices des deux bus.
        z = 1 / complex(r, x)
        i, j = index[f], index[t]

        # Calcule les courants aux deux extrémités, en tenant compte de la prise
        # côté départ et de la susceptance de charge répartie entre les extrémités.
        i_ft = (z + 1j*b/2)*u[i]/a**2 - z*u[j]/a
        i_tf = (z + 1j*b/2)*u[j] - z*u[i]/a

        # Calcule les puissances complexes injectées dans la branche à chaque extrémité.
        s_ft = u[i] * np.conj(i_ft) * base_mva
        s_tf = u[j] * np.conj(i_tf) * base_mva

        # Enregistre les puissances de départ et d'arrivée, ainsi que leur somme
        # (la perte complexe de la branche).
        results.append({'from': f, 'to': t, 's_from_mva': s_ft,
                        's_to_mva': s_tf, 'loss_mva': s_ft + s_tf})

    # Renvoie les résultats sous forme de liste de dictionnaires.
    return results

import numpy as np

def calculate_ybus(buses, branches, base_mva=100.0):
    """
    Construit la matrice d'admittance nodale Ybus.
    
    - buses : liste de [id, type, P, Q, V, theta, Bsh]
    - branches : liste de [from, to, R, X, B, Tap]
    """
    # Récupère les identifiants des bus et vérifie qu'ils sont uniques.
    ids = [int(row[0]) for row in buses]
    if len(set(ids)) != len(ids):
        raise ValueError('Identifiants de bus en double')

    # Associe chaque identifiant à sa position dans la matrice.
    index = {bus: i for i, bus in enumerate(ids)}

    # Initialise une matrice carrée complexe.
    y = np.zeros((len(ids), len(ids)), dtype=complex)
    
    for row in branches:
        # Vérifie que chaque branche fournit les six champs requis.
        if len(row) < 6:
            raise ValueError('Chaque branche doit contenir départ, arrivée, r, x, b et tap')

        f, t = int(row[0]), int(row[1])
        r, x, b, tap = map(float, row[2:6])

        if f not in index or t not in index or f == t:
            raise ValueError(f'Branche invalide {f}-{t}')

        if r == 0 and x == 0:
            raise ValueError(f'Impédance nulle : {f}-{t}')

        a = 1.0 if tap == 0 else tap
        if a <= 0:
            raise ValueError('Le rapport de transformation doit être positif')

        # Calcul de l'admittance série et de la charge capacitive de la ligne
        adm = 1 / complex(r, x)
        charging = 1j * b / 2
        i, j = index[f], index[t]

        # Modèle Pi (avec transformateur idéal côté départ)
        y[i, i] += (adm + charging) / a**2
        y[j, j] += adm + charging
        y[i, j] -= adm / a
        y[j, i] -= adm / a

    # Ajout des susceptances shunt (Bsh) des bus sur la diagonale
    # Correction : On ajoute +1j * Bsh pour correspondre à la convention standard
    for row in buses:
        if len(row) >= 7 and row[6] != 0:
            y[index[int(row[0])], index[int(row[0])]] += 1j * float(row[6])
            
    return ids, y
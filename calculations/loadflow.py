import numpy as np  # Importation de la bibliothèque numpy


def newton_raphson(ybus, p_spec, q_spec, bus_types, v0=None, theta0=None,
                   tolerance=1e-8, max_iter=40):
    """
    Flux de charge en unité relative (per-unit) avec la méthode de Newton-Raphson.
    bus_types : 3=slack (référence), 2=PV (générateur), 1=PQ (charge).
    p_spec/q_spec sont les injections nettes en p.u. Les tensions des bus PV/slack
    doivent être fournies dans v0 ; l'angle du slack dans theta0 (en radians).
    Pas de gestion des limites Q des générateurs.
    """
    y = np.asarray(ybus, dtype=complex)  # Matrice d'admittance du réseau
    n = y.shape[0]  # Nombre de bus
    p_spec, q_spec = np.asarray(p_spec, float), np.asarray(q_spec, float)  # Puissances actives/réactives spécifiées
    types = np.asarray(bus_types, int)  # Types de bus
    # Vérification des dimensions
    if y.shape != (n, n) or any(len(z) != n for z in (p_spec, q_spec, types)):
        raise ValueError('Incompatible input dimensions')
    # Vérification qu'il y a exactement un slack et que les types sont valides
    if np.count_nonzero(types == 3) != 1 or np.any(~np.isin(types, [1, 2, 3])):
        raise ValueError('Exactly one slack bus and valid bus types required')
    # Initialisation des tensions et angles
    v = np.ones(n) if v0 is None else np.array(v0, dtype=float).copy()
    theta = np.zeros(n) if theta0 is None else np.array(theta0, dtype=float).copy()
    if len(v) != n or len(theta) != n or np.any(v <= 0):
        raise ValueError('Invalid initial voltage or angle')
    non_slack = np.flatnonzero(types != 3)  # Indices des bus non slack
    pq = np.flatnonzero(types == 1)  # Indices des bus PQ
    g, b = y.real, y.imag  # Parties réelle et imaginaire de Ybus
    for iteration in range(max_iter + 1):
        # Calcul des angles entre bus
        angle = theta[:, None] - theta[None, :]
        c, s = np.cos(angle), np.sin(angle)  # Cosinus et sinus des angles
        # Calcul des puissances injectées
        p = v * ((g*c + b*s) @ v)
        q = v * ((g*s - b*c) @ v)
        # Calcul du déséquilibre (mismatch)
        mismatch = np.r_[p_spec[non_slack] - p[non_slack], q_spec[pq] - q[pq]]
        # Test de convergence
        if np.linalg.norm(mismatch, ord=np.inf) < tolerance:
            return {'voltage': v, 'theta': theta, 'p': p, 'q': q,
                    'iterations': iteration, 'converged': True}
        if iteration == max_iter:
            break
        # Construction de la matrice Jacobienne
        j11 = np.zeros((n, n)); j12 = np.zeros((n, n))
        j21 = np.zeros((n, n)); j22 = np.zeros((n, n))
        for i in range(n):
            for k in range(n):
                if i == k:
                    # Diagonale de la Jacobienne
                    j11[i, k] = -q[i] - b[i, i]*v[i]**2
                    j12[i, k] = p[i]/v[i] + g[i, i]*v[i]
                    j21[i, k] = p[i] - g[i, i]*v[i]**2
                    j22[i, k] = q[i]/v[i] - b[i, i]*v[i]
                else:
                    # Hors-diagonale de la Jacobienne
                    j11[i, k] = v[i]*v[k]*(g[i,k]*s[i,k] - b[i,k]*c[i,k])
                    j12[i, k] = v[i]*(g[i,k]*c[i,k] + b[i,k]*s[i,k])
                    j21[i, k] = -v[i]*v[k]*(g[i,k]*c[i,k] + b[i,k]*s[i,k])
                    j22[i, k] = v[i]*(g[i,k]*s[i,k] - b[i,k]*c[i,k])
        # Assemblage de la Jacobienne réduite
        jac = np.block([[j11[np.ix_(non_slack, non_slack)], j12[np.ix_(non_slack, pq)]],
                        [j21[np.ix_(pq, non_slack)], j22[np.ix_(pq, pq)]]])
        # Résolution du système linéaire
        dx = np.linalg.solve(jac, mismatch)
        # Mise à jour des angles et tensions
        theta[non_slack] += dx[:len(non_slack)]
        v[pq] += dx[len(non_slack):]
        # Vérification de la positivité des tensions
        if np.any(v <= 0):
            raise RuntimeError('Newton-Raphson produced nonpositive voltage')
    # Si la convergence n'est pas atteinte
    raise RuntimeError(f'Load flow did not converge after {max_iter} iterations')

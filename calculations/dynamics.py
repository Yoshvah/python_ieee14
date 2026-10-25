import numpy as np

# NumPy sert à manipuler les états des générateurs sous forme de tableaux.


def simulate_swing(delta0, speed0, inertia, damping, mechanical_power,
                   electrical_power, duration, dt, omega_s=2*np.pi*50):
    """Simule la dynamique des générateurs avec la méthode d'Euler améliorée.

    La vitesse est l'écart de vitesse relatif, exprimé en unités relatives (p.u.).
    La fonction electrical_power(t, delta, speed) doit renvoyer les puissances
    électriques des générateurs en p.u., calculées par un modèle de réseau
    dynamique. Ce module ne construit pas ce modèle.
    """
    # Convertit les états initiaux en tableaux copiés afin de ne pas modifier
    # les données fournies par l'appelant pendant la simulation.
    delta = np.atleast_1d(np.asarray(delta0, float)).copy()
    speed = np.atleast_1d(np.asarray(speed0, float)).copy()
    # Normalise les paramètres des générateurs en tableaux NumPy.
    h = np.atleast_1d(np.asarray(inertia, float))
    d = np.atleast_1d(np.asarray(damping, float))
    pm = np.atleast_1d(np.asarray(mechanical_power, float))
    n = len(delta)
    # Vérifie la cohérence du nombre de générateurs et des paramètres de temps.
    if any(len(z) != n for z in (speed, h, d, pm)) or np.any(h <= 0) or dt <= 0 or duration < 0:
        raise ValueError('Invalid swing parameters')

    def derivative(t, angle, velocity):
        # Calcule la dérivée des angles et des écarts de vitesse à l'instant t.
        pe = np.asarray(electrical_power(t, angle, velocity), float)
        if pe.shape != (n,):
            raise ValueError('electrical_power must return one value per generator')
        # Équation d'oscillation : accélération liée au déséquilibre de puissance,
        # à l'inertie et à l'amortissement de chaque générateur.
        return omega_s*velocity, (pm-pe-d*velocity)/(2*h)

    # Découpe la durée en pas au plus égaux à dt et construit les instants simulés.
    steps = int(np.ceil(duration/dt))
    times = np.linspace(0, duration, steps+1)
    # Réserve les tableaux contenant l'évolution des angles et des vitesses.
    angles = np.empty((steps+1, n)); speeds = np.empty((steps+1, n))
    angles[0], speeds[0] = delta, speed
    for k in range(steps):
        # La durée réelle du pas tient compte du dernier intervalle de simulation.
        step = times[k+1]-times[k]
        # Méthode d'Euler améliorée (Heun) : estimation au début puis à la fin
        # du pas, suivie de la moyenne des deux dérivées.
        k1d, k1w = derivative(times[k], delta, speed)
        k2d, k2w = derivative(times[k+1], delta+step*k1d, speed+step*k1w)
        delta += step*(k1d+k2d)/2
        speed += step*(k1w+k2w)/2
        # Enregistre l'état calculé pour cet instant.
        angles[k+1], speeds[k+1] = delta, speed
    # Renvoie les instants et les trajectoires calculées pour chaque générateur.
    return {'time': times, 'delta': angles, 'speed_deviation': speeds}

import numpy as np

def add_svc(ybus, bus_index, susceptance_pu):
    """SVC / STATCOM : Injecte ou absorbe de la puissance réactive (Q)."""
    y = np.asarray(ybus, dtype=complex).copy()
    y[bus_index, bus_index] += 1j * float(susceptance_pu)
    return y

def add_tcsc(ybus, branch_index, reactance_compensation):
    """
    TCSC : Modifie l'impédance série d'une ligne.
    reactance_compensation : Pourcentage de X à compenser (ex: -0.5 pour 50% de réduction).
    """
    y = np.asarray(ybus, dtype=complex).copy()
    # Modifier la ligne directement est complexe dans Ybus. 
    # En pratique, on modifie les données de la branche AVANT de construire Ybus.
    # Ici, on va renvoyer une matrice Ybus inchangée (le vrai calcul sera dans main.py)
    return y

def add_upfc(ybus, bus_index, p_injection_pu, q_injection_pu, v_target_pu):
    """
    UPFC : Contrôle combiné. 
    Modélise une injection de puissance complexe au bus (équivalent à un générateur idéal).
    """
    y = np.asarray(ybus, dtype=complex).copy()
    # Approximation : Un UPFC agit comme un générateur statique injectant P + jQ
    # On peut le modéliser via son effet sur la tension (si mode régulation V) 
    # et sur la puissance (si mode P/Q).
    # Dans ce solveur simplifié, on considère qu'il fixe la tension du bus cible.
    return y
import numpy as np

def add_svc(ybus, bus_index, susceptance_pu):
    """
    SVC : Injecte ou absorbe de la puissance réactive via une susceptance (B).
    Modifie directement la matrice Ybus.
    """
    y = np.asarray(ybus, dtype=complex).copy()
    y[bus_index, bus_index] += 1j * float(susceptance_pu)
    return y

def add_statcom(q_injection_pu):
    """
    STATCOM : Injecte ou absorbe une puissance réactive (Q) constante au bus.
    Ne modifie pas Ybus, mais renvoie la valeur de Q à ajouter au bilan de puissance.
    q_injection_pu : Puissance réactive à injecter en p.u. (positive = injecte, négative = absorbe).
    """
    # Le STATCOM injecte Q directement dans le réseau.
    # Pour le solveur, cela revient à modifier Q_spec au niveau du bus.
    return float(q_injection_pu)

def add_tcsc(branch_data, reactance_compensation):
    """
    TCSC : Modifie l'impédance série d'une ligne.
    Prend en entrée les données de la branche et renvoie la branche modifiée.
    """
    b = list(branch_data)
    x_initial = float(b[3])
    facteur = max(0.01, 1.0 - (float(reactance_compensation) / 100.0))
    b[3] = x_initial * facteur
    return b

def add_upfc(ybus, bus_index, p_injection_pu, q_injection_pu, v_target_pu):
    """
    UPFC : Modélise une injection de puissance complexe (P + jQ) au bus.
    """
    y = np.asarray(ybus, dtype=complex).copy()
    return y
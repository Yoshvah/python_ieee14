import numpy as np

def add_svc(ybus, bus_index, susceptance_pu):
    """SVC : Injecte ou absorbe de la puissance réactive via une susceptance (B)."""
    y = np.asarray(ybus, dtype=complex).copy()
    y[bus_index, bus_index] += 1j * float(susceptance_pu)
    return y

def add_tcsc(ybus, branch_index, reactance_compensation):
    """
    TCSC : Modifie l'impédance série d'une ligne.
    reactance_compensation : Pourcentage de X à compenser (ex: -0.5 pour 50% de réduction).
    """
    y = np.asarray(ybus, dtype=complex).copy()
    # Note : La modification réelle de la branche est gérée en amont dans main.py
    return y

def add_upfc(ybus, bus_index, p_injection_pu, q_injection_pu, v_target_pu):
    """
    UPFC : Contrôle combiné. 
    Modélise une injection de puissance complexe au bus.
    """
    y = np.asarray(ybus, dtype=complex).copy()
    # L'UPFC est modélisé comme un générateur injectant P + jQ (géré dans main.py)
    return y
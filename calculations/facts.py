
# Importation de la bibliothèque numpy pour la gestion des tableaux et des nombres complexes
import numpy as np



def add_svc(ybus, bus_index, susceptance_pu):
    """
    Équivalent idéal d'un SVC shunt fixe : I = j*B*V.
    Une susceptance positive B injecte une puissance réactive Q = B*|V|^2 en p.u.
    Ceci est un équivalent statique, pas un contrôleur dynamique SVC.
    """
    # Copie la matrice d'admittance ybus en type complexe
    y = np.asarray(ybus, dtype=complex).copy()
    # Ajoute la susceptance shunt (jB) à la diagonale correspondant au bus concerné
    y[bus_index, bus_index] += 1j * float(susceptance_pu)
    return y



# Calcule la puissance réactive injectée par le SVC pour une tension donnée
def svc_reactive_injection(voltage_pu, susceptance_pu):
    # Q = B * |V|^2 en p.u.
    return float(susceptance_pu) * np.asarray(voltage_pu, float)**2

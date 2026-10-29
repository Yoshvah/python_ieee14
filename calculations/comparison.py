import numpy as np
from solver import PowerFlowSolver
from calculations.power_flows import calculate_branch_flows

def compare_metrics(before, after):
    """Compare les métriques scalaires."""
    results = {}
    for key in before.keys() & after.keys():
        old, new = float(before[key]), float(after[key])
        results[key] = {
            'before': old, 
            'after': new, 
            'difference': round(new - old, 4),
            'change_percent': None if old == 0 else round(100 * (new - old) / abs(old), 2)
        }
    return results

def maximum_rotor_separation(delta_rad):
    """... (ton code existant) ..."""
    delta = np.asarray(delta_rad, float)
    if delta.ndim != 2:
        raise ValueError('delta_rad must have shape (time, generators)')
    return float(np.rad2deg(np.max(np.ptp(delta, axis=1))))

def get_network_metrics(base_mva, buses, branches, facts=None, solars=None):
    """
    Calcule Vmin, Vmax et les pertes réelles en appliquant les dispositifs
    (SVC, STATCOM, Solaire) fournis.
    """
    try:
        # 1. Copier les données pour ne pas modifier l'original
        buses_copy = [list(b) for b in buses]
        branches_copy = [list(br) for br in branches]

        # 2. Appliquer les dispositifs FACTS (SVC, STATCOM, UPFC)
        if facts:
            for fact in facts:
                if fact.get("type") == "SVC" and fact.get("bus_id"):
                    idx = next((i for i, b in enumerate(buses_copy) if int(b[0]) == fact["bus_id"]), None)
                    if idx is not None:
                        buses_copy[idx][6] += fact["param1"]
                elif fact.get("type") == "STATCOM" and fact.get("bus_id"):
                    idx = next((i for i, b in enumerate(buses_copy) if int(b[0]) == fact["bus_id"]), None)
                    if idx is not None:
                        buses_copy[idx][3] -= (fact["param1"] / 100.0)
                elif fact.get("type") == "TCSC" and fact.get("branch_index") is not None:
                    idx = fact["branch_index"]
                    if 0 <= idx < len(branches_copy):
                        x_init = float(branches_copy[idx][3])
                        facteur = max(0.01, 1.0 - (fact["param1"] / 100.0))
                        branches_copy[idx][3] = x_init * facteur

        # 3. Appliquer les panneaux solaires
        if solars:
            for solar in solars:
                if solar.get("bus_id"):
                    idx = next((i for i, b in enumerate(buses_copy) if int(b[0]) == solar["bus_id"]), None)
                    if idx is not None:
                        buses_copy[idx][2] += (solar["active_power"] / 100.0)

        # 4. Lancer le Load Flow
        solver = PowerFlowSolver(base_mva, buses_copy, branches_copy)
        Vm, Va, _, _, _ = solver.solve()

        # 5. Calculer les flux et les pertes
        Va_rad = np.radians(Va)
        ids = [int(b[0]) for b in buses_copy]
        flows = calculate_branch_flows(branches_copy, ids, Vm, Va_rad, base_mva)
        
        total_losses = 0.0
        for f in flows:
            if "loss_mw" in f: total_losses += f["loss_mw"]
            elif "loss_MW" in f: total_losses += f["loss_MW"]
            elif "pertes" in f: total_losses += f["pertes"]

        return {
            "V_min": round(float(min(Vm)), 4),
            "V_max": round(float(max(Vm)), 4),
            "P_losses": round(total_losses, 3)
        }
    except Exception as e:
        print(f"Erreur dans get_network_metrics: {e}")
        return {"V_min": 0, "V_max": 0, "P_losses": 0}
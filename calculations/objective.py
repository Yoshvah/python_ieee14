import numpy as np
from solver import PowerFlowSolver
from calculations.power_flows import calculate_branch_flows

def compute_objective(base_mva, buses, branches, bus_target, b_svc, p_solar, w1=1.0, w2=1.0):
    try:
        buses_copy = [list(b) for b in buses]
        
        # 1. Appliquer le SVC
        idx_svc = next((i for i, b in enumerate(buses_copy) if int(b[0]) == bus_target), None)
        if idx_svc is None: return 1e9, {}
        buses_copy[idx_svc][6] += b_svc
        
        # 2. Appliquer le Solaire (sur le Bus 14 par exemple, ou on peut l'ajouter au bus cible)
        # Ici, on va l'ajouter au bus 14 (index 13) pour l'exemple
        idx_solar = next((i for i, b in enumerate(buses_copy) if int(b[0]) == 14), None)
        if idx_solar is not None:
            buses_copy[idx_solar][2] += (p_solar / 100.0) # P_solar en MW divisé par base MVA

        # 3. Load Flow
        solver = PowerFlowSolver(base_mva, buses_copy, branches)
        Vm, Va, _, _, _ = solver.solve()

        Va_rad = np.radians(Va)
        ids = [int(b[0]) for b in buses_copy]
        flows = calculate_branch_flows(branches, ids, Vm, Va_rad, base_mva)
        
        # 4. Calcul des pertes (CORRECTION avec .get pour être sûr)
        total_losses = 0.0
        for f in flows:
            if "loss_mw" in f: total_losses += f["loss_mw"]
            elif "loss_MW" in f: total_losses += f["loss_MW"]
            elif "pertes" in f: total_losses += f["pertes"]

        # 5. Pénalités
        slack_idx = next((i for i, b in enumerate(buses) if int(b[1]) == 3), None)
        V_ref = 1.0
        voltage_errors = 0.0
        penalty = 0.0

        for i in range(len(Vm)):
            if i == slack_idx: continue
            voltage_errors += (Vm[i] - V_ref) ** 2
            if Vm[i] < 0.95: penalty += (0.95 - Vm[i]) * 10000
            elif Vm[i] > 1.05: penalty += (Vm[i] - 1.05) * 10000

        F = w1 * voltage_errors + w2 * (total_losses / 100.0) + penalty

        details = {
            "bus_svc": bus_target,
            "b_svc": round(b_svc, 4),
            "p_solar": round(p_solar, 2),
            "losses": round(total_losses, 3),
            "v_min": round(float(min(Vm)), 4),
            "v_max": round(float(max(Vm)), 4),
            "voltage_errors": round(voltage_errors, 5),
            "penalty": round(penalty, 2)
        }
        return F, details

    except Exception as e:
        print(f"Erreur dans compute_objective: {e}")
        return 1e9, {}
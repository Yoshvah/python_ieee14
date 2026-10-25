from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from solver import PowerFlowSolver

# Importations de tes modules de calcul (vérifie bien les noms de dossiers)
from calculations.ybus import calculate_ybus
from calculations.loadflow import newton_raphson
from calculations.power_flows import calculate_branch_flows
from calculations.dynamics import simulate_swing
from calculations.disturbances import three_phase_fault_ybus
from calculations.facts import add_svc

app = FastAPI(title="SPARK-Flow Solver API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class NetworkData(BaseModel):
    buses: list[list[float]]
    branches: list[list[float]]

class StabilityRequest(NetworkData):
    fault_bus: int
    fault_clear_time: float = 0.2
    duration: float = 5.0

@app.post("/api/loadflow")
def run_loadflow(data: NetworkData):
    try:
        # 1. CALCUL DE YBUS (Étape 1)
        ids, ybus = calculate_ybus(data.buses, data.branches, base_mva=100.0)
        
        # 2. LOAD FLOW (Étape 2)
        solver = PowerFlowSolver(100.0, data.buses, data.branches)
        Vm, Va, P, Q, history = solver.solve()
        
        # 3. FLUX DE BRANCHES (Étape 3)
        # Attention : Va du solver est en degrés, calculate_branch_flows attend des radians
        Va_rad = np.radians(Va)
        branch_flows = calculate_branch_flows(data.branches, ids, Vm, Va_rad, 100.0)
        
        flows_formatted = []
        for f in branch_flows:
            flows_formatted.append({
                "from": f["from"],
                "to": f["to"],
                "p_from_MW": round(float(np.real(f["s_from_mva"])), 2),
                "q_from_MVAr": round(float(np.imag(f["s_from_mva"])), 2),
                "p_to_MW": round(float(np.real(f["s_to_mva"])), 2),
                "q_to_MVAr": round(float(np.imag(f["s_to_mva"])), 2),
                "loss_MW": round(float(np.real(f["loss_mva"])), 2),
            })

        results = []
        for i in range(len(Vm)):
            results.append({
                "bus": i + 1,
                "V": round(float(Vm[i]), 4),
                "theta": round(float(Va[i]), 2),
                "P_MW": round(float(P[i]), 2),
                "Q_MVAr": round(float(Q[i]), 2)
            })
        
        # On renvoie TOUT au frontend
        return {
            "status": "Converged",
            "results": results,
            "history": history,         # Le détail des itérations
            "ybus": {                   # La matrice Ybus
                "ids": ids,
                "real": np.round(ybus.real, 4).tolist(),
                "imag": np.round(ybus.imag, 4).tolist(),
            },
            "branch_flows": flows_formatted # Le détail des pertes par ligne
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stability")
def run_stability(data: StabilityRequest):
    try:
        # 1. Load Flow initial pour l'état d'équilibre
        solver = PowerFlowSolver(100.0, data.buses, data.branches)
        Vm, Va, P, Q, _ = solver.solve()
        
        gen_buses = [b for b in data.buses if int(b[1]) in (2, 3)]
        gen_ids = [int(b[0]) for b in gen_buses]
        H = [5.0 for _ in gen_ids]
        D = [0.1 for _ in gen_ids]
        delta0 = [np.radians(Va[gen_id - 1]) for gen_id in gen_ids]
        speed0 = [0.0 for _ in gen_ids]
        Pm = [P[gen_id - 1] / 100.0 for gen_id in gen_ids]

        def Pe(t, delta, speed):
            if t < data.fault_clear_time:
                return [pm * 0.1 for pm in Pm]
            return Pm
        
        res = simulate_swing(
            delta0=delta0, speed0=speed0, inertia=H, damping=D,
            mechanical_power=Pm, electrical_power=Pe,
            duration=data.duration, dt=0.01
        )
        
        return {
            "time": res['time'].tolist(),
            "generators": [{"bus": gid, "delta": np.rad2deg(res['delta'][:, i]).tolist()} for i, gid in enumerate(gen_ids)],
            "history": [{"time": round(t, 2), "angles": [round(np.rad2deg(d), 2) for d in deltas]} for t, deltas in zip(res['time'][::50], res['delta'][::50])] # Historique échantillonné toutes les 0.5s
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
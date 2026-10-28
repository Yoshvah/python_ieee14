from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from solver import PowerFlowSolver

# Importations de tes modules de calcul
from calculations.ybus import calculate_ybus
from calculations.power_flows import calculate_branch_flows
from calculations.dynamics import simulate_swing
from calculations.facts import add_svc

app = FastAPI(title="SPARK-Flow Solver API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================================
# MODÈLES DE DONNÉES
# ============================================================
class FactsConfig(BaseModel):
    bus_id: int
    type: str          # "SVC", "TCSC", "UPFC"
    param1: float      # SVC: B (pu) | TCSC: % Compensation | UPFC: P (MW)
    param2: float | None = None # UPFC: Q (MVAr)

class SolarConfig(BaseModel):
    bus_id: int
    active_power: float  # en MW

class NetworkData(BaseModel):
    buses: list[list[float]]
    branches: list[list[float]]
    facts: list[FactsConfig] = []     
    solars: list[SolarConfig] = [] 

class StabilityRequest(NetworkData):
    fault_bus: int
    fault_clear_time: float = 0.2
    duration: float = 5.0

# ============================================================
# ENDPOINT : LOAD FLOW
# ============================================================
@app.post("/api/loadflow")
def run_loadflow(data: NetworkData):
    try:
        # 1. Préparation des branches (Modification pour le TCSC)
        modified_branches = [list(b) for b in data.branches]
        
        for fact in data.facts:
            if fact.type == "TCSC":
                # On cherche la première branche connectée à ce bus pour modifier son impédance
                for b in modified_branches:
                    if int(b[0]) == fact.bus_id or int(b[1]) == fact.bus_id:
                        # param1 est le pourcentage de compensation (ex: 50 pour réduire X de 50%)
                        x_initial = float(b[3])
                        facteur = max(0.01, 1.0 - (fact.param1 / 100.0))
                        b[3] = x_initial * facteur
                        break # On ne modifie qu'une seule branche par TCSC

        # 2. Calcul de Ybus avec les branches (potentiellement) modifiées
        ids, ybus = calculate_ybus(data.buses, modified_branches, base_mva=100.0)
        
        # 3. Application des autres dispositifs
        for fact in data.facts:
            if fact.type in ["SVC", "STATCOM"]:
                if fact.bus_id in ids:
                    bus_idx = ids.index(fact.bus_id)
                    # param1 = Susceptance B
                    ybus = add_svc(ybus, bus_idx, fact.param1)
            
            elif fact.type == "UPFC":
                if fact.bus_id in ids:
                    bus_idx = ids.index(fact.bus_id)
                    # Un UPFC injecte de la puissance active (P) et réactive (Q)
                    data.buses[bus_idx][2] += (fact.param1 / 100.0)
                    if fact.param2 is not None:
                        data.buses[bus_idx][3] += (fact.param2 / 100.0)

        # 4. Application des Panneaux Solaires
        for solar in data.solars:
            if solar.bus_id in ids:
                bus_idx = ids.index(solar.bus_id)
                # Le solaire injecte de la puissance active pure
                data.buses[bus_idx][2] += (solar.active_power / 100.0)

        # 5. Load Flow
        solver = PowerFlowSolver(100.0, data.buses, modified_branches)
        solver.Ybus = ybus 
        
        try:
            Vm, Va, P, Q, history = solver.solve()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Le Load Flow n'a pas convergé (réseau instable) : {str(e)}")
        
        # 6. Calcul des flux de branches (avec les branches modifiées)
        Va_rad = np.radians(Va)
        branch_flows = calculate_branch_flows(modified_branches, ids, Vm, Va_rad, 100.0)
        
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
        
        return {
            "status": "Converged",
            "results": results,
            "history": history,
            "ybus": {
                "ids": ids,
                "real": np.round(ybus.real, 4).tolist(),
                "imag": np.round(ybus.imag, 4).tolist(),
            },
            "branch_flows": flows_formatted,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# ENDPOINT : STABILITÉ
# ============================================================
@app.post("/api/stability")
def run_stability(data: StabilityRequest):
    try:
        solver = PowerFlowSolver(100.0, data.buses, data.branches)
        
        # Application des SVC pour l'état initial
        if data.facts:
            try:
                ids, ybus = calculate_ybus(data.buses, data.branches)
                for fact in data.facts:
                    if fact.type in ["SVC", "STATCOM"] and fact.bus_id in ids:
                        bus_idx = ids.index(fact.bus_id)
                        ybus = add_svc(ybus, bus_idx, fact.param1)
                solver.Ybus = ybus
            except: pass
            
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
            "history": [{"time": round(t, 2), "angles": [round(np.rad2deg(d), 2) for d in deltas]} for t, deltas in zip(res['time'][::50], res['delta'][::50])]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
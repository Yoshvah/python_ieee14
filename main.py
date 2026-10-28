from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from solver import PowerFlowSolver

from calculations.ybus import calculate_ybus
from calculations.power_flows import calculate_branch_flows
from calculations.dynamics import simulate_swing
from calculations.facts import add_svc, add_statcom, add_tcsc, add_upfc

app = FastAPI(title="SPARK-Flow Solver API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================================
# MODÈLES DE DONNÉES
# ============================================================
class FactsConfig(BaseModel):
    bus_id: int | None = None       # Optionnel (SVC, STATCOM, UPFC)
    branch_index: int | None = None # Optionnel (TCSC)
    type: str          
    param1: float      
    param2: float | None = None

class SolarConfig(BaseModel):
    bus_id: int
    active_power: float

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
        modified_branches = [list(b) for b in data.branches]
        
        # 1. TCSC (modification de X sur la ligne ciblée)
        for fact in data.facts:
            if fact.type == "TCSC" and fact.branch_index is not None:
                idx = fact.branch_index
                if 0 <= idx < len(modified_branches):
                    modified_branches[idx] = add_tcsc(modified_branches[idx], fact.param1)

        # 2. Calcul de la matrice Ybus de base
        ids, ybus = calculate_ybus(data.buses, modified_branches, base_mva=100.0)
        
        # 3. Application des dispositifs (SVC, STATCOM, UPFC)
        for fact in data.facts:
            if fact.bus_id is not None and fact.bus_id in ids:
                bus_idx = ids.index(fact.bus_id)
                
                if fact.type == "SVC":
                    ybus = add_svc(ybus, bus_idx, fact.param1)
                
                elif fact.type == "STATCOM":
                    q_inj = add_statcom(fact.param1 / 100.0)
                    data.buses[bus_idx][3] += q_inj
                    
                elif fact.type == "UPFC":
                    add_upfc(ybus, bus_idx, fact.param1 / 100.0, (fact.param2 or 0) / 100.0, 1.0)
                    data.buses[bus_idx][2] += (fact.param1 / 100.0)
                    if fact.param2 is not None:
                        data.buses[bus_idx][3] += (fact.param2 / 100.0)

        # 4. Panneaux Solaires
        for solar in data.solars:
            if solar.bus_id in ids:
                bus_idx = ids.index(solar.bus_id)
                data.buses[bus_idx][2] += (solar.active_power / 100.0)

        # 5. Load Flow
        solver = PowerFlowSolver(100.0, data.buses, modified_branches)
        solver.Ybus = ybus 
        
        try:
            Vm, Va, P, Q, history = solver.solve()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Le Load Flow n'a pas convergé : {str(e)}")
        
        # 6. Flux de branches
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

        # 7. Formatage des résultats
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
        
        if data.facts:
            try:
                ids, ybus = calculate_ybus(data.buses, data.branches)
                for fact in data.facts:
                    if fact.type == "SVC" and fact.bus_id in ids:
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
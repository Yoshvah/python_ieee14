from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from data import BUS, BRANCH, GEN, BASE_MVA

app = FastAPI(title="GridFACTS Studio API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

class Network(BaseModel):
    buses: list[list[float]]
    branches: list[list[float]]
    generators: list[list[float]]

class Addition(BaseModel):
    a: float
    b: float

@app.post("/api/add")
def add(payload: Addition):
    return {"c": payload.a + payload.b, "resultat": payload.c}

@app.get("/api/network")
def network():
    return {"baseMVA":BASE_MVA, "buses":[list(r) for r in BUS], "branches":[list(r) for r in BRANCH], "generators":[list(r) for r in GEN]}

@app.post("/api/ybus")
def ybus(network: Network):
    buses, branches = network.buses, network.branches
    ids = [int(row[0]) for row in buses]
    if len(ids) != len(set(ids)): raise HTTPException(422, "Duplicate bus IDs")
    index = {bus:i for i,bus in enumerate(ids)}
    y = np.zeros((len(ids),len(ids)),dtype=complex)
    for row in branches:
        if len(row)<6: raise HTTPException(422,"Each branch needs six fields")
        f,t,r,x,b,tap=row[:6]
        f,t=int(f),int(t)
        if f not in index or t not in index: raise HTTPException(422,f"Unknown branch bus: {f}-{t}")
        if r==0 and x==0: raise HTTPException(422,f"Zero impedance on branch {f}-{t}")
        a=1.0 if tap==0 else tap
        if a<=0: raise HTTPException(422,"Tap ratio must be positive")
        adm=1/complex(r,x); charging=1j*b/2
        i,j=index[f],index[t]
        y[i,i]+=(adm+charging)/(a*a)
        y[j,j]+=adm+charging
        y[i,j]-=adm/a
        y[j,i]-=adm/a
    for row in buses:
        if len(row)>=7: y[index[int(row[0])],index[int(row[0])]]+=1j*row[6]/BASE_MVA
    return {"busIds":ids,"real":np.round(y.real,7).tolist(),"imag":np.round(y.imag,7).tolist(),"baseMVA":BASE_MVA}

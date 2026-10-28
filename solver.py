import numpy as np

class PowerFlowSolver:
    def __init__(self, base_mva, buses, branches):
        self.base_mva = base_mva
        self.buses = buses
        self.branches = branches
        self.nb_bus = len(buses)
        self.bus_map = {int(b[0]): i for i, b in enumerate(buses)}
        self.history = []
        self.Ybus = self._build_ybus()
        self.P_spec, self.Q_spec = self._get_specifications()

    def _build_ybus(self):
        """Construit la matrice d'admittance Ybus complète."""
        Ybus = np.zeros((self.nb_bus, self.nb_bus), dtype=complex)
        
        # 1. Ajout des lignes et transformateurs
        for br in self.branches:
            f, t = int(br[0]), int(br[1])
            r, x = float(br[2]), float(br[3])
            b = float(br[4]) if len(br) > 4 else 0.0
            tap = float(br[5]) if len(br) > 5 else 0.0
            
            if r == 0 and x == 0:
                continue
                
            y_series = 1 / complex(r, x)
            y_shunt_line = 1j * b / 2.0
            a = 1.0 if tap == 0 else tap
            
            i, j = self.bus_map[f], self.bus_map[t]
            
            # Modèle Pi
            Ybus[i, i] += (y_series + y_shunt_line) / (a**2)
            Ybus[j, j] += y_series + y_shunt_line
            Ybus[i, j] -= y_series / a
            Ybus[j, i] -= y_series / a

        # 2. Ajout des shunts des bus (Bsh)
        # Correction : On ajoute +1j * Bsh pour correspondre au standard
        for i, b in enumerate(self.buses):
            if len(b) >= 7 and b[6] != 0:
                Ybus[i, i] += 1j * float(b[6])
                
        return Ybus

    def _get_specifications(self):
        """Prépare les vecteurs P_spec et Q_spec."""
        P_spec = np.zeros(self.nb_bus)
        Q_spec = np.zeros(self.nb_bus)
        for i, b in enumerate(self.buses):
            type_bus = int(b[1])
            if type_bus == 1:   # PQ
                P_spec[i] = b[2]
                Q_spec[i] = b[3]
            elif type_bus == 2: # PV
                P_spec[i] = b[2]
                Q_spec[i] = 0.0
            else:               # Slack
                P_spec[i] = 0.0
                Q_spec[i] = 0.0
        return P_spec, Q_spec

    def solve(self, tol=1e-8, max_iter=20):
        """Exécute Newton-Raphson et retourne (V, theta, P, Q, history)."""
        # Réinitialisation de l'historique pour chaque nouveau calcul
        self.history = []

        V = np.ones(self.nb_bus, dtype=complex)
        for i, b in enumerate(self.buses):
            V[i] = b[4] * np.exp(1j * np.radians(b[5]))

        types = [int(b[1]) for b in self.buses]
        pq_idx = [i for i, t in enumerate(types) if t == 1]
        pv_idx = [i for i, t in enumerate(types) if t == 2]
        
        angle_idx = pq_idx + pv_idx
        V_idx = pq_idx

        self.history.append({
            "iteration": 0,
            "description": "État initial",
            "V": np.abs(V).round(4).tolist(),
            "theta": np.degrees(np.angle(V)).round(2).tolist(),
            "mismatch_max": None
        })

        converged = False
        for iteration in range(1, max_iter + 1):
            V_mag = np.abs(V)
            V_ang = np.angle(V)
            P_calc = np.zeros(self.nb_bus)
            Q_calc = np.zeros(self.nb_bus)
            
            for i in range(self.nb_bus):
                for j in range(self.nb_bus):
                    P_calc[i] += V_mag[i] * V_mag[j] * (self.Ybus[i, j].real * np.cos(V_ang[i] - V_ang[j]) + self.Ybus[i, j].imag * np.sin(V_ang[i] - V_ang[j]))
                    Q_calc[i] += V_mag[i] * V_mag[j] * (self.Ybus[i, j].real * np.sin(V_ang[i] - V_ang[j]) - self.Ybus[i, j].imag * np.cos(V_ang[i] - V_ang[j]))

            dP = self.P_spec - P_calc
            dQ = self.Q_spec - Q_calc
            mismatch = np.concatenate((dP[angle_idx], dQ[V_idx]))
            max_err = float(np.max(np.abs(mismatch))) if len(mismatch) > 0 else 0.0
            
            self.history.append({
                "iteration": iteration,
                "description": f"Itération {iteration}",
                "V": V_mag.round(4).tolist(),
                "theta": np.degrees(V_ang).round(2).tolist(),
                "P_calc": (P_calc * self.base_mva).round(2).tolist(),
                "Q_calc": (Q_calc * self.base_mva).round(2).tolist(),
                "mismatch_max": round(max_err, 8)
            })

            if max_err < tol:
                self.history[-1]["description"] = f"✅ Convergence atteinte à l'itération {iteration}"
                converged = True
                break

            # Jacobien
            J11 = np.zeros((len(angle_idx), len(angle_idx)))
            J12 = np.zeros((len(angle_idx), len(V_idx)))
            J21 = np.zeros((len(V_idx), len(angle_idx)))
            J22 = np.zeros((len(V_idx), len(V_idx)))

            for i_idx, i in enumerate(angle_idx):
                for j_idx, j in enumerate(angle_idx):
                    if i == j:
                        J11[i_idx, j_idx] = -Q_calc[i] - (V_mag[i]**2) * self.Ybus[i, i].imag
                    else:
                        J11[i_idx, j_idx] = V_mag[i] * V_mag[j] * (self.Ybus[i, j].real * np.sin(V_ang[i] - V_ang[j]) - self.Ybus[i, j].imag * np.cos(V_ang[i] - V_ang[j]))
                for j_idx, j in enumerate(V_idx):
                    if i == j:
                        J12[i_idx, j_idx] = P_calc[i] / V_mag[i] + V_mag[i] * self.Ybus[i, i].real
                    else:
                        J12[i_idx, j_idx] = V_mag[i] * (self.Ybus[i, j].real * np.cos(V_ang[i] - V_ang[j]) + self.Ybus[i, j].imag * np.sin(V_ang[i] - V_ang[j]))

            for i_idx, i in enumerate(V_idx):
                for j_idx, j in enumerate(angle_idx):
                    if i == j:
                        J21[i_idx, j_idx] = P_calc[i] - (V_mag[i]**2) * self.Ybus[i, i].real
                    else:
                        J21[i_idx, j_idx] = -V_mag[i] * V_mag[j] * (self.Ybus[i, j].real * np.cos(V_ang[i] - V_ang[j]) + self.Ybus[i, j].imag * np.sin(V_ang[i] - V_ang[j]))
                for j_idx, j in enumerate(V_idx):
                    if i == j:
                        J22[i_idx, j_idx] = Q_calc[i] / V_mag[i] - V_mag[i] * self.Ybus[i, i].imag
                    else:
                        J22[i_idx, j_idx] = V_mag[i] * (self.Ybus[i, j].real * np.sin(V_ang[i] - V_ang[j]) - self.Ybus[i, j].imag * np.cos(V_ang[i] - V_ang[j]))

            J = np.block([[J11, J12], [J21, J22]])
            dx = np.linalg.solve(J, mismatch)
            
            d_theta = dx[:len(angle_idx)]
            d_V = dx[len(angle_idx):]
            
            for i_idx, i in enumerate(angle_idx):
                V_ang[i] += d_theta[i_idx]
            for i_idx, i in enumerate(V_idx):
                V_mag[i] += d_V[i_idx]
            
            V = V_mag * np.exp(1j * V_ang)

        if not converged:
            raise RuntimeError(f"Le Load Flow n'a pas convergé après {max_iter} itérations.")

        # Recalcul final
        V_mag = np.abs(V)
        V_ang = np.angle(V)
        P_final = np.zeros(self.nb_bus)
        Q_final = np.zeros(self.nb_bus)
        for i in range(self.nb_bus):
            for j in range(self.nb_bus):
                P_final[i] += V_mag[i] * V_mag[j] * (self.Ybus[i, j].real * np.cos(V_ang[i] - V_ang[j]) + self.Ybus[i, j].imag * np.sin(V_ang[i] - V_ang[j]))
                Q_final[i] += V_mag[i] * V_mag[j] * (self.Ybus[i, j].real * np.sin(V_ang[i] - V_ang[j]) - self.Ybus[i, j].imag * np.cos(V_ang[i] - V_ang[j]))

        return V_mag, np.degrees(V_ang), P_final * self.base_mva, Q_final * self.base_mva, self.history
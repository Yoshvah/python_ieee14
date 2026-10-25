import numpy as np

class PowerFlowSolver:
    def __init__(self, base_mva, buses, branches):
        self.base_mva = base_mva
        self.buses = buses
        self.branches = branches
        self.nb_bus = len(buses)
        self.bus_map = {int(b[0]): i for i, b in enumerate(buses)}
        self.history = []  # Stocke l'historique pour l'affichage des étapes
        self.Ybus = self._build_ybus()
        self.P_spec, self.Q_spec = self._get_specifications()

    def _build_ybus(self):
        """Construit la matrice d'admittance Y-bus à partir des branches."""
        Ybus = np.zeros((self.nb_bus, self.nb_bus), dtype=complex)
        for br in self.branches:
            f, t, r, x = int(br[0]), int(br[1]), br[2], br[3]
            if r == 0 and x == 0:
                continue
            y = 1 / complex(r, x)
            i, j = self.bus_map[f], self.bus_map[t]
            Ybus[i, i] += y
            Ybus[j, j] += y
            Ybus[i, j] -= y
            Ybus[j, i] -= y
        return Ybus

    def _get_specifications(self):
        """Extrait les puissances spécifiées (P et Q) selon le type de bus."""
        P_spec = np.zeros(self.nb_bus)
        Q_spec = np.zeros(self.nb_bus)
        for i, b in enumerate(self.buses):
            type_bus = int(b[1])
            if type_bus == 1:   # PQ (charge)
                P_spec[i] = b[2]
                Q_spec[i] = b[3]
            elif type_bus == 2: # PV (générateur)
                P_spec[i] = b[2]
                Q_spec[i] = 0.0
            else:               # Slack
                P_spec[i] = 0.0
                Q_spec[i] = 0.0
        return P_spec, Q_spec

    def solve(self, tol=1e-6, max_iter=20):
        """Exécute Newton-Raphson et retourne (V, theta, P, Q, history)."""
        V = np.ones(self.nb_bus, dtype=complex)
        for i, b in enumerate(self.buses):
            V[i] = b[4] * np.exp(1j * np.radians(b[5]))

        types = [int(b[1]) for b in self.buses]
        pq_idx = [i for i, t in enumerate(types) if t == 1]
        pv_idx = [i for i, t in enumerate(types) if t == 2]
        
        angle_idx = pq_idx + pv_idx
        V_idx = pq_idx

        # --- ÉTAPE 0 : État initial ---
        self.history.append({
            "iteration": 0,
            "description": "État initial (estimations)",
            "V": np.abs(V).round(4).tolist(),
            "theta": np.degrees(np.angle(V)).round(2).tolist(),
            "mismatch_max": None
        })

        for iteration in range(1, max_iter + 1):
            V_mag = np.abs(V)
            V_ang = np.angle(V)
            P_calc = np.zeros(self.nb_bus)
            Q_calc = np.zeros(self.nb_bus)
            
            # Calcul des puissances injectées
            for i in range(self.nb_bus):
                for j in range(self.nb_bus):
                    P_calc[i] += V_mag[i] * V_mag[j] * (self.Ybus[i, j].real * np.cos(V_ang[i] - V_ang[j]) + self.Ybus[i, j].imag * np.sin(V_ang[i] - V_ang[j]))
                    Q_calc[i] += V_mag[i] * V_mag[j] * (self.Ybus[i, j].real * np.sin(V_ang[i] - V_ang[j]) - self.Ybus[i, j].imag * np.cos(V_ang[i] - V_ang[j]))

            # Calcul des erreurs (mismatch)
            dP = self.P_spec - P_calc
            dQ = self.Q_spec - Q_calc
            mismatch = np.concatenate((dP[angle_idx], dQ[V_idx]))
            max_err = float(np.max(np.abs(mismatch))) if len(mismatch) > 0 else 0.0
            
            # Enregistrement de l'itération (avec V et theta à chaque étape)
            self.history.append({
                "iteration": iteration,
                "description": f"Itération {iteration} - Calcul des puissances et Jacobien",
                "V": V_mag.round(4).tolist(),
                "theta": np.degrees(V_ang).round(2).tolist(),
                "P_calc": (P_calc * self.base_mva).round(2).tolist(),
                "Q_calc": (Q_calc * self.base_mva).round(2).tolist(),
                "mismatch_max": round(max_err, 6)
            })

            # Critère de convergence
            if max_err < tol:
                self.history[-1]["description"] = f"✅ Convergence atteinte à l'itération {iteration}"
                break

            # --- Construction du Jacobien ---
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

            # Résolution du système linéaire J * dx = mismatch
            J = np.block([[J11, J12], [J21, J22]])
            dx = np.linalg.solve(J, mismatch)
            
            d_theta = dx[:len(angle_idx)]
            d_V = dx[len(angle_idx):]
            
            # Mise à jour des variables
            for i_idx, i in enumerate(angle_idx):
                V_ang[i] += d_theta[i_idx]
            for i_idx, i in enumerate(V_idx):
                V_mag[i] += d_V[i_idx]
            
            V = V_mag * np.exp(1j * V_ang)

        # Calcul final des puissances (après convergence)
        P_final = np.zeros(self.nb_bus)
        Q_final = np.zeros(self.nb_bus)
        for i in range(self.nb_bus):
            for j in range(self.nb_bus):
                P_final[i] += V_mag[i] * V_mag[j] * (self.Ybus[i, j].real * np.cos(V_ang[i] - V_ang[j]) + self.Ybus[i, j].imag * np.sin(V_ang[i] - V_ang[j]))
                Q_final[i] += V_mag[i] * V_mag[j] * (self.Ybus[i, j].real * np.sin(V_ang[i] - V_ang[j]) - self.Ybus[i, j].imag * np.cos(V_ang[i] - V_ang[j]))

        return V_mag, np.degrees(V_ang), P_final * self.base_mva, Q_final * self.base_mva, self.history
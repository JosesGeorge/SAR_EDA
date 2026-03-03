import numpy as np

class VitalSignSimulator:
    def generate(self):
        hr = int(np.random.uniform(60, 100))   # bpm
        rr = int(np.random.uniform(12, 20))    # breaths/min

        t = np.linspace(0, 6, 600)             # 6 seconds

        # ECG — PQRST approximation
        ecg = np.zeros_like(t)
        period = 60.0 / hr
        for i, ti in enumerate(t):
            phase = (ti % period) / period
            if   phase < 0.02:  ecg[i] = -0.10
            elif phase < 0.04:  ecg[i] =  0.15   # P
            elif phase < 0.06:  ecg[i] = -0.05
            elif phase < 0.09:  ecg[i] = -0.30   # Q
            elif phase < 0.12:  ecg[i] =  1.00   # R
            elif phase < 0.14:  ecg[i] = -0.30   # S
            elif phase < 0.20:  ecg[i] =  0.00
            elif phase < 0.36:  ecg[i] =  0.20   # T
        ecg += np.random.randn(len(t)) * 0.03

        # Respiration — smooth sine
        resp = 0.6 * np.sin(2 * np.pi * (rr / 60) * t)
        resp += np.random.randn(len(t)) * 0.04

        return {
            "heart_rate": hr,
            "resp_rate":  rr,
            "time":       t.tolist(),
            "ecg":        ecg.tolist(),
            "resp":       resp.tolist(),
        }
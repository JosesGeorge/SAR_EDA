import numpy as np

class SonarSimulator:
    def __init__(self, n_points=120, noise_level=0.3, depth_range=(2.0, 15.0)):
        self.n_points    = n_points
        self.noise_level = noise_level
        self.depth_range = depth_range

    def generate(self, object_type="random"):
        if object_type == "random":
            object_type = np.random.choice(["human", "debris"])
        
        if object_type == "human":
            return self._generate_human()
        return self._generate_debris()

    def _generate_human(self):
        n   = self.n_points
        nl  = self.noise_level
        cx  = np.random.uniform(-4, 4)
        cy  = np.random.uniform(-4, 4)
        dep = np.random.uniform(*self.depth_range)

        parts = [
            dict(ox=0,     oy=0,    rx=0.20, ry=0.50, rz=0.80, frac=0.35),  # torso
            dict(ox=0,     oy=0.65, rx=0.10, ry=0.10, rz=0.12, frac=0.12),  # head
            dict(ox=0.30,  oy=0.10, rx=0.07, ry=0.45, rz=0.07, frac=0.12),  # R arm
            dict(ox=-0.30, oy=0.10, rx=0.07, ry=0.45, rz=0.07, frac=0.12),  # L arm
            dict(ox=0.12,  oy=-0.6, rx=0.09, ry=0.55, rz=0.09, frac=0.145), # R leg
            dict(ox=-0.12, oy=-0.6, rx=0.09, ry=0.55, rz=0.09, frac=0.145), # L leg
        ]

        xs, ys, zs, intens, tds, dops = [], [], [], [], [], []
        for p in parts:
            k = int(n * p["frac"])
            x = cx + p["ox"] + np.random.uniform(-p["rx"], p["rx"], k) + np.random.randn(k)*nl*0.05
            y = cy + p["oy"] + np.random.uniform(-p["ry"], p["ry"], k) + np.random.randn(k)*nl*0.05
            z = dep +           np.random.uniform(-p["rz"], p["rz"], k) * 0.5
            td = (2 * z) / 1500.0
            iv = np.clip(0.7 + np.random.rand(k)*0.3 - nl*np.random.rand(k)*0.3, 0, 1)
            dp = np.random.randn(k)*1.2 + np.sin(np.arange(k)*0.3)*0.5
            xs.extend(x); ys.extend(y); zs.extend(z)
            intens.extend(iv); tds.extend(td); dops.extend(dp)

        return {"x": xs, "y": ys, "z": zs, "intensity": intens,
                "time_delay": tds, "doppler_shift": dops, "true_label": "human"}

    def _generate_debris(self):
        n   = self.n_points
        nl  = self.noise_level
        cx  = np.random.uniform(-4, 4)
        cy  = np.random.uniform(-4, 4)
        dep = np.random.uniform(*self.depth_range)
        shape = np.random.choice(["blob", "box", "scatter"])

        xs, ys, zs = [], [], []
        if shape == "blob":
            r     = np.random.uniform(0.3, 1.2, n)
            theta = np.random.uniform(0, 2*np.pi, n)
            xs = cx + r*np.cos(theta)*(0.5+np.random.rand(n))
            ys = cy + r*np.sin(theta)*(0.3+np.random.rand(n))
            zs = dep + np.random.uniform(-0.8, 0.8, n)
        elif shape == "box":
            xs = cx + np.random.uniform(-1.5, 1.5, n)
            ys = cy + np.random.uniform(-0.75, 0.75, n)
            zs = dep + np.random.uniform(-0.6, 0.6, n)
        else:
            xs = cx + np.random.uniform(-3, 3, n) * np.random.choice([-1,1], n)
            ys = cy + np.random.uniform(-3, 3, n) * np.random.choice([-1,1], n)
            zs = dep + np.random.uniform(-1.2, 1.2, n)

        xs += np.random.randn(n)*nl*0.2
        ys += np.random.randn(n)*nl*0.2
        td  = (2 * np.array(zs)) / 1500.0
        iv  = np.clip(0.3 + np.random.rand(n)*0.5 - nl*np.random.rand(n)*0.2, 0, 1)
        dp  = np.random.randn(n)*0.3

        return {"x": list(xs), "y": list(ys), "z": list(zs), "intensity": list(iv),
                "time_delay": list(td), "doppler_shift": list(dp), "true_label": "debris"}
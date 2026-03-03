import numpy as np

class FeatureEngineer:
    def extract(self, sonar_data, processed=None):
        x  = np.array(sonar_data["x"])
        y  = np.array(sonar_data["y"])
        z  = np.array(sonar_data["z"])
        iv = np.array(sonar_data["intensity"])
        dp = np.array(sonar_data["doppler_shift"])
        n  = len(x)

        # Use np.ptp instead of array.ptp() (NumPy 2.x compatible)
        xr = np.ptp(x)
        yr = np.ptp(y)
        zr = np.ptp(z)

        height  = max(xr, yr)
        width   = min(xr, yr)

        volume  = max(xr * yr * zr, 1e-6)
        density = n / volume

        intensity_mean   = float(np.mean(iv))
        doppler_variance = float(np.var(dp))

        # Symmetry: compare point counts left/right of centroid
        cx = np.mean(x)
        left  = np.sum(x <= cx)
        right = np.sum(x > cx)
        sym = 1 - abs(left - right) / max(n, 1)

        # Movement score (normalized doppler variance)
        movement_score = min(1.0, doppler_variance / 2.0)

        return {
            "height":           round(float(height), 3),
            "width":            round(float(width), 3),
            "density":          round(min(float(density), 999), 2),
            "intensity_mean":   round(intensity_mean, 4),
            "doppler_variance": round(doppler_variance, 4),
            "symmetry_score":   round(float(sym), 4),
            "movement_score":   round(float(movement_score), 4),
        }

    def to_vector(self, feats):
        return [
            feats["height"],
            feats["width"],
            feats["density"],
            feats["intensity_mean"],
            feats["doppler_variance"],
            feats["symmetry_score"],
        ]
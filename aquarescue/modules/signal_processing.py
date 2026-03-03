import numpy as np
from scipy.signal import butter, filtfilt

class SignalProcessor:
    def process(self, sonar_data):
        intensity = np.array(sonar_data["intensity"])
        n = len(intensity)

        # Bandpass filter (5-tap Butterworth approx)
        b, a = butter(2, [0.05, 0.45], btype='bandpass')
        try:
            filtered = filtfilt(b, a, intensity)
        except Exception:
            window = min(5, n)
            filtered = np.convolve(intensity, np.ones(window)/window, mode='same')

        # Normalize [0, 1]
        mn, mx = filtered.min(), filtered.max()
        normalized = (filtered - mn) / (mx - mn + 1e-8)

        # Time delay → distance
        distances = [(1500 * td) / 2 for td in sonar_data["time_delay"]]

        return {
            "intensity_filtered":   filtered.tolist(),
            "intensity_normalized": normalized.tolist(),
            "distances": distances,
        }
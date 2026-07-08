"""Prosodic features for cat meows: duration, mean F0, and intonation slope.

Grounded in Schötz et al. 2023 (Applied Animal Behaviour Science, 70 cats /
969 meows): context significantly affects meow *duration* and *mean F0*, but
NOT F0 range, and the intonation contour is context-specific (e.g. carrier =
falling, cuddle/door = level, food/play/greeting = rise+fall). So we expose
mean F0, duration, and an F0 slope (rising vs falling) — deliberately not F0
range, which the study found non-discriminative.

These features are concatenated onto the frozen AST embedding before the
logistic head; see actions/extract_prosody.py and actions/train_ast_head.py.
"""
import numpy as np
import librosa

# Domestic cat meows sit roughly in this F0 band; pyin needs explicit bounds.
# Kept wide to tolerate kittens/large cats and octave errors at the edges.
F0_MIN = 100.0
F0_MAX = 1500.0

# Order matters: it must match between training and inference.
FEATURE_NAMES = ["duration_s", "mean_f0_hz", "f0_slope_oct_per_s"]


def prosody_features(waveform, sample_rate):
    """Return a fixed-length prosody vector for one meow clip.

    [duration in seconds, mean F0 in Hz over voiced frames,
     intonation slope in octaves/second (positive = rising)].
    Unvoiced or too-short clips fall back to neutral zeros for F0/slope.
    """
    wav = np.asarray(waveform, dtype="float32")
    duration = len(wav) / sample_rate

    f0, _, _ = librosa.pyin(wav, sr=sample_rate, fmin=F0_MIN, fmax=F0_MAX)
    times = librosa.times_like(f0, sr=sample_rate)
    voiced = np.isfinite(f0)

    if voiced.sum() < 2:
        return np.array([duration, 0.0, 0.0], dtype="float32")

    mean_f0 = float(np.mean(f0[voiced]))

    # Intonation slope: fit log2(F0) against time over voiced frames, giving a
    # rate in octaves/second that is invariant to sample rate and F0 register.
    t = times[voiced]
    log_f0 = np.log2(f0[voiced])
    slope = float(np.polyfit(t - t.mean(), log_f0, 1)[0])

    return np.array([duration, mean_f0, slope], dtype="float32")

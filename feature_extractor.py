import numpy as np
import librosa
from singer_dp import get_singer_feature_database


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y)

    if len(y) == 0:
        raise ValueError("Audio appears to be empty or silent.")

    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6")
    )

    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) > 0:
        pitch_mean = float(np.mean(valid_f0))
        pitch_min = float(np.min(valid_f0))
        pitch_max = float(np.max(valid_f0))
    else:
        pitch_mean = 0.0
        pitch_min = 0.0
        pitch_max = 0.0

    feature_vector = np.concatenate([
        np.array([
            rms,
            zcr,
            spectral_centroid,
            spectral_bandwidth,
            pitch_mean,
            pitch_min,
            pitch_max
        ]),
        mfcc_mean
    ])

    return {
        "vector": feature_vector.tolist(),
        "pitch_mean": pitch_mean,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "brightness": spectral_centroid,
        "energy": rms
    }


def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def describe_voice(features):
    pitch = features["pitch_mean"]
    brightness = features["brightness"]
    energy = features["energy"]

    descriptions = []

    if pitch == 0:
        descriptions.append("unclear pitch detection")
    elif pitch > 220:
        descriptions.append("higher-pitched")
    else:
        descriptions.append("lower-pitched")

    if brightness > 2000:
        descriptions.append("bright tone")
    else:
        descriptions.append("warm/dark tone")

    if energy > 0.03:
        descriptions.append("strong vocal energy")
    else:
        descriptions.append("soft vocal energy")

    return ", ".join(descriptions)


def compare_to_singers(user_features):
    singer_db = get_singer_feature_database()
    user_vector = user_features["vector"]

    scored = []

    for singer_name, singer_vector in singer_db.items():
        score = cosine_similarity(user_vector, singer_vector) * 100
        scored.append({
            "singer": singer_name,
            "similarity": round(score, 2)
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "top_matches": scored[:5],
        "voice_description": describe_voice(user_features),
        "pitch_mean": round(user_features["pitch_mean"], 2),
        "pitch_min": round(user_features["pitch_min"], 2),
        "pitch_max": round(user_features["pitch_max"], 2)
    }
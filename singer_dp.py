import os
import numpy as np
import librosa

SINGER_SAMPLE_PATH = "singer_samples"


def extract_feature_vector(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y)

    if len(y) == 0:
        return None

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

    return np.concatenate([
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


def get_singer_feature_database():
    if not os.path.exists(SINGER_SAMPLE_PATH):
        return {
            "IU": [0.02] * 20,
            "Taeyeon": [0.03] * 20,
            "Jungkook": [0.04] * 20,
            "Bruno Mars": [0.05] * 20,
            "Ariana Grande": [0.06] * 20
        }

    database = {}

    for singer_name in os.listdir(SINGER_SAMPLE_PATH):
        singer_folder = os.path.join(SINGER_SAMPLE_PATH, singer_name)

        if not os.path.isdir(singer_folder):
            continue

        vectors = []

        for file_name in os.listdir(singer_folder):
            if file_name.lower().endswith((".wav", ".mp3", ".m4a")):
                file_path = os.path.join(singer_folder, file_name)
                vector = extract_feature_vector(file_path)

                if vector is not None:
                    vectors.append(vector)

        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            database[singer_name] = avg_vector.tolist()

    if not database:
        return {
            "IU": [0.02] * 20,
            "Taeyeon": [0.03] * 20,
            "Jungkook": [0.04] * 20,
            "Bruno Mars": [0.05] * 20,
            "Ariana Grande": [0.06] * 20
        }

    return database
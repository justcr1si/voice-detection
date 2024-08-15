import numpy as np
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pydub import AudioSegment
from dotenv import load_dotenv, find_dotenv
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import librosa
from os import getenv

load_dotenv(find_dotenv())


def extract_mfcc(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    sr = audio_segment.frame_rate
    mfccs = librosa.feature.mfcc(y=samples.astype(float), sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)


def main():
    VOICE_PATH = 'voices/2-651_2024.mp3'

    model = Model.from_pretrained(
        'pyannote/segmentation-3.0', use_auth_token=getenv('TOKEN'))

    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        'min_duration_on': 0.1,
        'min_duration_off': 0.1,
    }

    pipeline.instantiate(HYPER_PARAMETERS)
    speech_regions = pipeline(VOICE_PATH)

    voices = {}
    audio_file = AudioSegment.from_mp3(VOICE_PATH)

    mfccs_list = []
    segments_list = []

    for track in speech_regions.itertracks(yield_label=True):
        segment, *rest = track
        start_time = segment.start
        end_time = segment.end

        voice_segment = audio_file[start_time * 1000:end_time * 1000]
        mfcc_features = extract_mfcc(voice_segment)

        mfccs_list.append(mfcc_features)
        # Сохраняем таймкоды
        segments_list.append((float(f'{start_time:.2f}'),
                              float(f'{end_time:.2f}')))

    if not mfccs_list:
        print("Не обнаружены сегменты речи.")
        return

    print(f"Количество извлеченных MFCC: {len(mfccs_list)}")

    mfccs_scaled = StandardScaler().fit_transform(mfccs_list)

    clustering = DBSCAN(eps=1, min_samples=2).fit(mfccs_scaled)

    unique_labels = set(clustering.labels_)
    print(f"Количество уникальных меток: {len(unique_labels)}")

    for voice_id in unique_labels:
        if voice_id == -1:  # -1 - шум, игнорируем
            continue

        voice_key = f'Голос №{voice_id + 1}'
        if voice_key not in voices:
            voices[voice_key] = []

        indices = np.where(clustering.labels_ == voice_id)[0]
        for index in indices:
            voices[voice_key].append(segments_list[index])

    if voices:
        for key, value in voices.items():
            print(f'{key}: {len(value)} сегментов')
            for segment in value[:3]:
                print(f" - Таймкод: {segment[0]}s - {segment[1]}s")
    else:
        print("Голоса не обнаружены.")

    print(voices)


if __name__ == '__main__':
    main()

from pyannote.audio import Pipeline
from dotenv import load_dotenv, find_dotenv
from os import getenv

load_dotenv(find_dotenv())


def main():
    VOICE_PATH = 'voices/2-651_2024.wav'

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=getenv('TOKEN'))

    diarization = pipeline(VOICE_PATH)

    # Делаем финальный дикт для хранения конечного результата
    result_dict = {}

    for segment, dict_value in diarization._tracks.items():
        speaker_id = list(dict_value.values())[0][-2::]
        if speaker_id[0] == '0':
            speaker_id = speaker_id[1:]

        speaker_id = int(speaker_id)
        phrase = f'Голос №{speaker_id + 1}'
        result_segment = (float(f'{segment.start:.2f}'),
                          float(f'{segment.end:.2f}'))

        if phrase not in result_dict:
            result_dict[phrase] = [result_segment]
        else:
            result_dict[phrase].append(result_segment)

    with open('audio.txt', 'w', encoding='utf-8') as file:
        for speaker in result_dict:
            file.write(f'{speaker} - {result_dict[speaker]}\n')


if __name__ == '__main__':
    main()

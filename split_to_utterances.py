# Loads diarization outputs (.rttms) and splits long audio files to utterance-sized
# wav-files based on the diarization results. Short files are temporarily stored
# to ALICE/tmo_data/short/

from pathlib import Path
from dataclasses import dataclass
from scipy.io import wavfile
import librosa

valid_speakers = ["FEM", "MAL"]


@dataclass
class RTTMLine:
    uri: str
    segment_onset_s: float
    duration_s: float
    speaker: str

    @property
    def segment_offset_s(self):
        return self.segment_onset_s + self.duration_s

    @classmethod
    def from_rttm_line(cls, line: str):
        fields = line.strip().split(" ")
        return cls(
            uri=fields[1],
            segment_onset_s=float(fields[3]),
            duration_s=float(fields[4]),
            speaker=fields[7],
        )


def read_rttm(rttm_path: str | Path):
    """Parse an .rttm file into RTTMLines, one per non-blank row."""
    rttm_path = Path(rttm_path)
    rttm_data = rttm_path.read_text()
    return [RTTMLine.from_rttm_line(line) for line in rttm_data.splitlines() if line]


def split_to_utterances(curdir, rttm_path=None):
    if rttm_path is None:
        rttm_path = curdir + "/output_voice_type_classifier/tmp_data/all.rttm"

    curfile = []
    for segment in read_rttm(rttm_path):
        orig_filename = curdir + "/tmp_data/" + segment.uri + ".wav"
        if curfile != orig_filename:  # read .wav if not read already
            try:
                fs, data = wavfile.read(orig_filename)
            except ValueError:  # Reading failed. Try with librosa.
                data, fs = librosa.load(orig_filename, sr=16000)
            curfile = orig_filename
        onset = segment.segment_onset_s
        offset = segment.segment_offset_s
        if segment.speaker in valid_speakers:
            y = data[max(0, round(onset * fs)) : min(len(data), round(offset * fs))]
            new_filename = (
                curdir
                + "/tmp_data/short/"
                + segment.uri
                + ("_%010.0f" % (onset * 10000))
                + "_"
                + ("%010.0f" % (offset * 10000))
                + ".wav"
            )
            wavfile.write(new_filename, fs, y)

#!/usr/bin/env python3
"""Run the ALICE pipeline: audio in, adult linguistic unit counts out.

Python port of run_ALICE.sh, minus the voice-type-classifier stage. Diarization
is now an *input*: point --rttm at an .rttm produced by whichever diarizer you
use. Everything downstream of that is unchanged.

Stages, in order (each one reads what the previous one wrote, under tmp_data/):

    prepare_data           copy input wavs into tmp_data/
    split_to_utterances    cut adult (FEM/MAL) segments out of them
    SylNet                 syllable count per utterance
    extract_basic_features duration, energy, zero-crossing rate
    paste_columns          -> final_feats.txt [sylnet, duration, energy, zcr]
    regress_ALUCs          linear model -> phoneme/syllable/word counts
    get_final_estimates    aggregate utterances back to per-file totals
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from extract_basic_features import extract_basic_features
from getFinalEstimates import get_final_estimates
from prepare_data import prepare_data
from regress_ALUCs import regress_ALUCs
from split_to_utterances import split_to_utterances

THISDIR = Path(__file__).resolve().parent

# The TF2 entry point: it drives TensorFlow through tensorflow.compat.v1, so the
# TF1-era graph code and the model_1 checkpoint still load. Needs
# TF_USE_LEGACY_KERAS=1, since Keras 3 dropped the tf.compat.v1.layers shim.
SYLNET_SCRIPT = THISDIR / "SylNet" / "run_SylNet_tf2.py"
SYLNET_MODEL = THISDIR / "SylNet_model" / "model_1"
SYLNET_LOG = THISDIR / "sylnet.log"

DEFAULT_RTTM = THISDIR / "output_voice_type_classifier" / "tmp_data" / "all.rttm"


def reset_tmp_data(tmp_data):
    """Recreate the scratch tree. Any previous run's intermediates are lost."""
    shutil.rmtree(tmp_data, ignore_errors=True)
    (tmp_data / "short").mkdir(parents=True)
    (tmp_data / "features").mkdir(parents=True)


def run_sylnet(short_dir, out_file):
    """Estimate syllable counts per utterance.

    Run as a subprocess rather than imported: run_SylNet_tf2.py is a script that
    imports its sibling SylNet_model_tf2, which only resolves when its own
    directory leads sys.path. Its very noisy TF output goes to sylnet.log.
    """
    env = {**os.environ, "TF_USE_LEGACY_KERAS": "1"}
    command = [
        sys.executable,
        str(SYLNET_SCRIPT),
        f"{short_dir}{os.sep}",
        str(out_file),
        str(SYLNET_MODEL),
    ]

    with open(SYLNET_LOG, "w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False
        )

    if result.returncode != 0:
        sys.exit(f"SylNet failed. See {SYLNET_LOG} for more information")

    print("SylNet completed")


def paste_columns(left, right, out):
    """Tab-join two files column-wise, the way coreutils paste would.

    The resulting column order is a contract with regress_ALUCs, so a row count
    mismatch is fatal here -- padding it would silently produce wrong counts.
    """
    left_lines = left.read_text().splitlines()
    right_lines = right.read_text().splitlines()

    if len(left_lines) != len(right_lines):
        sys.exit(
            f"Cannot combine {left.name} ({len(left_lines)} rows) with "
            f"{right.name} ({len(right_lines)} rows): row counts must match."
        )

    out.write_text("".join(f"{a}\t{b}\n" for a, b in zip(left_lines, right_lines)))


def run_alice(data_location, rttm, keep_tmp=False):
    curdir = str(THISDIR)
    tmp_data = THISDIR / "tmp_data"
    features = tmp_data / "features"
    short_dir = tmp_data / "short"
    utterance_counts = features / "ALUCs_out_individual.txt"

    reset_tmp_data(tmp_data)

    # prepare_data globs as <arg>*.wav, so a directory needs its trailing separator.
    if data_location.is_dir():
        data_location = f"{data_location}{os.sep}"
    prepare_data(curdir, str(data_location))

    split_to_utterances(curdir, rttm_path=str(rttm))

    if any(short_dir.iterdir()):
        sylnet_counts = features / "SylNet_out.txt"
        run_sylnet(short_dir, sylnet_counts)

        extract_basic_features(curdir)

        paste_columns(
            sylnet_counts, features / "other_feats.txt", features / "final_feats.txt"
        )

        regress_ALUCs(curdir)

        # Attach the filenames SylNet reported back onto the per-utterance counts.
        paste_columns(
            features / "SylNet_out_files.txt",
            features / "ALUCs_out_individual_tmp.txt",
            utterance_counts,
        )
        (features / "ALUCs_out_individual_tmp.txt").unlink()
    else:
        # No adult speech anywhere: every input file gets a zero row below.
        utterance_counts.touch()

    get_final_estimates(curdir, str(tmp_data))

    shutil.copyfile(utterance_counts, THISDIR / "ALICE_output_utterances.txt")
    shutil.copyfile(rttm, THISDIR / "diarization_output.rttm")

    if not keep_tmp:
        shutil.rmtree(tmp_data, ignore_errors=True)

    print(
        f"ALICE completed. Results written to {THISDIR / 'ALICE_output.txt'} "
        f"and {THISDIR / 'diarization_output.rttm'}."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_location",
        type=Path,
        help="a folder of .wavs, a single .wav, or a .txt file listing .wav paths, one per row",
    )
    parser.add_argument(
        "--rttm",
        type=Path,
        default=DEFAULT_RTTM,
        help="diarization output to split on (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="keep tmp_data/ instead of deleting it, to inspect intermediates",
    )
    args = parser.parse_args()

    if not args.data_location.exists():
        parser.error(f"data_location does not exist: {args.data_location}")
    if not args.rttm.is_file():
        parser.error(
            f"no diarization output at {args.rttm}. Run a diarizer first and pass "
            f"its .rttm with --rttm."
        )

    run_alice(args.data_location, args.rttm, keep_tmp=args.keep_tmp)


if __name__ == "__main__":
    main()

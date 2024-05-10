import os
import glob
import pickle
import numpy as np
import miditoolkit
import collections

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================== #  
#  Configuration                                     #
# ================================================== #  
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0}
MIN_BPM = 40
MIN_VELOCITY = 40
NOTE_SORTING = 1 #  0: ascending / 1: descending

DEFAULT_VELOCITY_BINS = np.linspace(0,  128, 64+1, dtype=int)
DEFAULT_BPM_BINS      = np.linspace(32, 224, 64+1, dtype=int)
DEFAULT_SHIFT_BINS    = np.linspace(-60, 60, 60+1, dtype=int)
DEFAULT_DURATION_BINS = np.arange(
        BEAT_RESOL/8, BEAT_RESOL*8+1, BEAT_RESOL/8)

# ================================================== #  


def traverse_dir(
        root_dir,
        extension=('mid', 'MID', 'midi'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list


def proc_one(path_midi, path_outfile):
    # Load MIDI file
    midi_obj = miditoolkit.midi.parser.MidiFile(path_midi)

    # Process instrument notes
    instr_notes = collections.defaultdict(list)
    for instr in midi_obj.instruments:
        if instr.name not in INSTR_NAME_MAP:
            continue

        instr_idx = INSTR_NAME_MAP[instr.name]
        for note in instr.notes:
            note.instr_idx = instr_idx
            instr_notes[instr_idx].append(note)

        # Sort notes by start time (descending or ascending)
        instr_notes[instr_idx].sort(
            key=lambda x: (x.start, x.pitch),
            reverse=(NOTE_SORTING == 1)  # True for descending, False for ascending
        )

    # Calculate song duration in ticks
    first_note_time = min([instr_notes[k][0].start for k in instr_notes.keys()])
    last_note_time = max([instr_notes[k][-1].end for k in instr_notes.keys()])
    song_duration_ticks = int(np.ceil(last_note_time))

    # Quantization offset (empty bars at the start)
    offset_bars = first_note_time // BAR_RESOL

    # Prepare data structures for notes, chords, tempos, and labels
    note_grid = collections.defaultdict(list)
    chord_grid = collections.defaultdict(list)
    tempo_grid = collections.defaultdict(list)
    label_grid = collections.defaultdict(list)

    # Iterate over instrument notes
    for key in instr_notes.keys():
        notes = instr_notes[key]
        for note in notes:
            # Adjust start time relative to offset
            quant_time = int(np.round((note.start - offset_bars * BAR_RESOL) / TICK_RESOL) * TICK_RESOL)

            # Calculate note duration based on start and end times
            note_duration_ticks = int(np.ceil(note.end - note.start))

            # Quantize note velocity and shift
            note.velocity = max(MIN_VELOCITY, note.velocity)
            note.shift = DEFAULT_SHIFT_BINS[np.argmin(abs(DEFAULT_SHIFT_BINS - (note.start - quant_time)))]

            # Append note to grid
            note_grid[quant_time].append(note)

    # Serialize the processed data into a dictionary
    song_data = {
        'notes': note_grid,
        'chords': chord_grid,
        'tempos': tempo_grid,
        'labels': label_grid,
        'metadata': {
            'global_bpm': DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - 120))],
            'last_bar': song_duration_ticks // BAR_RESOL
        }
    }

    # Save the serialized data as a pickle file
    os.makedirs(os.path.dirname(path_outfile), exist_ok=True)
    with open(path_outfile, 'wb') as f:
        pickle.dump(song_data, f)

if __name__ == '__main__':
    # Define input and output directories
    path_indir = 'baseline/dataset/midi_analyzed_m'
    path_outdir = 'baseline/dataset/corpus_m'
    os.makedirs(path_outdir, exist_ok=True)

    # Traverse MIDI files in the input directory
    midifiles = traverse_dir(
        path_indir,
        is_pure=True,
        is_sort=True
    )
    n_files = len(midifiles)
    print('Number of files:', n_files)

    # Process each MIDI file
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx + 1, n_files))

        # Define input and output file paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, path_midi + '.pkl')

        # Process the MIDI file and save the processed data
        proc_one(path_infile, path_outfile)



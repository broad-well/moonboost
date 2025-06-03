import pretty_midi
import os
from pathlib import Path
import argparse
import numpy as np

def extract_midi_segment_pm(input_midi_path: Path, output_midi_path: Path, duration_seconds: float = 5.0):
    """
    Extracts a segment of a MIDI file starting from the first audible note
    using the pretty_midi library.

    Args:
        input_midi_path: Path to the input MIDI file.
        output_midi_path: Path to save the extracted MIDI segment.
        duration_seconds: Duration of the segment to extract in seconds.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(input_midi_path))
    except Exception as e:
        print(f"Error: Could not read MIDI file {input_midi_path}. Skipping. Details: {e}")
        return

    first_audible_note_time = float('inf')
    has_audible_notes = False

    # Find the start time of the first audible note across all instruments
    for instrument in pm.instruments:
        if not instrument.is_drum: # Standard instruments
            for note in instrument.notes:
                if note.velocity > 0: # Audible note
                    first_audible_note_time = min(first_audible_note_time, note.start)
                    has_audible_notes = True
        else: # Drum instruments (velocity check still applies)
             for note in instrument.notes:
                if note.velocity > 0:
                    first_audible_note_time = min(first_audible_note_time, note.start)
                    has_audible_notes = True


    if not has_audible_notes:
        print(f"No audible notes found in {input_midi_path}. Skipping.")
        return

    # Define the end time for the segment
    segment_end_time_abs = first_audible_note_time + duration_seconds

    # Create a new PrettyMIDI object for the output
    output_pm = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo()) # Use original estimated tempo or a specific one

    # Copy relevant tempo changes (optional, pretty_midi handles note timings in seconds)
    # For simplicity, we'll let pretty_midi handle tempo based on note timings.
    # If you need to precisely replicate tempo changes within the 5s chunk:
    # new_tick_scales = []
    # for time, tempo in pm.get_tempo_changes():
    #     if time >= first_audible_note_time and time < segment_end_time_abs:
    #         new_tick_scales.append((time - first_audible_note_time, tempo))
    # if new_tick_scales:
    #     # This is a bit more involved as _tick_scales is (tick, qpm) and needs ticks_per_beat
    #     # For now, we rely on absolute note timings in seconds.
    #     pass


    for instrument in pm.instruments:
        # Create a new instrument for the output MIDI
        new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)

        for note in instrument.notes:
            if note.velocity == 0: # Skip silent notes
                continue

            # Check if the note is within our desired segment window
            # (original note start < segment absolute end) AND (original note end > segment absolute start)
            if note.start < segment_end_time_abs and note.end > first_audible_note_time:
                # Adjust note times to be relative to the new segment's start (0 seconds)
                new_note_start = note.start - first_audible_note_time
                new_note_end = note.end - first_audible_note_time

                # Clip the note to the 0 to duration_seconds window
                new_note_start_clipped = max(0, new_note_start)
                new_note_end_clipped = min(duration_seconds, new_note_end)

                # Only add the note if it has a positive duration after clipping
                if new_note_end_clipped > new_note_start_clipped:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=new_note_start_clipped,
                        end=new_note_end_clipped
                    )
                    new_instrument.notes.append(new_note)
        
        # Add the new instrument to the output MIDI if it has any notes
        if new_instrument.notes:
            output_pm.instruments.append(new_instrument)

    if not output_pm.instruments:
        print(f"No notes fell within the 5-second segment for {input_midi_path} after processing. Skipping save.")
        return

    try:
        output_midi_path.parent.mkdir(parents=True, exist_ok=True)
        output_pm.write(str(output_midi_path))
        print(f"Successfully processed and saved (using pretty_midi): {output_midi_path}")
    except Exception as e:
        print(f"Error: Could not save MIDI file {output_midi_path} (using pretty_midi). Details: {e}")


def process_directory_pm(input_dir_str: str, output_dir_str: str, duration: float):
    """
    Recursively processes MIDI files from input_dir to output_dir using pretty_midi.
    """
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory.")
        return

    print(f"Starting MIDI processing from '{input_dir}' to '{output_dir}' (using pretty_midi)...")

    file_extensions = ['*.mid', '*.midi']
    for ext in file_extensions:
        for midi_file_path in input_dir.glob(f'**/{ext}'):
            relative_path = midi_file_path.relative_to(input_dir)
            output_file_path = output_dir / relative_path
            print(f"\nProcessing: {midi_file_path}")
            try:
                extract_midi_segment_pm(midi_file_path, output_file_path, duration_seconds=duration)
            except Exception as e:
                print("Failed: " + str(e))
    
    print("\nMIDI processing complete (using pretty_midi).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively extracts the first N seconds of audible music from MIDI files using pretty_midi."
    )
    parser.add_argument("input_dir", help="Directory containing input MIDI files.")
    parser.add_argument("output_dir", help="Directory where processed MIDI files will be saved.")
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration of the segment to extract in seconds (default: 5.0)"
    )

    args = parser.parse_args()

    process_directory_pm(args.input_dir, args.output_dir, args.duration)
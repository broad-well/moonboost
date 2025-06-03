import pretty_midi
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import warnings

def format_key_signature_from_obj(key_obj):
    """
    Formats a pretty_midi KeySignature object into a human-readable string.
    """
    if key_obj is None:
        return None
    try:
        # key_number_to_mode_string directly gives "C major", "A minor", etc.
        return pretty_midi.key_number_to_mode_string(key_obj.key_number)
    except IndexError:
        # This can happen if pretty_midi's internal tables don't cover an unusual key_number
        # Fallback for unusual key numbers that might not be in pretty_midi's direct mapping
        key_name = pretty_midi.key_number_to_key_name(key_obj.key_number % 12)
        mode = "major" if key_obj.key_number < 12 else "minor" # Basic assumption
        return f"{key_name} {mode} (approx.)"
    except Exception as e:
        # Catch any other unexpected errors during key formatting
        # warnings.warn(f"Could not format key signature for key number {key_obj.key_number}: {e}")
        return "Unknown Key"

def format_time_signature_from_obj(ts_obj):
    """
    Formats a pretty_midi TimeSignature object into a human-readable string.
    """
    if ts_obj is None:
        return None
    return f"{ts_obj.numerator}/{ts_obj.denominator}"

def calculate_weighted_average_tempo(tempo_times, tempos_bpm, total_duration):
    """
    Calculates a weighted average tempo based on the duration each tempo is active.
    """
    if not tempos_bpm or total_duration == 0:
        return None
    if len(tempos_bpm) == 1:
        return tempos_bpm[0]

    weighted_tempo_sum = 0
    
    for i in range(len(tempos_bpm)):
        start_time = tempo_times[i]
        end_time = tempo_times[i+1] if (i + 1) < len(tempo_times) else total_duration
        duration_of_segment = end_time - start_time
        
        if duration_of_segment < 0: # Should not happen with sorted tempo_times
            duration_of_segment = 0
            
        weighted_tempo_sum += tempos_bpm[i] * duration_of_segment
        
    return weighted_tempo_sum / total_duration


def analyze_midi_file(filepath: Path) -> dict:
    """
    Analyzes a single MIDI file and extracts descriptive statistics.
    """
    try:
        with warnings.catch_warnings(): # Suppress common UserWarnings from pretty_midi for unusual MIDI files
            warnings.simplefilter("ignore", category=UserWarning)
            pm = pretty_midi.PrettyMIDI(str(filepath))
        
        # Basic info
        # pretty_midi considers tracks with note events as "instruments"
        num_instrument_tracks = len(pm.instruments)
        duration_sec = pm.get_end_time()
        
        # Notes, Pitch, Velocity
        all_notes = []
        for instrument in pm.instruments:
            all_notes.extend(instrument.notes)
        
        total_notes = len(all_notes)
        
        if duration_sec > 0:
            note_density_per_sec = total_notes / duration_sec
        elif total_notes == 0:
            note_density_per_sec = 0.0
        else: # Notes exist but duration is zero (unlikely for valid MIDIs with notes)
            note_density_per_sec = np.inf

        pitches = [note.pitch for note in all_notes]
        # Consider only notes with velocity > 0 for meaningful velocity stats
        velocities = [note.velocity for note in all_notes if note.velocity > 0]

        min_pitch = int(np.min(pitches)) if pitches else None
        max_pitch = int(np.max(pitches)) if pitches else None
        avg_pitch = float(np.mean(pitches)) if pitches else None

        min_velocity = int(np.min(velocities)) if velocities else None
        max_velocity = int(np.max(velocities)) if velocities else None
        avg_velocity = float(np.mean(velocities)) if velocities else None
        
        # Tempo
        tempo_times, tempos_bpm_list = pm.get_tempo_changes()
        initial_tempo_bpm = float(tempos_bpm_list[0]) if len(tempos_bpm_list) > 0 else None
        num_tempo_changes = len(tempos_bpm_list)
        
        # Simple average of listed tempo changes
        avg_of_listed_tempos_bpm = float(np.mean(tempos_bpm_list)) if len(tempos_bpm_list) > 0 else None
        # Weighted average tempo
        weighted_avg_tempo_bpm = calculate_weighted_average_tempo(tempo_times, tempos_bpm_list, duration_sec)
        
        estimated_tempo_pm_bpm = float(pm.estimate_tempo()) # pretty_midi's single estimate

        # Key Signatures
        num_key_changes = len(pm.key_signature_changes)
        initial_key_obj = pm.key_signature_changes[0] if num_key_changes > 0 else None
        initial_key_str = format_key_signature_from_obj(initial_key_obj)

        # Time Signatures
        num_time_sig_changes = len(pm.time_signature_changes)
        initial_time_sig_obj = pm.time_signature_changes[0] if num_time_sig_changes > 0 else None
        initial_time_sig_str = format_time_signature_from_obj(initial_time_sig_obj)

        # Resolution (ticks per beat/quarter note)
        resolution_ticks_per_beat = pm.resolution

        return {
            "filepath": str(filepath.name), # Store only filename for brevity, or filepath for full path
            "full_path": str(filepath),
            "duration_seconds": round(duration_sec, 2) if duration_sec is not None else None,
            "num_instrument_tracks": num_instrument_tracks,
            "total_notes": total_notes,
            "note_density_notes_per_sec": round(note_density_per_sec, 2) if np.isfinite(note_density_per_sec) else str(note_density_per_sec),
            "ticks_per_beat": resolution_ticks_per_beat,
            "initial_tempo_bpm": round(initial_tempo_bpm, 1) if initial_tempo_bpm is not None else None,
            "avg_of_listed_tempos_bpm": round(avg_of_listed_tempos_bpm,1) if avg_of_listed_tempos_bpm is not None else None,
            "weighted_avg_tempo_bpm": round(weighted_avg_tempo_bpm,1) if weighted_avg_tempo_bpm is not None else None,
            "estimated_tempo_pretty_midi_bpm": round(estimated_tempo_pm_bpm,1) if estimated_tempo_pm_bpm is not None else None,
            "num_tempo_changes": num_tempo_changes,
            "min_pitch_midi": min_pitch,
            "max_pitch_midi": max_pitch,
            "avg_pitch_midi": round(avg_pitch,1) if avg_pitch is not None else None,
            "min_velocity": min_velocity,
            "max_velocity": max_velocity,
            "avg_velocity": round(avg_velocity,1) if avg_velocity is not None else None,
            "num_key_signature_changes": num_key_changes,
            "initial_key_signature": initial_key_str,
            "num_time_signature_changes": num_time_sig_changes,
            "initial_time_signature": initial_time_sig_str,
        }

    except Exception as e:
        # warnings.warn(f"Could not process {filepath}: {e}")
        print(f"Skipping file {filepath.name} due to error: {e}")
        return None

def create_midi_dataframe_from_directory(directory_path: str) -> pd.DataFrame:
    """
    Recursively searches for MIDI files in a directory and creates a Pandas DataFrame
    with descriptive statistics for each file.
    """
    all_midi_data = []
    root_dir = Path(directory_path)
    
    if not root_dir.is_dir():
        print(f"Error: Directory '{directory_path}' not found.")
        return pd.DataFrame()

    # Get a list of all MIDI files for progress indication
    midi_file_paths = list(root_dir.rglob('*.mid')) + list(root_dir.rglob('*.midi'))
    total_files = len(midi_file_paths)
    
    if total_files == 0:
        print(f"No MIDI files (.mid or .midi) found in '{directory_path}' or its subdirectories.")
        return pd.DataFrame()
        
    print(f"Found {total_files} MIDI files. Starting analysis...")

    for i, file_path in enumerate(midi_file_paths):
        print(f"Processing ({i+1}/{total_files}): {file_path.name}")
        data = analyze_midi_file(file_path)
        if data:
            all_midi_data.append(data)
            
    if not all_midi_data:
        print("No MIDI files were successfully processed.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_midi_data)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze MIDI files in a directory recursively and create a Pandas DataFrame."
    )
    parser.add_argument(
        "midi_directory", 
        type=str, 
        help="Directory containing MIDI files."
    )
    parser.add_argument(
        "-o", "--output_csv", 
        type=str, 
        help="Optional: Path to save the output DataFrame as a CSV file.", 
        default=None
    )
    
    args = parser.parse_args()
    
    midi_dataframe = create_midi_dataframe_from_directory(args.midi_directory)
    
    if not midi_dataframe.empty:
        print("\n--- MIDI Analysis Complete ---")
        print(f"Successfully analyzed {len(midi_dataframe)} out of a possible {len(list(Path(args.midi_directory).rglob('*.mid')) + list(Path(args.midi_directory).rglob('*.midi')))} files.")
        
        pd.set_option('display.max_columns', None) # Show all columns
        pd.set_option('display.width', 1000) # Widen the display for head
        print("\nDataFrame Head (First 5 files):")
        print(midi_dataframe.head())
        
        print("\nDataFrame Description (Summary Statistics):")
        # For describe, we want to see stats for numeric columns primarily.
        # For non-numeric, we can list unique values or counts if needed separately.
        print(midi_dataframe.describe(include=[np.number])) # Describe only numeric columns
        
        # Example for describing object columns (like key/time signatures)
        # for col in midi_dataframe.select_dtypes(include=['object']).columns:
        #     print(f"\nValue counts for column: {col}")
        #     print(midi_dataframe[col].value_counts(dropna=False).head(10))


        if args.output_csv:
            try:
                output_path = Path(args.output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
                midi_dataframe.to_csv(output_path, index=False)
                print(f"\nDataFrame successfully saved to: {output_path}")
            except Exception as e:
                print(f"\nError saving DataFrame to CSV '{args.output_csv}': {e}")
    else:
        print("No data was extracted from MIDI files.")
import os
import numpy as np
import soundfile as sf
from pathlib import Path

def combine_audio_files(input_dir, output_file, silence_duration=0.5):
    # Get all audio files
    alex_files = sorted([f for f in os.listdir(input_dir) if f.startswith('alex_')])
    beth_files = sorted([f for f in os.listdir(input_dir) if f.startswith('beth_')])
    
    # Read the first file to get sample rate
    first_file = os.path.join(input_dir, alex_files[0])
    _, sample_rate = sf.read(first_file)
    
    # Calculate silence samples
    silence_samples = int(silence_duration * sample_rate)
    silence = np.zeros(silence_samples)
    
    # Combine all audio files in conversation order
    combined = []
    max_turns = max(len(alex_files), len(beth_files))
    
    for i in range(max_turns):
        # Add Alex's line if available
        if i < len(alex_files):
            audio, _ = sf.read(os.path.join(input_dir, alex_files[i]))
            combined.extend(audio)
            combined.extend(silence)
        
        # Add Beth's line if available
        if i < len(beth_files):
            audio, _ = sf.read(os.path.join(input_dir, beth_files[i]))
            combined.extend(audio)
            combined.extend(silence)
    
    # Convert to numpy array
    combined = np.array(combined)
    
    # Save combined audio
    sf.write(output_file, combined, sample_rate)
    print(f"Combined audio saved to: {output_file}")

def main():
    input_dir = "example/results/algorand_podcast"
    output_file = "example/results/algorand_podcast_combined.wav"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Combine the audio files
    combine_audio_files(input_dir, output_file)

if __name__ == "__main__":
    main()
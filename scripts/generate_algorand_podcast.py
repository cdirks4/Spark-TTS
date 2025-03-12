import os
import sys
import torch
import soundfile as sf
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cli.SparkTTS import SparkTTS

def read_conversation(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    alex_lines = []
    beth_lines = []
    
    for line in lines:
        if line.strip():
            if line.startswith('Alex:'):
                alex_lines.append(line.replace('Alex:', '').strip())
            elif line.startswith('Beth:'):
                beth_lines.append(line.replace('Beth:', '').strip())
    
    return alex_lines, beth_lines

def main():
    # Initialize model
    device = torch.device("cpu")
    model = SparkTTS("pretrained_models/Spark-TTS-0.5B", device)
    
    # Read the conversation
    alex_lines, beth_lines = read_conversation("scripts/algorand_podcast.txt")
    
    # Ensure output directory exists
    os.makedirs("example/results/algorand_podcast", exist_ok=True)
    
    # Generate Alex's lines (male voice with adjusted parameters)
    print("Generating Alex's lines...")
    for i, text in enumerate(alex_lines):
        wav = model.inference(
            text=text,
            gender="male",
            pitch="moderate",  # Changed from low to moderate
            speed="moderate",
            temperature=0.7,   # Reduced temperature for more stable output
            top_k=40,         # Adjusted for clearer speech
            top_p=0.9         # Slightly reduced for more focused sampling
        )
        output_path = f"example/results/algorand_podcast/alex_{i+1}.wav"
        sf.write(output_path, wav, samplerate=16000)
        print(f"Generated: {output_path}")
    
    # Generate Beth's lines (female voice with adjusted parameters)
    print("\nGenerating Beth's lines...")
    for i, text in enumerate(beth_lines):
        wav = model.inference(
            text=text,
            gender="female",
            pitch="moderate",  # Using moderate pitch instead of high
            speed="moderate",
            temperature=0.65,  # Lower temperature for more stable output
            top_k=35,         # Adjusted for clearer speech
            top_p=0.85        # More focused sampling for clearer articulation
        )
        output_path = f"example/results/algorand_podcast/beth_{i+1}.wav"
        sf.write(output_path, wav, samplerate=16000)
        print(f"Generated: {output_path}")
    
    print("\nAll audio files generated successfully!")

if __name__ == "__main__":
    main()
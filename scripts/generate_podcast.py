import os
import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.SparkTTS import SparkTTS

def main():
    # Initialize model
    device = torch.device("cpu")
    model = SparkTTS("pretrained_models/Spark-TTS-0.5B", device)
    
    # Process and generate audio
    print("Generating audio...")
    
    # Use voice parameters instead of cloning
    wav = model.inference(
        text="testing greatness",
        gender="male",
        pitch="very_low",
        speed="moderate"
    )
    
    # Save the audio
    output_dir = "example/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "trump_speech3.wav")
    sf.write(output_path, wav, samplerate=16000)
    
    print(f"Audio generated successfully! Saved to: {output_path}")

if __name__ == "__main__":
    main()
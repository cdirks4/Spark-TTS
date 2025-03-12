import os
import sys
import torch
import soundfile as sf
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cli.SparkTTS import SparkTTS

def main():
    # Ensure directories exist
    os.makedirs("example/results", exist_ok=True)
    
    # Initialize model
    device = torch.device("cpu")
    model = SparkTTS("pretrained_models/Spark-TTS-0.5B", device)
    
    # Generate speech using voice cloning
    print("Generating audio...")
    wav = model.inference(
        text="Donald trump talking",
        prompt_speech_path="example/prompts/trump_sample.wav",
        prompt_text="We know technology is moving quickly but ai is moving even faster"
    )
    
    # Save the audio
    output_path = os.path.join("example/results", "trump_speech.wav")
    sf.write(output_path, wav, samplerate=16000)
    
    print(f"Speech generated successfully! Saved to: {output_path}")

if __name__ == "__main__":
    main()
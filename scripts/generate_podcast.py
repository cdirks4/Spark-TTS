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
    
    # Download Trump's voice sample (you'll need to provide the path to a Trump speech sample)
    prompt_speech_path = "example/prompts/trump_sample.wav"  # You need to provide this file
    prompt_text = "Make America Great Again"  # Example prompt text matching the audio
    
    # Read transcript
    transcript_path = "scripts/mit_podcast_transcript.txt"
    with open(transcript_path, 'r') as f:
        transcript = f.read()
    
    # Process transcript and generate audio segments
    print("Generating audio segments...")
    segments = []
    
    # Split transcript into lines
    lines = transcript.strip().split('\n')
    
    for line in lines:
        # Skip empty lines and headers
        if not line.strip() or '=' in line or '-' in line:
            continue
            
        # Check if line starts with speaker indicator
        if line.startswith('Alex:') or line.startswith('Jamie:'):
            speaker, text = line.split(':', 1)
            text = text.strip()
            
            # Generate audio with voice cloning for Alex (Trump voice)
            if speaker.startswith('Alex'):
                wav = model.inference(
                    text,
                    prompt_speech_path=prompt_speech_path,
                    prompt_text=prompt_text
                )
            else:  # Jamie keeps original voice settings
                wav = model.inference(
                    text,
                    gender="female",
                    pitch="moderate",
                    speed="moderate"
                )
            
            segments.append(wav)
            
            # Add a small pause between segments
            pause = np.zeros(int(16000 * 0.5))  # 0.5 second pause
            segments.append(pause)
    
    # Concatenate all segments
    print("Concatenating audio segments...")
    final_audio = np.concatenate(segments)
    
    # Save final audio
    output_dir = "example/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mit_podcast.wav")
    sf.write(output_path, final_audio, samplerate=16000)
    
    print(f"Podcast generated successfully! Saved to: {output_path}")

if __name__ == "__main__":
    main()
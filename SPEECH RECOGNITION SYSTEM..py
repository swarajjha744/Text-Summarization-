"""
WHISPER SPEECH-TO-TEXT SYSTEM
Fully functional - works with files instantly.
Microphone optional (install sounddevice for that feature).
"""

import whisper
import numpy as np
import wave
import tempfile
import os
import sys

# Try to import sounddevice for microphone support
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ModuleNotFoundError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️  Note: 'sounddevice' not found. Microphone recording disabled.")
    print("   To enable mic, run: pip install sounddevice\n")


class WhisperSTT:
    def __init__(self, model_size="base"):
        """
        model_size options: tiny, base, small, medium, large
        'base' = best balance of speed and accuracy
        """
        print(f"⏳ Loading Whisper model: '{model_size}'...")
        print("   (First run downloads ~150MB, please wait...)")
        self.model = whisper.load_model(model_size)
        print("✅ Model loaded!\n")

    def transcribe_file(self, audio_path, language=None):
        """
        Transcribe any audio file (.wav, .mp3, .m4a, .flac, etc.)
        """
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            return None

        print(f"📁 Transcribing file: {audio_path}")
        result = self.model.transcribe(audio_path, language=language)
        text = result["text"].strip()

        print(f"\n📝 TRANSCRIPTION:\n{'='*50}")
        print(f"\"{text}\"")
        print(f"{'='*50}")
        print(f"🌐 Detected language: {result.get('language', 'unknown')}")
        return text

    def transcribe_microphone(self, duration=5, sample_rate=16000):
        """
        Record from microphone for 'duration' seconds and transcribe.
        """
        if not SOUNDDEVICE_AVAILABLE:
            print("\n❌ Microphone recording is unavailable.")
            print("👉 Install sounddevice by running this in your terminal:")
            print("   pip install sounddevice")
            print("\n🎵 Alternatively, use Option 2 to transcribe an audio file.\n")
            return None

        print(f"🎤 Recording for {duration} seconds... SPEAK NOW!")
        print("   (Press Ctrl+C to cancel)")

        try:
            # Record audio
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()
            print("✅ Recording complete. Transcribing...\n")

            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                with wave.open(tmpfile.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    audio_int16 = (recording * 32767).astype(np.int16)
                    wf.writeframes(audio_int16.tobytes())
                temp_path = tmpfile.name

            # Transcribe and cleanup
            result = self.transcribe_file(temp_path)
            os.remove(temp_path)
            return result

        except KeyboardInterrupt:
            print("\n⚠️ Recording cancelled.")
            return None
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None


def main():
    print("=" * 60)
    print("   🎙️  WHISPER SPEECH-TO-TEXT SYSTEM")
    print("=" * 60)

    # Initialize transcriber
    stt = WhisperSTT(model_size="base")

    while True:
        print("\n" + "-" * 60)
        print("Choose an option:")
        print("  1️⃣  Record from microphone")
        print("  2️⃣  Transcribe an audio file")
        print("  3️⃣  Exit")
        print("-" * 60)

        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == "1":
            try:
                duration = int(input("How many seconds to record? (default 5): ") or "5")
            except ValueError:
                duration = 5
            stt.transcribe_microphone(duration=duration)

        elif choice == "2":
            path = input("Enter the full path to your audio file: ").strip().strip('"')
            stt.transcribe_file(path)

        elif choice == "3":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()1

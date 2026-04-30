"""
GENERATIVE TEXT MODEL (GPT-2)
Fixed version: Smaller model, faster download, offline support.
Copy-paste this entire file into VS Code and run it.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import warnings

# Suppress symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class TextGenerator:
    def __init__(self, model_name="distilgpt2"):
        """
        model_name options:
        - 'distilgpt2'    : ~240MB, FASTEST, good quality (RECOMMENDED)
        - 'gpt2'          : ~548MB, moderate speed, good quality
        - 'gpt2-medium'   : ~1.4GB, slower, better quality
        """
        print(f"⏳ Loading model: '{model_name}'...")
        print("   (First run downloads the model. Please wait 1-3 minutes...)")
        print("   (If stuck at 0%, your internet is slow. Just wait it out.)\n")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set cache directory to avoid re-downloading
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            self.model = GPT2LMHeadModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            ).to(self.device)
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print("💡 Try switching to 'distilgpt2' which is much smaller.")
            print("💡 Or check your internet connection and try again.")
            raise

        self.model.eval()
        print(f"✅ Model loaded on {self.device}!\n")

    def generate(self, prompt, max_length=150, temperature=0.9, top_k=50,
                 top_p=0.95, repetition_penalty=1.2, num_return_sequences=1):
        """
        Generate text from a prompt.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        print(f"🧠 Generating...")
        print(f"   Prompt: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"\n")

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_texts = []
        for sequence in output:
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def interactive_mode(self):
        """Run an interactive session."""
        print("=" * 60)
        print("   📝 GENERATIVE TEXT MODEL")
        print("=" * 60)
        print("Type your prompt and hit Enter.")
        print("Type 'exit' to quit.\n")

        settings = {
            "max_length": 150,
            "temperature": 0.9,
            "top_p": 0.92,
            "repetition_penalty": 1.2
        }

        while True:
            try:
                prompt = input("🎯 Prompt: ").strip()
                if not prompt:
                    continue
                if prompt.lower() == "exit":
                    print("👋 Goodbye!")
                    break
                if prompt.lower() == "settings":
                    self._change_settings(settings)
                    continue

                texts = self.generate(prompt, **settings)

                for i, text in enumerate(texts, 1):
                    print(f"\n{'='*60}")
                    print(f"   ✨ GENERATED TEXT #{i}")
                    print(f"{'='*60}")
                    print(text)
                    print(f"{'='*60}\n")

            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted. Type 'exit' to quit.\n")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def _change_settings(self, settings):
        """Allow user to tweak generation parameters."""
        print("\n--- Current Settings ---")
        for k, v in settings.items():
            print(f"  {k}: {v}")
        print("------------------------")
        print("Enter 'key=value' to change (e.g., temperature=1.2)")
        print("Or press Enter to keep current settings.\n")

        while True:
            inp = input("Setting (or 'done'): ").strip()
            if inp.lower() == "done" or inp == "":
                break
            try:
                key, value = inp.split("=")
                key = key.strip()
                value = float(value.strip()) if "." in value else int(value.strip())
                if key in settings:
                    settings[key] = value
                    print(f"✅ Updated {key} = {value}")
                else:
                    print(f"❌ Unknown setting: {key}")
            except Exception:
                print("❌ Invalid format. Use: key=value")

        print("Settings saved!\n")


def quick_demo():
    """Non-interactive demo with preset prompts."""
    generator = TextGenerator(model_name="distilgpt2")

    prompts = [
        "The future of artificial intelligence is",
        "In a small village surrounded by mountains, there lived a",
        "The secret to building great software is",
        "Once upon a time in a galaxy far away,"
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")
        texts = generator.generate(
            prompt,
            max_length=120,
            temperature=0.85,
            top_p=0.92
        )
        print(texts[0])
        print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Choose mode:")
    print("  1️⃣  Interactive Mode (type your own prompts)")
    print("  2️⃣  Quick Demo (preset prompts)\n")

    mode = input("Enter 1 or 2: ").strip()

    if mode == "2":
        quick_demo()
    else:
        gen = TextGenerator(model_name="distilgpt2")
        gen.interactive_mode()

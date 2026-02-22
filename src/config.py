from dataclasses import dataclass

@dataclass
class TrainConfig:
    # --- Paths ---
    # Directory where setup.py downloaded the files
    model_dir: str = "./pretrained_models"

    # Path to your metadata CSV (Format: ID|RawText|NormText)
    # Using absolute path from workspace root
    csv_path: str = "/workspace/data/processed/MyTTSDataset/metadata.csv"
    metadata_path: str = "./metadata.json"

    # Directory containing WAV files
    wav_dir: str = "/workspace/data/processed/MyTTSDataset/wavs"
    #wav_dir: str = "./FileBasedDataset"

    preprocessed_dir = "/workspace/data/processed/MyTTSDataset/preprocess"
    #preprocessed_dir = "./FileBasedDataset/preprocess"

    # Output directory for the finetuned model
    # Fixed path so train.py can auto-resume from latest checkpoint.
    # (Timestamped paths create a new dir each run, breaking resume.)
    output_dir: str = "/data/output/chatterbox_output_ro_phoneme"

    is_inference = True  # Disabled: Causes CUDA errors after step 200 (GPU resource cleanup issue)
    inference_prompt_path: str = "./speaker_reference/bas_rnd1_011.wav"
    # inference_test_text: str = "Înțelepciunea înseamnă să știi că și cea mai întunecată noapte își arată strălucirea stelelor."
    # NOTE: Romanian preprocessing is applied automatically at inference time.
    # Write text in normal Romanian — it will be mapped to existing vocab tokens.
    inference_test_text: str = "De asemenea, contează și dacă imobilul este la stradă sau nu."


    ljspeech = True # Set True if the dataset format is ljspeech, and False if it's file-based.
    json_format = False # Set True if the dataset format is json, and False if it's file-based or ljspeech.
    preprocess = False # Run preprocessing separately: python -m src.preprocess_ljspeech

    is_turbo: bool = False # Set True if you're training Turbo, False if you're training Normal.

    # --- Romanian Preprocessing ---
    # Instead of extending the tokenizer (which causes posterior collapse),
    # we map Romanian chars to existing vocabulary tokens.
    # See: src/romanian_preprocessor.py for details.
    romanian_preprocessing: bool = True   # Enable Romanian text preprocessing
    romanian_mode: str = "phoneme"        # "phoneme" (ș→sh, keep ă/â/î) or "ascii" (all→ASCII)

    # --- Vocabulary ---
    # Original vocab size (2454) — NO extension needed with Romanian preprocessing.
    # The phoneme mapping avoids adding new tokens, preventing the weak-embedding
    # posterior collapse problem (see upstream issues #6, #12).
    # For Turbo mode: Use the exact number provided by setup.py (e.g., 52260)
    new_vocab_size: int = 52260 if is_turbo else 2454

    # --- Hyperparameters ---
    batch_size: int = 16         # Per-GPU batch size for 32GB VRAM
    grad_accum: int = 1
    learning_rate: float = 5e-6  # T3 is sensitive, keep low
    num_epochs: int = 1_000

    save_steps: int = 100        # Save checkpoint every 100 steps
    save_total_limit: int = 10   # Keep last 15 checkpoints
    dataloader_num_workers: int = 4  # Parallel data loading per GPU

    # --- Constraints ---
    start_text_token = 255
    stop_text_token = 0
    max_text_len: int = 256
    max_speech_len: int = 850   # Truncates very long audio
    prompt_duration: float = 3.0 # Duration for the reference prompt (seconds)ear
    
"""
Standalone LoRA merge script.
Merges a PEFT LoRA adapter into its base model and saves the full model.

Usage:
    python merge_lora.py --adapter_path <dir> --output_dir <dir>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)


def merge(adapter_path: Path, output_dir: Path) -> None:
    """Load base model + LoRA adapter, merge, and save."""
    # Read adapter_config.json to get the base model
    adapter_config_file = adapter_path / "adapter_config.json"
    if not adapter_config_file.exists():
        log.error(f"adapter_config.json not found in {adapter_path}")
        sys.exit(1)

    adapter_config = json.loads(adapter_config_file.read_text())
    base_model_id = adapter_config.get("base_model_name_or_path")
    if not base_model_id:
        log.error("base_model_name_or_path not found in adapter_config.json")
        sys.exit(1)

    merged_path = output_dir / "merged"

    log.info(f"Base model: {base_model_id}")
    log.info(f"Adapter: {adapter_path}")
    log.info(f"Output: {merged_path}")

    # Load base model
    log.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )

    # Load and merge adapter
    log.info("Loading adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    log.info("Merging adapter into base model...")
    model = model.merge_and_unload(progressbar=True)

    model.generation_config.do_sample = True
    model.config.use_cache = True

    # Save
    log.info(f"Saving merged model to: {merged_path}")
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_path), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_path))
    log.info("Merge complete.")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter_path", type=Path, required=True,
                        help="Path to LoRA adapter directory (contains adapter_model.safetensors)")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Parent directory; merged model saved to <output_dir>/merged/")
    args = parser.parse_args()

    if not args.adapter_path.exists():
        log.error(f"Adapter path does not exist: {args.adapter_path}")
        sys.exit(1)

    merge(args.adapter_path, args.output_dir)


if __name__ == "__main__":
    main()

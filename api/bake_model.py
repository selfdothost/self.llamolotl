"""
Bake pipeline: merge multiple LoRA adapters into a base model, convert to GGUF,
and optionally quantize — all in one shot.

Usage:
    python bake_model.py --config bake_config.json

Config JSON format:
{
    "base_model": "NousResearch/Llama-3.2-1B",
    "adapters": [
        {"path": "/workspace/training/outputs/lora-coding", "weight": 0.7},
        {"path": "/workspace/training/outputs/lora-chat", "weight": 0.3}
    ],
    "output_name": "my-baked-model",
    "outtype": "f16",
    "quant_type": "Q4_K_M",
    "models_dir": "/models",
    "convert_script": "/app/convert/convert_hf_to_gguf.py",
    "quantize_bin": "/app/llama-quantize"
}
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)



def step_merge(config: dict, merged_dir: Path) -> None:
    """Step 1: Load base model, apply LoRA adapters with weights, merge, save."""
    import torch
    from peft import PeftModel, set_peft_model_id
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_id = config["base_model"]
    adapters = config["adapters"]

    log.info(f"Loading base model: {base_model_id}")
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

    if len(adapters) == 1:
        # Single adapter — simple merge
        adapter = adapters[0]
        log.info(f"Loading single adapter: {adapter['path']}")
        model = PeftModel.from_pretrained(model, adapter["path"])
        # Scale if weight != 1.0
        weight = adapter.get("weight", 1.0)
        if weight != 1.0:
            log.info(f"Scaling adapter with weight {weight}")
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.data *= weight
        log.info("Merging adapter into base model...")
        model = model.merge_and_unload(progressbar=True)
    else:
        # Multiple adapters — load first, then add remaining, combine with weights
        log.info(f"Loading adapter 0: {adapters[0]['path']}")
        model = PeftModel.from_pretrained(
            model,
            adapters[0]["path"],
            adapter_name="adapter_0",
        )

        for i, adapter in enumerate(adapters[1:], start=1):
            log.info(f"Loading adapter {i}: {adapter['path']}")
            model.load_adapter(adapter["path"], adapter_name=f"adapter_{i}")

        # Combine all adapters with weighted average
        adapter_names = [f"adapter_{i}" for i in range(len(adapters))]
        weights = [a.get("weight", 1.0) for a in adapters]
        combination_name = "combined"

        log.info(f"Combining adapters {adapter_names} with weights {weights}")
        model.add_weighted_adapter(
            adapters=adapter_names,
            weights=weights,
            combination_type="linear",
            adapter_name=combination_name,
        )
        model.set_adapter(combination_name)

        log.info("Merging combined adapter into base model...")
        model = model.merge_and_unload(progressbar=True)

    model.generation_config.do_sample = True
    model.config.use_cache = True

    log.info(f"Saving merged model to: {merged_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    log.info("Merge complete.")


def step_convert(config: dict, merged_dir: Path, gguf_path: Path) -> None:
    """Step 2: Convert merged HF model to GGUF."""
    convert_script = config.get("convert_script", "/app/convert/convert_hf_to_gguf.py")
    outtype = config.get("outtype", "f16")

    log.info(f"Converting to GGUF ({outtype}): {merged_dir} -> {gguf_path}")
    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile", str(gguf_path),
        "--outtype", outtype,
    ]

    env = dict(__import__("os").environ)
    env["NO_LOCAL_GGUF"] = "1"

    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed with exit code {result.returncode}")
    log.info("GGUF conversion complete.")


def step_quantize(config: dict, gguf_path: Path, quant_path: Path) -> None:
    """Step 3: Quantize GGUF to target type."""
    quantize_bin = config.get("quantize_bin", "/app/llama-quantize")
    quant_type = config["quant_type"]

    log.info(f"Quantizing: {gguf_path} -> {quant_path} ({quant_type})")
    cmd = [str(quantize_bin), str(gguf_path), str(quant_path), quant_type]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed with exit code {result.returncode}")
    log.info("Quantization complete.")


def main():
    parser = argparse.ArgumentParser(description="Bake LoRA adapters into a GGUF model")
    parser.add_argument("--config", type=str, required=True, help="Path to bake config JSON")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())

    models_dir = Path(config.get("models_dir", "/models"))
    output_name = config["output_name"]
    outtype = config.get("outtype", "f16")
    quant_type = config.get("quant_type")

    # Use a temp directory for the intermediate merged HF model
    work_dir = Path(tempfile.mkdtemp(prefix="bake-", dir="/workspace/training"))
    merged_dir = work_dir / "merged"

    # Final output filenames
    gguf_name = f"{output_name}-{outtype}.gguf"
    gguf_path = models_dir / gguf_name

    try:
        # Step 1: Merge LoRAs
        log.info("=" * 60)
        log.info("STEP 1/3: Merging LoRA adapters")
        log.info("=" * 60)
        step_merge(config, merged_dir)

        # Step 2: Convert to GGUF
        log.info("=" * 60)
        log.info("STEP 2/3: Converting to GGUF")
        log.info("=" * 60)
        step_convert(config, merged_dir, gguf_path)

        # Step 3: Quantize (optional)
        if quant_type:
            log.info("=" * 60)
            log.info("STEP 3/3: Quantizing")
            log.info("=" * 60)
            quant_name = f"{output_name}-{quant_type}.gguf"
            quant_path = models_dir / quant_name
            step_quantize(config, gguf_path, quant_path)

            # Remove the unquantized intermediate if quantization succeeded
            if quant_path.exists() and gguf_path.exists():
                log.info(f"Removing intermediate {gguf_name}")
                gguf_path.unlink()
                final_path = quant_path
            else:
                final_path = gguf_path
        else:
            log.info("STEP 3/3: Skipping quantization (no quant_type specified)")
            final_path = gguf_path

        log.info("=" * 60)
        log.info(f"BAKE COMPLETE: {final_path}")
        log.info(f"Size: {final_path.stat().st_size / (1024**3):.2f} GB")
        log.info("=" * 60)

    finally:
        # Clean up intermediate merged HF model
        if work_dir.exists():
            log.info(f"Cleaning up work directory: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

"""
Patch transformers' Qwen3.5 model to skip fla (flash-linear-attention) imports
and use the built-in torch fallback (torch_chunk_gated_delta_rule).

triton 3.6.0 has a regression in its MLIR code_generator that causes all fla
triton kernels to segfault at first invocation (during kernel JIT compilation).
transformers already ships a pure-PyTorch fallback for Qwen3.5's GatedDeltaRule
layers — we just need to prevent fla from being loaded so the fallback activates.
"""

import importlib.util
from pathlib import Path

TARGET = (
    Path(importlib.util.find_spec("transformers").submodule_search_locations[0])
    / "models" / "qwen3_5" / "modeling_qwen3_5.py"
)

src = TARGET.read_text()

OLD = "if is_flash_linear_attention_available():"
NEW = "if False:  # fla disabled: triton 3.6.0 MLIR code_generator segfault; using torch fallback"

if OLD not in src:
    print(f"WARNING: target pattern not found in {TARGET} — may already be patched or file changed")
else:
    TARGET.write_text(src.replace(OLD, NEW, 1))
    print(f"Patched {TARGET}")

# Verify
import importlib, transformers.models.qwen3_5.modeling_qwen3_5 as mod
importlib.reload(mod)
# chunk_gated_delta_rule should now be None (torch fallback path)
assert mod.chunk_gated_delta_rule is None, "Expected fla to be disabled"
print("Verification OK: chunk_gated_delta_rule is None, torch fallback active")

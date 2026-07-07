#!/bin/bash
# Wrapper for llama-server that reads extra arguments from a config file.
# This allows the API to dynamically configure LoRA adapters, chat templates,
# and other flags without modifying supervisord config.

EXTRA_ARGS_FILE="/app/llama-server.args"

EXTRA_ARGS=""
if [ -f "$EXTRA_ARGS_FILE" ]; then
    EXTRA_ARGS=$(cat "$EXTRA_ARGS_FILE")
fi

# Chat template: rely on GGUF-embedded tokenizer.chat_template by default.
# A custom template override can be written to /app/chat-template-override.jinja
# by the API (via UI upload), in which case we pass --chat-template-file.
TEMPLATE_ARGS=""
OVERRIDE_FILE="/app/chat-template-override.jinja"
if [ -f "$OVERRIDE_FILE" ]; then
    TEMPLATE_ARGS="--chat-template-file $OVERRIDE_FILE"
fi

# shellcheck disable=SC2086
exec /app/llama-server --jinja $TEMPLATE_ARGS $EXTRA_ARGS "$@"

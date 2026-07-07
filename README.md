# self.llamolotl

Inference + training backbone of the self.ai platform. Dual-purpose container:
`llama.cpp` (via the `self.llama` submodule) serves GGUF models over HTTP;
a FastAPI service on the side runs HuggingFace Trainer + DeepSpeed + PEFT
training jobs against the same GPU.

---

**Component of [self.ai](https://github.com/selfdothost/self.ai)** — the self-hosted AI-serving stack. Licensed under GPL-3.0 (see [`LICENSE`](LICENSE)). See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution model (DCO sign-off + `Assisted-by:` AI-disclosure trailer).

---

## CI

Pipeline is defined in `.gitlab-ci.yml`. Three stages, rolled out in order:

| Stage | Job | Purpose | Status |
|-------|-----|---------|--------|
| `test` | `test:unit` | Build the full image and run `api/tests/` (31 pytest tests) inside it. No GPU at runtime. | **live** |
| `smoke` | `test:inference-smoke` | Pull the stage-1 image, start it with `--gpus all`, verify `/health`, completion, chat, LoRA endpoints. | (pending T-026+) |
| `perf` | `perf:inference` + `perf:training` | Run `llama-bench` + a 10-step LoRA training run with 500 ms telemetry, compare to baselines. | (pending T-040+) |

### Runner requirement

All jobs — including unit tests — target the self.ai GPU runner pool via
`tags: [cuda-13, docker, self.ai, gpu]`. Unit tests do **not** need runtime
GPU and are launched without `--gpus all`; they share the pool with stages 2
and 3 purely to avoid CI config drift between runners. The Dockerfile itself
is GPU-less at build time (CUDA stub libs, `GGML_NATIVE=OFF`, fixed
`TORCH_CUDA_ARCH_LIST`, `DS_SKIP_CUDA_CHECK=1`), so the image can build on a
plain host — pinning to the GPU fleet is operational, not technical.

### When does the pipeline run?

`workflow:` rules restrict pipelines to merge-request events and
default-branch pushes. Branch pushes outside the default branch do **not**
trigger a pipeline. To force a pipeline on a feature branch, open an MR.

Default-branch merge protection (require a green pipeline before merge) is
configured in the GitLab project settings UI, **not** in `.gitlab-ci.yml`.

### Image tags published by `test:unit`

| Tag | When | Consumers |
|-----|------|-----------|
| `${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHORT_SHA}` | every pipeline | stage-2 `test:inference-smoke`, stage-3 `perf:*` pull by this tag |
| `${CI_REGISTRY_IMAGE}:ci-latest` | default-branch only | human-use `latest` pointer |
| `${CI_REGISTRY_IMAGE}:cache` | every build (`buildctl --export-cache`) | BuildKit layer-cache manifest — not a runnable image |

### Local reproduction

The CI pipeline runs rootless — `buildctl` against a user-scope `buildkitd`
for the build, `podman run` for the pytest exec, no Docker daemon anywhere.
Local reproduction uses the same two tools (both install standalone on
Debian/Ubuntu/Fedora/macOS).

You need:
- A recursively-initialized checkout (`git submodule update --init --recursive`)
- `buildctl` + a reachable `buildkitd` (rootless works: `systemctl --user enable --now buildkitd.socket` after installing buildkit from the [moby/buildkit](https://github.com/moby/buildkit/releases) release)
- `podman` (any modern version — reads `~/.docker/config.json` for registry auth same as Docker)

From repo root:

```bash
git submodule update --init --recursive && \
  buildctl build \
    --frontend dockerfile.v0 \
    --local context=. \
    --local dockerfile=. \
    --opt filename=Dockerfile \
    --output type=image,name=localhost/self-llamolotl:ci && \
  podman run --rm -w /workspace/training localhost/self-llamolotl:ci \
    python -m pytest api/tests/ -v
```

The CI `test:unit` adds `--junitxml=/ci-out/report.xml` over a bind-mount
(`-v "$(pwd)/ci-out:/ci-out"`) so the report survives `--rm`; locally you
can drop both the flag and the mount.

### Build timing

| | Cold registry | Warm registry |
|--|---------------|---------------|
| `test:unit` image build | TBD (first pipeline) | target < 5 min |

Numbers land after the first MR pipeline actually runs — filled in by the
T-015 measurement pass.

### Debugging a red pipeline

1. Check the GitLab MR test-report widget for pytest failures first —
   `report.xml` is uploaded as a JUnit artifact even when the job fails
   (`when: always`).
2. Download the `ci-out/` artifact from the job page for the full report
   plus any incidentally-captured outputs.
3. Reproduce locally with the one-liner above.
4. If the failure is in `buildctl build` (not pytest), grep the job log
   for the BuildKit step that failed and correlate with `Dockerfile` /
   `self.llama/` state — the `GIT_SUBMODULE_STRATEGY: recursive` guard at
   the top of `test:unit` catches a missing submodule with a clear FATAL
   message before the build even starts.

---

## Layout

- `api/` — FastAPI training service + pytest suite
- `self.llama/` — llama.cpp submodule (pinned)
- `training/` — training scripts, configs, fixtures
- `context/kits/` — implementation-agnostic specs (Cavekit)
- `context/plans/` — generated build site and task graph
- `context/impl/` — per-domain implementation tracking
- `Dockerfile` — full multi-stage image (llama.cpp + training stack)
- `Dockerfile.llama-only` — slim image with just llama.cpp + llama-server
- `.gitlab-ci.yml` — CI pipeline (see above)

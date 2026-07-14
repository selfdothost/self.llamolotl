# self.llamolotl

Inference + training backbone of the self.ai platform. Dual-purpose container:
`llama.cpp` (vendored at `self.llama/`, see [`NOTICE`](NOTICE)) serves GGUF
models over HTTP; a FastAPI service on the side runs HuggingFace Trainer +
DeepSpeed + PEFT training jobs against the same GPU.

---

**Component of [self.ai](https://github.com/selfdothost/self.ai)** — the self-hosted AI-serving stack. Licensed under GPL-3.0 (see [`LICENSE`](LICENSE)). `self.llama/` is a fully vendored, GPLv3-relicensed tree of [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) — see [`NOTICE`](NOTICE) for provenance; the original MIT text is retained in `LICENSE.llama-cpp-mit`. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution model (DCO sign-off + `Assisted-by:` AI-disclosure trailer).

---

## CI

Pipeline is defined in `.gitlab-ci.yml`. It is a **build-only** pipeline today —
two `kaniko` image-build jobs, no test/smoke/perf stages:

| Stage | Job | Purpose |
|-------|-----|---------|
| `build` | `build:llama-server` | `kaniko` build of `Dockerfile.llama-only` (llama.cpp + llama-server only). Publishes `:llama-only-<sha>` and `:llama-only-latest`. |
| `build` | `build:training` | `kaniko` build of `Dockerfile` (full llama.cpp + HF Trainer/DeepSpeed/PEFT stack). Publishes `:<sha>`, `:latest`, and `:training-<sha>`. `:latest` is the tag self.ai's llamolotl deployment tracks. |
| `publish` | `sync:public-alpha` | Scheduled mirror of `main` to the `public-alpha` branch (GitLab + GitHub), via the shared template in `selfai/self.ai`. Unrelated to the build jobs. |

Both build jobs run on the `bigbuild` runner tag (kaniko, no GPU needed — the
image builds GPU-less: CUDA stub libs, `GGML_NATIVE=OFF`, fixed
`TORCH_CUDA_ARCH_LIST`, `DS_SKIP_CUDA_CHECK=1`) and are gated by per-job
`rules:` to merge-request events and pushes to `main`.

### There is no test stage right now

The previous pipeline shape (rootless `buildctl`/`podman`, a `test:unit` job
that built the full image and ran `api/tests/` inside it, plus planned
`smoke`/`perf` stages) was ripped out in commit `00b5de2` in favor of the
kaniko build-only pipeline above. That wasn't a decision to drop testing —
the from-source `llama.cpp` + `DeepSpeed` build was too slow for the runner
of the day, and when the test job *did* run to completion, container startup
was broken in the CI path (`supervisord` launching the full app stack instead
of `pytest`, plus broken `libcuda`/Python-import wiring). None of that was
fixed; the build was reshaped around it instead so images could keep
publishing.

This is a known, tracked gap, not an oversight — see:
- `self.llamolotl#3` — stage-1 build time on the (then) CPU runner; a
  24-vCPU runner bump is in flight to unblock this.
- `self.llamolotl#4` — the deeper build-time investigation beyond just
  scaling vCPUs (prebuilt binaries, image-split CI, base-image choice, etc.).
- `self.llamolotl#6` — the container-startup-in-CI bugs (`supervisord` vs
  `pytest` invocation, `libcuda` dlopen failure, relative-import breakage)
  that surfaced the last time the test job actually ran.

Until those close, there is no CI signal on `api/tests/` — run pytest
locally against a built image before merging changes to `api/`.

### Local reproduction (build only)

```bash
docker build -f Dockerfile.llama-only -t self-llamolotl:llama-only .
# or, for the full training image:
docker build -f Dockerfile -t self-llamolotl:full .
```

`kaniko` in CI builds the same `Dockerfile`/`Dockerfile.llama-only` against
the same checkout; a plain `docker build` locally is a reasonable proxy for
whether the image itself builds, though it won't catch anything
`kaniko`-cache-specific.

To run the pytest suite against a built image, see the caveat above — there
is currently no supported one-liner for this because the container's startup
path doesn't cleanly support a `pytest`-only invocation (`self.llamolotl#6`).

---

## Layout

- `api/` — FastAPI training service + pytest suite
- `self.llama/` — vendored llama.cpp tree (pinned, see [`NOTICE`](NOTICE))
- `training/` — training scripts, configs, fixtures
- `context/kits/` — implementation-agnostic specs (Cavekit)
- `context/plans/` — generated build site and task graph
- `context/impl/` — per-domain implementation tracking
- `Dockerfile` — full multi-stage image (llama.cpp + training stack)
- `Dockerfile.llama-only` — slim image with just llama.cpp + llama-server
- `.gitlab-ci.yml` — CI pipeline (see above)

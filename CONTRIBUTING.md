# Contributing to self.llamolotl

Thanks for your interest. self.llamolotl is GPL-3.0 software (see `LICENSE`).
Contributions of all kinds are welcome — code, docs, bug reports, and reviews.

## A person stands behind every contribution

You may use AI tools to help write a contribution. That is fine and expected.
But **a human must stand behind every commit** and take responsibility for its
entire contents. We enforce this two ways:

### 1. Developer Certificate of Origin (DCO) — required

Every commit must carry a real `Signed-off-by:` line. By signing off you certify
the [Developer Certificate of Origin](https://developercertificate.org/) — that
you wrote the change or otherwise have the right to submit it under the project's
license.

```
Signed-off-by: Jane Developer <jane@example.com>
```

Add it automatically with `git commit -s`. The name and email must be real and
must match the commit author. A DCO check runs on every pull request; commits
without a valid sign-off will not be merged.

**AI agents never sign off.** Only a human can certify the DCO. A tool did not
"submit" the change — you did.

### 2. Disclose AI assistance with `Assisted-by:`

If an AI tool wrote a non-trivial part of a commit, disclose it with an
`Assisted-by:` trailer alongside your sign-off. This follows the convention used
by the Linux kernel, Fedora, and LLVM:

```
Assisted-by: Claude:claude-opus-4-8
Signed-off-by: Jane Developer <jane@example.com>
```

The trailer documents which tool helped; the `Signed-off-by` is still yours and
still means you are accountable for the whole commit.

## Pull request flow

1. Fork the repo and branch from `main`.
2. Make focused commits, each `Signed-off-by` (and `Assisted-by:` where it
   applies).
3. Open a PR and fill in the template, including the AI-disclosure field.
4. CI runs lint/tests and the DCO check. First-time contributors may need a
   maintainer to approve the workflow run.
5. A maintainer reviews. Address feedback with additional signed-off commits.

## Report security issues privately

Do **not** open a public issue for a vulnerability. This repo does not yet have
a `SECURITY.md`; report suspected vulnerabilities privately to a maintainer
instead (e.g. via a private issue or direct message) rather than in public.

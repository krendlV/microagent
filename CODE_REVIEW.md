# MicroAgent — Code Review & Path to Public Release (Round 3)

**Review date:** 2026-06-06
**Reviewed at commit:** `2df0b09` (branch `fix/multichannel-hwc-segmentation`)
**Author of review:** prepared for Valentin Krendl
**Scope:** Public-release readiness. This round assumes the pipeline *works* (Round 2,
`726082e`, verified that end-to-end) and instead asks: *what stands between this repo and
a confident public launch + a paper?* It covers the giant-HTML problem, multi-provider /
open-LLM support, non-technical usability and the UI question, a hostable demo, CC0 test
datasets, and packaging/citation truth.
**Supersedes:** Round 2 review (`726082e`). The Round 2 "remaining issues" are folded in
below where still open.
**Companion doc:** `PUBLIC_RELEASE_PROMPTS.md` — every fix below as a paste-ready prompt.

---

## 1. Executive summary

MicroAgent is a **solid beta with an unusually clean architecture** (CLI ↔ MCP parity,
disciplined optional-deps, real FAIR provenance, reproducibility bundles). The core
pipeline runs clean from checkout to report. The gap to *public release* is no longer
"does it work" — it's **truth-in-packaging, one serious output bug, openness of the LLM
layer, and the on-ramp for non-technical users.**

**Grade: solid beta; ~2–3 focused days from a defensible v0.1.0 public tag.**

The five things that actually matter before you show this to anyone:

1. **🔴 Reports are 117–122 MB each.** This is the single most embarrassing thing a new
   user will hit. It's a one-file fix (§2). **Release blocker.**
2. **🟠 The LLM layer is hardwired to Anthropic.** Your stated goal is "open — Mistral,
   local LLMs, anyone's provider." Right now `extract_from_text` only talks to
   `anthropic`, and the only agentic surface is MCP. Both are fixable cleanly (§4).
3. **🟠 Packaging advertises things that don't exist yet** — `pip install microagent`,
   PyPI/Codecov badges, a published version. A reviewer or student who copy-pastes the
   README's first install line gets an error. (§6)
4. **🟡 Non-technical on-ramp is missing.** Today the entry point is a terminal + `uv`.
   For students, that's a wall. The good news: you do *not* need to build a big UI to
   fix this — see §5 for the cheapest high-impact path.
5. **🟡 No bundled CC0 test data / no demo anyone can click.** You can't show off model
   training/optimization without annotated data, and "clone this and run `uv sync`" is
   not a demo. (§7, §8)

Everything else is polish.

---

## 2. 🔴 The 117 MB HTML report — root cause and fix

### What's happening

```
test_output/results/myotubes/report.html   122 MB
test_output/results/nuclei/report.html      117 MB

overlays/1_overlay.png      4.9 MB
overlays/1_sidebyside.png   9.2 MB
overlays/montage.png        5.5 MB         ← all of these get embedded
```

Three compounding causes, all in `viz/`:

1. **Full-sensor-resolution, lossless PNG.** Overlays are rendered at the native image
   size (2048×2048) and `montage` is saved at `dpi=300` (`viz/overlays.py:265`). A
   photographic overlay as lossless PNG is 3–10 MB. The *same picture* as quality-85
   JPEG or WebP is typically **20–60× smaller**.
2. **base64 embedding inflates everything by ~33%.** `viz/report.py::_embed_image`
   inlines each PNG as a `data:` URI so the report is self-contained — a *good* default,
   but it means file size matters enormously.
3. **Every PNG in `overlays/` is embedded, three per image.** `cli.py::report` points
   `overlay_dir` at `overlays/` (or `masks/`), and the template's gallery loops over
   *all* of them — so `_overlay`, `_sidebyside`, **and** `montage` are each inlined.
   For 3 images that's ~9 lossless full-res PNGs in one HTML file. The gallery also
   wraps each thumbnail in `<a href="{{ same data URI }}">`, so the multi-MB string is
   the both the thumbnail *and* the "full size" link.

### Is base64-embedding the right approach at all?

Yes — **for self-contained reports it's the correct default**, and you should keep it as
*an* option. A single shareable file with no broken-image links is exactly what a PI
wants to email or attach to a paper. The mistake is not the embedding; it's embedding
**giant lossless images at full resolution, all of them.** Fix the inputs and embedding
becomes cheap.

### The fix (target: <2 MB reports, visually identical)

In `viz/report.py::_embed_image` (and the montage/overlay savers), before encoding:

1. **Downscale for display.** Cap the embedded image at ~1600 px on the long edge
   (gallery thumbnails need far less; full-res helps no one inside an HTML `<img>`).
2. **Re-encode as JPEG q≈85** (or WebP) for photographic overlays; keep PNG only for
   line-art plots/charts. Pillow is already a hard dependency.
3. **Stop embedding redundant variants.** Pick *one* representation per image for the
   gallery (the overlay), and drop `montage` + `sidebyside` from the default embed set
   — or gate them behind a flag.
4. **Offer two modes** via a `report --embed/--no-embed` flag:
   - `--embed` (default): self-contained, downscaled JPEG — small and emailable.
   - `--no-embed`: write a `report_assets/` folder of full-res PNGs next to the HTML and
     reference them with relative paths — for users who want full resolution and don't
     need a single file.

Expected result: the same 117 MB report drops to **~0.5–2 MB** with no visible quality
loss at screen resolution. This is the highest-value 30 lines of code in the repo right
now. *(Prompt R1.)*

---

## 3. What's actually missing / incomplete in the code

Beyond the report bloat, concrete gaps found by reading the tree:

### 3.1 Correctness / robustness (carryover + new)
- **`n_labels = int(mask.max())`** still reports the max label *value* as the object
  count (`segment.py`). Wrong for non-contiguous labels. Use `len(np.unique(mask)) - 1`.
  *(Low, but it's a number that goes in reports and a paper — fix it.)* *(Prompt R5.)*
- **Silent empty masks on poor-fit fallback.** H&E project + no StarDist → CellPose on
  the red channel → ~0 objects, no warning. A student will read "0 cells" as a result,
  not a misconfiguration. Emit a one-line warning when falling back off the recommended
  backend. *(Prompt R6.)*
- **`inspect` writes thumbnails into the *input* directory**
  (`<data>/microagent_inspection/`). Mutating the user's data folder is surprising and,
  on read-only mounts (a hosted demo!), a hard crash. Route to `--output`/cwd. *(R7.)*

### 3.2 MCP / agentic surface — the parity gaps matter more now
Because the MCP server is your *only* "LLM co-works on the images" surface today, its
gaps are release-relevant, not cosmetic:
- **MCP `segment` ignores `project.yaml`** (no `project` arg) → LLM users never get
  recommended-model selection.
- **MCP tools don't log to `experiments.jsonl`** → every reproducibility/FAIR claim is
  CLI-only. For an *agentic* tool whose pitch is reproducibility, LLM-driven runs being
  invisible is a contradiction. Wrap MCP tool bodies in `tracked_run` (or, at minimum,
  document the limitation prominently). *(Prompt R4.)*
- **No `report`-equivalent richness in MCP** — `generate_report` exists, good, but the
  LLM has no tool to *open/return* the report or a summary back to the user.

### 3.3 Optional-dep / install reality
- **micro-SAM is conda-only**, correctly excluded from `[all]`, but the README still
  lists it in the model table without a loud "conda-only" caveat at point of use.
- **`bioimageio.core`** is declared in `[all]` and `[bioimage]` but I see no code path
  that imports it. Either wire it (BioImage.IO model zoo is a *great* fit for your
  "look up alternative models" pitch) or drop it from deps. *(Dead dependency — R8.)*

### 3.4 Tests
- Round 2's "232 passed, 3 skipped" is real but **CI only installs `--extra dev`**, so
  optimize/stardist/mcp paths skip in CI. Add one `--extra all` (+ optuna/mcp/stardist)
  job so the optional paths are actually exercised before you tag. *(Prompt R9.)*
- No test asserts report **size stays bounded** — add one after the §2 fix so the bloat
  can't silently regress (it's the kind of thing that comes back). *(Part of R1.)*

---

## 4. 🟠 Making the LLM layer genuinely open (Mistral / local / any provider)

Your differentiator vs. a napari plugin is real: a CLI+MCP tool can be driven by *any*
agent, can fetch alternative models from the web/GitHub, and can emit reproducible
artifacts. But the code doesn't yet live up to "open provider" in two places:

### 4.1 The one hardcoded provider: `project/knowledge.py::extract_from_text`
It calls `anthropic` directly with a pinned `claude-sonnet-4-6` model. For your Europe /
Mistral / local-LLM goal, refactor to a **provider-agnostic shim**:
- Read `MICROAGENT_LLM_PROVIDER` (`anthropic` | `openai` | `mistral` | `ollama` | `none`)
  and `*_API_KEY` / `*_BASE_URL` env vars.
- Default to an **OpenAI-compatible** client (`base_url` + `api_key`), because Mistral,
  Ollama, vLLM, LM Studio, and OpenRouter all speak that dialect — one code path covers
  local *and* most cloud providers. Keep the Anthropic branch as one option.
- Keep the keyword-heuristic fallback (already there) for the no-key / offline case.

This is small and makes "anyone can use their provider" true in code, not just in spirit.
*(Prompt R2.)*

### 4.2 The agentic story: MCP is the right backbone — make it provider-neutral in docs
MCP itself is already provider-neutral (Claude Code, Cursor, Continue, LibreChat, and
others speak it; local models via an MCP-capable client work too). The gap is
**documentation**: `docs/mcp-integration.md` and the README show only Claude Code. Add a
short "Use it with any MCP client / local model" section (LibreChat + Ollama is a good
fully-open recipe; Mistral via an MCP-capable client for the EU story). *(Prompt R2.)*

> **Recommendation:** *Don't* build a provider-abstraction layer for chat inside
> MicroAgent. The tool's job is to be a great MCP server + CLI; let the *client* own the
> LLM. The only in-process LLM call is `extract_from_text`, and that just needs the shim
> above. This keeps MicroAgent genuinely open and avoids you maintaining N SDK
> integrations.

---

## 5. 🟡 Non-technical users & the UI question

You asked the right question: *is building a UI the answer?* Here's the honest staged
answer, cheapest-first. **You almost certainly do not need a heavy GUI to start.**

### Tier 0 — make the CLI itself non-scary (½ day, do this first)
- A single `microagent run <folder>` "do everything" command (inspect → segment →
  evaluate-if-GT → report → open it) so a student types **one line**, not five.
- Friendlier errors and a "what next?" hint after each command (you already use `rich`
  beautifully — lean into it).
- A `--open` flag on `report`/`run` that auto-opens the HTML.

### Tier 1 — the report *is* the UI (1–2 days, highest ROI for students)
Your strongest asset is the HTML report. Make it the teaching surface:
- After §2's fix, embed **explanatory text** ("what is F1? what does a low score mean?"),
  a glossary, and "try this next" suggestions. A student learns by reading their own
  result. This is the "iteratively learn about microscopy imaging" goal, delivered with
  zero new infrastructure.

### Tier 2 — the chat *is* the UI (the agentic pitch, mostly already built)
For "an LLM co-works on their images," the MCP server + a ready-made chat client (Claude
Desktop, LibreChat, Cursor) **is** the product. The student talks to the model; the model
drives MicroAgent. Ship a copy-paste MCP config and a 3-line "ask it to segment your
images" recipe and you're 80% there. The remaining 20% is §3.2 (MCP parity + tracking).

### Tier 3 — a thin web UI (only if you want a hosted classroom demo)
*If* you want a clickable, no-install experience (e.g. for a workshop), the cheapest real
option is a **Gradio or Streamlit** app (~150–250 lines) that wraps the existing core
functions: upload images → pick/auto model → segment → show overlays + metrics → download
report. Keep it in an *optional* extra (`microagent[webui]`) and out of `core/` (respects
your "no GUI deps in core" rule). This is the thing you'd put on a server for a demo (§8).
Do **not** build a custom React app for v0.1 — it's months of work for a research tool.

**My recommendation for your stated audience (students + LLM co-work):** Tier 0 + Tier 1
+ Tier 2 now (days, not weeks); Tier 3 Gradio app only when you need the hosted workshop
demo. *(Prompts R3, R10.)*

---

## 6. 🟠 Packaging & citation truth (release blocker)

The repo currently *describes a published package that doesn't exist*:
- README leads with `pip install microagent` and shows PyPI/Python/Codecov badges. None
  resolve. A new user's first action fails.
- `version = "0.1.0"`, no git tags, not on PyPI.

Before public:
- **Lead the README with the source/`uvx` install**; move `pip install microagent` under
  a "once published" note (Round 2 already softened this — finish the job and gate the
  badges behind real CI/coverage or remove them). *(Prompt R8.)*
- **Citation:** README's BibTeX now correctly says `author = {Krendl, Valentin}` ✅. But
  the in-code `extract_from_text` aside and `docs/` don't carry author/citation. Add a
  `CITATION.cff` (GitHub renders a "Cite this repository" button — exactly what you want
  for the paper) with your name + ORCID, and a "How to cite" section in the docs.
  Reserve a Zenodo DOI on first tagged release so the paper can cite a versioned archive.
  *(Prompt R11.)*
- **Tag `v0.1.0`** and publish to **TestPyPI** at least, so the quickstart is real.

---

## 7. 🟡 CC0 test data for training / optimization

You correctly note you need annotated data to exercise train/optimize. Don't commit it
to the repo (violates CLAUDE.md's "no real data" rule and bloats clones). Instead add a
**`microagent fetch-dataset <name>`** helper that downloads to a cache dir. Good CC0 /
permissive, annotated, small options:

| Dataset | Content | License | Why |
|---|---|---|---|
| **Cellpose specialist/generalist sample images** | varied cells, GT masks | permissive | matches your default backend |
| **DSB 2018 (Kaggle BBBC038 / Broad)** | nuclei, instance masks | CC0 | the canonical nuclei benchmark |
| **BBBC datasets (Broad Bioimage Benchmark Collection)** | many, with GT | mostly CC0/public | huge variety, citable |
| **StarDist demo data (`2D_demo`)** | fluorescence nuclei + masks | permissive | exercises StarDist path |

Implement as an optional downloader (guarded `requests`/`urllib`, cached under
`~/.cache/microagent/datasets/`, checksum-verified), plus a `--dataset dsb2018` shortcut
on `train`/`optimize` for one-command demos. This *also* powers the hosted demo (§8) and
gives you reproducible numbers for the paper. *(Prompt R12.)*

---

## 8. 🟡 A demo anyone can run (and one you can host)

Two levels:
1. **Zero-install local demo** already exists (`microagent demo`) — good. Make sure it
   produces a *small* report after §2 and link the resulting `report.html` from the
   README (commit one tiny example report, post-fix, so people see the output without
   running anything). A short asciinema/GIF of `microagent run` in the README converts
   far better than prose.
2. **Hosted demo (for showing people / workshops):** the Tier-3 Gradio app (§5) on a
   small GPU box or HuggingFace Spaces. Wire it to the §7 sample dataset so visitors can
   click "segment the example" with no upload, then optionally upload their own. Keep
   uploads ephemeral and the LLM optional (so it runs without API keys). A `Dockerfile`
   already exists — extend `docker-compose.yml` with a `webui` service. *(Prompt R10.)*

> Caveat to flag for any hosted deployment: §3.1's "inspect writes into the input dir"
> will crash on read-only inputs — fix that before hosting.

---

## 9. What's genuinely good (don't touch)

- Optional-dependency discipline and graceful fallbacks.
- CLI ↔ MCP verb parity and the `{"status": "error"}` MCP convention.
- Rich, honest provenance (git hash + dirty flag, lib versions, GPU/CUDA, data hash).
- The reproducibility bundle is substantive (regenerates an environment).
- `core/cellpose_compat.py` centralizing version skew — the right instinct.
- Docs are already broad (getting-started, user-guide, mcp, faq, architecture).

---

## 10. Prioritized roadmap to public release

| # | Action | Why it blocks/limits release | Effort | Prompt |
|---|--------|------------------------------|--------|--------|
| 1 | **Shrink HTML reports** (downscale + JPEG/WebP + don't embed all variants + `--no-embed`) | 117 MB output is the #1 first-impression killer | **S** | R1 |
| 2 | **Provider-agnostic LLM shim** + "any MCP client / local model" docs | Delivers the stated "open / Mistral / local" promise | M | R2 |
| 3 | **`microagent run` one-liner** + `--open` + next-step hints | Non-technical on-ramp (Tier 0) | S | R3 |
| 4 | **MCP: `project` arg + `tracked_run` logging** | Agentic surface ↔ reproducibility parity | M | R4 |
| 5 | Fix `n_labels` to count unique nonzero labels | Correct counts in reports/paper | S | R5 |
| 6 | Warn on poor-fit backend fallback | Avoid silent "0 cells" for students | S | R6 |
| 7 | Stop `inspect` writing into input dir | Correctness + required for hosting | S | R7 |
| 8 | Packaging truth: README install order, gate badges, drop/justify `bioimageio` | Don't ship a broken first command | S | R8 |
| 9 | CI job with `--extra all` | Actually exercise optional paths pre-tag | S | R9 |
| 10 | Optional **Gradio web UI** (`[webui]`) + compose service | Hostable demo / workshop (Tier 3) | M | R10 |
| 11 | `CITATION.cff` + ORCID + Zenodo DOI + docs "how to cite" | Paper-readiness for Valentin Krendl | S | R11 |
| 12 | `microagent fetch-dataset` (CC0) + `--dataset` shortcut | Test train/optimize; demo data; paper numbers | M | R12 |
| 13 | Teaching-mode report text/glossary | Student learning surface (Tier 1) | S | R13 |
| 14 | Tag `v0.1.0`, publish TestPyPI, example report + GIF in README | Make the quickstart real | M | R14 |

**Suggested order:** R1 → R8 → R3 → R5/R6/R7 (the quick correctness/UX wins) → R2 → R4 →
R9 → R11 → R12 → R10/R13 → R14.

R1, R3, R5, R6, R7, R8, R9 are roughly a day and get you to "won't embarrass you in
public." R2, R4, R11, R12 are the next day and deliver the openness + paper + demo-data
story. R10/R13/R14 are the polish that makes a great first impression.

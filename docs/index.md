# MicroAgent Documentation

Open-source Python CLI + MCP server for automated microscopy image segmentation, evaluation, training, and reporting.

## Guides

- [Getting Started](getting-started.md) — installation, first run, understanding output
- [User Guide](user-guide.md) — project.yaml, model selection, fine-tuning, batch processing
- [API Reference](api-reference.md) — every CLI command and option
- [MCP Integration](mcp-integration.md) — Claude Code, Cursor, tool reference
- [project.yaml Reference](project-yaml-reference.md) — full schema documentation
- [Architecture](architecture.md) — module design, data flow, extension points
- [FAQ](faq.md) — common issues, GPU troubleshooting, positioning

## Quick Links

```bash
pip install microagent
microagent demo
```

[GitHub](https://github.com/krendlV/microagent) · [PyPI](https://pypi.org/project/microagent/) · [License: BSD-3-Clause](https://github.com/krendlV/microagent/blob/main/LICENSE)

---

## How to Cite

The authoritative citation metadata is in [`CITATION.cff`](https://github.com/krendlV/microagent/blob/main/CITATION.cff) (Citation File Format 1.2.0). GitHub renders a **"Cite this repository"** button at the top of the repo page.

**BibTeX:**

```bibtex
@software{microagent2026,
  author  = {Krendl, Valentin},
  title   = {{MicroAgent}: Agentic Microscopy Image Analysis},
  year    = {2026},
  url     = {https://github.com/krendlV/microagent},
  license = {BSD-3-Clause}
}
```

A journal article is in preparation; the `preferred-citation` block in `CITATION.cff` will be updated with the final reference once published.

### Zenodo DOI

A persistent Zenodo DOI will be minted automatically on the first tagged release (`v0.1.0`).
**Before tagging**, enable the GitHub–Zenodo integration at <https://zenodo.org/account/settings/github/> and flip the toggle for `krendlV/microagent`. After tagging, Zenodo issues a DOI of the form `10.5281/zenodo.XXXXXXX`; update `CITATION.cff` with the real DOI and replace the badge placeholder in `README.md`.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

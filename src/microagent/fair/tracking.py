"""Experiment tracking: log, retrieve and compare runs via JSONL."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from microagent.fair.provenance import RunMetadata, collect_metadata


class ExperimentTracker:
    """Append-only JSONL experiment log.

    Parameters
    ----------
    path:
        Path to the JSONL file. Created on first write if absent.
    """

    def __init__(self, path: Path = Path("experiments.jsonl")) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log_run(self, metadata: RunMetadata, results: dict) -> str:
        """Append a run record and return its run_id.

        Parameters
        ----------
        metadata:
            Provenance metadata collected for this run.
        results:
            Arbitrary result dict (metrics, output paths, …).

        Returns
        -------
        str
            8-character run_id derived from the SHA-256 of the JSON line.
        """
        record: dict = {
            "metadata": metadata.to_dict(),
            "results": results,
        }
        line = json.dumps(record, default=str, sort_keys=True)
        run_id = hashlib.sha256(line.encode()).hexdigest()[:8]
        record["run_id"] = run_id

        # Re-serialise with run_id included (deterministic key order).
        final_line = json.dumps(record, default=str, sort_keys=True)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(final_line + "\n")

        return run_id

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _iter_records(self) -> Generator[dict, None, None]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def get_run(self, run_id: str) -> dict:
        """Retrieve a run record by its run_id.

        Parameters
        ----------
        run_id:
            8-character run identifier returned by :meth:`log_run`.

        Returns
        -------
        dict
            The full run record.

        Raises
        ------
        KeyError
            If *run_id* is not found.
        """
        for record in self._iter_records():
            if record.get("run_id") == run_id:
                return record
        raise KeyError(f"Run '{run_id}' not found in {self.path}")

    def list_runs(self, last_n: int = 10) -> list[dict]:
        """Return the *last_n* run records (most recent last).

        Parameters
        ----------
        last_n:
            Maximum number of records to return.

        Returns
        -------
        list[dict]
            Up to *last_n* records from the end of the log.
        """
        all_records = list(self._iter_records())
        return all_records[-last_n:]

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        """Compare two runs and return a diff of metadata and results.

        Parameters
        ----------
        run_id_a:
            First run identifier.
        run_id_b:
            Second run identifier.

        Returns
        -------
        dict
            Mapping of key → {"a": value_a, "b": value_b} for keys that differ.
        """
        a = self.get_run(run_id_a)
        b = self.get_run(run_id_b)

        def _flatten(d: dict, prefix: str = "") -> dict:
            out: dict = {}
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    out.update(_flatten(v, full_key))
                else:
                    out[full_key] = v
            return out

        flat_a = _flatten(a)
        flat_b = _flatten(b)
        all_keys = set(flat_a) | set(flat_b)
        return {
            k: {"a": flat_a.get(k), "b": flat_b.get(k)}
            for k in sorted(all_keys)
            if flat_a.get(k) != flat_b.get(k)
        }


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def tracked_run(
    tracker: ExperimentTracker,
    command: str,
    params: dict,
    data_path: Path | None = None,
    *,
    log_on_exception: bool = True,
):
    """Context manager that times a block and logs provenance on exit.

    Parameters
    ----------
    tracker:
        :class:`ExperimentTracker` instance to log into.
    command:
        Full CLI command string being executed.
    params:
        All parameters (including defaults) for the run.
    data_path:
        Path to input data for computing data_hash (optional).
    log_on_exception:
        If ``True`` (default), persist partial results even when the block
        raises. CLI commands set this to ``False`` so failed invocations do not
        create successful-looking run records.

    Yields
    ------
    dict
        A mutable ``results`` dict — populate it inside the ``with`` block
        and it will be persisted to the tracker on exit.

    Example
    -------
    >>> tracker = ExperimentTracker()
    >>> with tracked_run(tracker, "microagent segment …", params, data) as r:
    ...     r["ap50"] = 0.87
    """
    results: dict = {}
    start = time.monotonic()

    def _log() -> None:
        elapsed = time.monotonic() - start
        metadata = collect_metadata(
            command=command,
            parameters=params,
            random_seed=params.get("seed", 0),
            data_path=data_path,
            wall_clock_seconds=round(elapsed, 3),
        )
        results["run_id"] = tracker.log_run(metadata, results)

    try:
        yield results
    except Exception:
        if log_on_exception:
            _log()
        raise
    else:
        _log()

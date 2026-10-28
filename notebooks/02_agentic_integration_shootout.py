# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anndata==0.13.*",
#     "annoy",
#     "anthropic==1.17.*",
#     "arraylake==1.7.*",
#     "bbknn",
#     "h5py",
#     "harmonypy",
#     "icechunk==2.3.*",
#     "jax==0.11.*",
#     "leidenalg",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "openai==3.26.1",
#     "pandas",
#     "pooch",
#     "pyarrow==24.0.0",
#     "scanorama",
#     "scanpy==1.12.*",
#     "scib-metrics",
#     "scikit-learn",
#     "scipy",
#     "scvi-tools==1.5.*",
#     "torch==2.14.*",
#     "zarr==3.4.*",
# ]
#
# [tool.pixi.workspace]
# channels = ["conda-forge"]
# platforms = ["osx-arm64", "linux-64", "linux-aarch64"]
# conda-pypi-map = { "conda-forge" = "conda-pypi-map.json" }
#
# [tool.pixi.dependencies]
# python = "3.12.*"
# icechunk = "2.3.*"
# arraylake = "1.7.*"
# zarr = "3.4.*"
# anndata = "0.13.*"
# scanpy = "1.12.*"
# scvi-tools = "1.5.*"
# pytorch = "2.14.*"
# jax = "0.11.*"
# python-annoy = "*"
# leidenalg = "*"
# pooch = "*"
# anthropic = "1.17.*"
# openai = "==3.26.1"
# pyarrow = "==24.0.0"
# h5py = "*"
# matplotlib-base = "*"
# numpy = "*"
# pandas = "*"
# scipy = "*"
# scikit-learn = "*"
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App()

with app.setup(hide_code=True):
    import contextlib
    import dataclasses
    import datetime
    import functools
    import json
    import logging
    import multiprocessing
    import os
    import queue
    import secrets
    import shutil
    import signal
    import statistics
    import tempfile
    import time
    import warnings
    from collections.abc import Callable, Mapping
    from dataclasses import dataclass
    from pathlib import Path

    # Scoring runs on the CPU. Only scvi and scanvi use ICB_ACCELERATOR.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import anndata
    import anthropic
    import arraylake as al
    import bbknn
    import h5py
    import harmonypy
    import icechunk
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import openai
    import pandas as pd
    import pooch
    import scanpy as sc
    import scipy.sparse
    import scvi
    import torch
    import zarr
    from anndata import AnnData
    from anthropic import beta_tool
    from arraylake import Client
    from arraylake.exceptions import ArraylakeClientError, AuthException
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
    from scipy.sparse import issparse
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Commit writes one manifest per new array, serially by default.
    COMMIT_MANIFEST_CONCURRENCY = 32

    REQUIRED_BY_STORE = {
        "arraylake": ("ICB_ARRAYLAKE_ORG",),
        "s3": ("ICB_S3_BUCKET", "ICB_S3_PREFIX", "ICB_S3_REGION"),
    }

    API_KEY_BY_PROVIDER = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

    # Lightning accelerator names that `ICB_ACCELERATOR` accepts.
    ACCELERATORS = ("cpu", "gpu", "auto")

    DATA_DIR = Path(__file__).resolve().parents[1] / "data"

    IMMUNE_URL = "https://ndownloader.figshare.com/files/25717328"
    IMMUNE_FILENAME = "Immune_ALL_human.h5ad"
    IMMUNE_SHA256 = "sha256:96709b5fa677d5eafc734793a7873a2c763453ba8abddad24c0fba39e0d7f2c8"
    # Its layers["counts"] is 99% non-integer; the other batches hold raw counts.
    IMMUNE_EXCLUDED_BATCH = "Villani"

    # Base snapshot and run branches for NB2.
    # The base keeps all genes, because scVI trains on `layers["counts"]` of the same genes
    # the other methods use. It stores `var["highly_variable"]` and `obsm["X_pca"]`, not an HVG subset.
    # Cells are sorted by batch, because scanorama needs contiguous batches.
    BASE_TAG = "immune-base-v2"
    BUILD_BRANCH = "base-build"
    BATCH_KEY = "batch"
    N_HVG, N_PCS = 2000, 50

    # Integration method runners, parameter checks and the branch writer for NB2.
    # Methods run on the HVG subset of the base. The base cells are sorted by batch, as scanorama needs.
    LABEL_KEY = "final_annotation"
    # Query batches: scANVI trains without their labels, and Bio conservation scores only their cells.
    QUERY_BATCHES = ("Oetjen_A", "Oetjen_P", "Oetjen_U")
    PRETRAIN_EPOCHS = 100

    PARAM_SPACES: dict[str, dict[str, tuple]] = {
        "unintegrated": {"n_comps": (int, 20, 50)},
        "harmony": {"theta": (float, 0.5, 4.0), "sigma": (float, 0.05, 0.2)},
        "combat": {"n_comps": (int, 20, 50)},
        "scanorama": {"knn": (int, 10, 40), "sigma": (float, 5, 30), "alpha": (float, 0.05, 0.3)},
        "bbknn": {"neighbors_within_batch": (int, 1, 6), "trim": (int, 0, 50)},
        "scvi": {"n_latent": (int, 10, 50), "n_layers": (int, 1, 3), "max_epochs": (int, 20, 100)},
        "scanvi": {"max_epochs": (int, 5, 20)},
    }

    # Iteration limits per agent at ICB_ITERATION_SCALE=1.
    ITERATION_LIMITS = {"unintegrated": 1, "harmony": 4, "combat": 2, "scanorama": 4, "bbknn": 3, "scvi": 3, "scanvi": 3}

    # Harmony params of the by-hand iteration: the harmonypy defaults.
    BY_HAND_PARAMS = {"theta": 2.0, "sigma": 0.1}

    # Single-embedding scIB scoring for NB2.
    SCORES = ("Batch correction", "Bio conservation", "Total")
    PCR_CLIPPED = "PCR comparison score is negative"

    # Cost and wall-clock caps for the NB2 agents.
    log = logging.getLogger(__name__)

    # Prices in US$ per million tokens.
    # Anthropic cache_write is the rate for 5-minute entries, 1.25x input.
    PRICES_PER_MTOK = {
        "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20, "cache_write": 2.5},
        # Server-side fallback target of claude-sonnet-5-5 refusals; usage then names this model.
        "claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20, "cache_write": 2.5},
        "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0},
        # OpenAI (developers.openai.com/api/docs/pricing, 2026-10-22): cache_read is the cached-input rate.
        "gpt-6-luna": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
        "gpt-6.1-sol": {"input": 2.0, "output": 10.0, "cache_read": 0.10},
    }

    USAGE_FIELDS = {
        "input_tokens": "input",
        "output_tokens": "output",
        "cache_read_input_tokens": "cache_read",
        "cache_creation_input_tokens": "cache_write",
        # OpenAI usage: `input_tokens` also counts these, so `tokens_cost` takes them out of the input charge.
        "cached_input_tokens": "cache_read",
    }

    # An unknown model costs the highest rate of each field, so the cap errs toward an early stop.
    MAX_PRICES = {
        p: max(prices[p] for prices in PRICES_PER_MTOK.values() if p in prices) for p in USAGE_FIELDS.values()
    }
    warned: set[str] = set()

    # File checkpoints for the failure scenarios.
    # A writer writes `<prefix>.<step>` and waits. The parent then kills it, or writes `<prefix>.<step>.go` to let it continue.
    WAIT_S = 1800.0
    POLL_S = 0.05

    # Provider-neutral tool loop for the NB2 agents. The backends are `run_openai_conversation` and `run_anthropic_conversation`.
    # The OpenAI backend uses the Responses API with function tools. The Anthropic backend uses the SDK beta tool runner.
    MODELS = {
        "openai": {"method": "gpt-6-luna", "evaluator": "gpt-6.1-sol"},
        "anthropic": {"method": "claude-sonnet-5-5", "evaluator": "claude-opus-5-5"},
    }
    MAX_TOKENS = 16000

    # System prompts for the NB2 agents.
    NOTES = {
        "unintegrated": "This is the baseline: the first n_comps base PCs, with no batch correction.",
        "harmony": "Harmony corrects the 50 base PCs. theta is the diversity penalty; sigma is the soft-cluster width.",
        "combat": "ComBat corrects the HVG expression, then a scaled PCA with n_comps components gives the embedding.",
        "scanorama": "Scanorama corrects the 50 base PCs by mutual nearest neighbours across batches.",
        "bbknn": "BBKNN builds a batch-balanced kNN graph on the base PCs. The embedding is the 2-D UMAP of that graph.",
        "scvi": "scVI trains a VAE on the raw counts of the HVGs, on CPU. More epochs cost more time.",
        "scanvi": "scANVI starts from one fixed 100-epoch scVI pretrain and uses the cell-type labels, "
        "except in the 3 Oetjen batches, which have no labels for it. "
        "Only max_epochs is tunable.",
    }

    EVALUATOR_SYSTEM_PROMPT = (
        "You are the evaluator of a batch-integration shootout on the human immune atlas: "
        "32,484 cells from several batches, with expert cell-type labels. One method agent tuned each method on its own "
        "branch. All branches forked from the base snapshot of the run main.\n"
        "\n"
        "Tools:\n"
        "- branch_scores(): reads each commit of each method branch, read only. For each method it gives the scIB "
        "metrics of the iteration with the highest Total (\"Bio conservation\", \"Batch correction\", \"Total\"), its "
        "params and iteration number, the fork snapshot and the iteration count. A method with no metrics has no result.\n"
        "- check_lineage(method): tells if the branch descends from the run base and if main still points to the base.\n"
        "- promote(method): fast-forwards main to the best iteration of the branch, moves the method labels to obs/leiden_integrated, "
        "and adds that column to obs. It refuses if the lineage check fails.\n"
        "- tag_winner(method): tags the promoted main.\n"
        "- discard(methods): deletes the branches of the losing methods.\n"
        "- collect_garbage(): deletes the objects that no branch or tag can reach. The history stays. "
        "The result gives the reclaimed bytes in bytes_deleted.\n"
        "\n"
        "Steps:\n"
        "1. Call branch_scores.\n"
        "2. Pick one winner. Weigh \"Bio conservation\" against \"Batch correction\". "
        "The default rank is \"Total\" = 0.6 * Bio conservation + 0.4 * Batch correction. "
        "Batch correction that removes biology is a loss: if the Total leader has much lower Bio conservation "
        "than the runner-up, you can pick the runner-up. Tell why. Never pick a method that has no metrics.\n"
        "3. Call check_lineage for the winner, then promote. If either one refuses, stop: "
        "do not tag, discard or collect garbage. Report the reason.\n"
        "4. Call tag_winner for the winner.\n"
        "5. Call discard with all the other methods.\n"
        "6. Call collect_garbage.\n"
        "7. Write a short report: the winner and why, with its Bio conservation, Batch correction and Total "
        "against the runner-up; the tag; the discarded branches; the reclaimed bytes (bytes_deleted) and chunks.\n"
        "\n"
        "Rules:\n"
        "- Call one tool at a time.\n"
        "- Call promote one time only.\n"
    )

    # NB2 method agents and the shootout runner.
    # Each agent is one `spawn` process that tunes one method on its own branch.
    # The parent drains usage reports into a `Budget` and terminates every agent at a cap.

    # Library thread pools read these at import, so the parent sets them before a child starts.
    THREAD_ENV_VARS = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    )
    TERMINATE_GRACE_S = 10.0
    # After a stop, agents get this long to report their last calls and exit before termination.
    STOP_GRACE_S = 30.0
    # Env key of the checkpoint prefix for the failure scenarios. Unset in the shootout.
    WRITE_SIGNAL_VAR = "ICB_WRITE_SIGNAL"

    # Read-only branch scores, lineage checks and promotion of the winning branch to the run main.
    # Promotion copies no data: a fast-forward, one metadata move, and one attribute commit.
    PROMOTED_COLUMN = "leiden_integrated"
    GC_FIELDS = (
        "bytes_deleted",
        "chunks_deleted",
        "manifests_deleted",
        "snapshots_deleted",
        "attributes_deleted",
        "transaction_logs_deleted",
        "objects_failed_to_delete",
        "throttled_batches",
        "skipped_phases",
        "delete_errors",
    )

    # The NB2 evaluator agent: it compares the method branches, promotes one winner, and removes the losers.
    # The promotion steps are plain code; the model only picks the winner and the order of calls.
    AGENT = "evaluator"
    # The full flow is 6 tool turns and 1 report turn; the rest covers refused calls.
    MAX_TURNS = 12

    # The NB2 dry run: the real method tools, promotion and GC, with fixed choices in place of the LLM.
    # `ICB_NB2_DRY_RUN=1` selects it. It makes no API call, so it needs no API key and costs nothing.
    ENV_VAR = "ICB_NB2_DRY_RUN"
    # Only tests set it, to a local repo path. The notebook dry run opens the reader store.
    LOCAL_REPO_VAR = "ICB_NB2_LOCAL_REPO"
    MODEL = "none (dry run, no LLM)"
    # The real scANVI pretrain has 100 epochs; the dry run only shows the flow.
    DRY_RUN_PRETRAIN_EPOCHS = 20
    # Fixed params per iteration. The harmony values differ from SEED_PARAMS, so a killed write differs.
    PARAMS = {
        "unintegrated": [{"n_comps": 30}],
        "harmony": [{"theta": 3.0, "sigma": 0.1}, {"theta": 1.0, "sigma": 0.1}],
        "combat": [{"n_comps": 30}],
        "scanorama": [{"knn": 20, "sigma": 15.0, "alpha": 0.1}],
        "bbknn": [{"neighbors_within_batch": 3, "trim": 0}],
        "scvi": [{"n_latent": 10, "n_layers": 1, "max_epochs": 20}],
        "scanvi": [{"max_epochs": 5}],
    }

    # NB2 failure scenarios: each failure mode on Icechunk and on the careful sidecar baseline.
    # Each scenario returns `{"icechunk": {...}, "sidecar": {...}}` observations. The caller gives each one its own run.
    # Each scenario takes `produce`, the spawn target that writes method output; its default is the real agent.
    METHOD = "harmony"
    # Scenario 3 commits this untuned result as the branch that it discards.
    BASELINE, BASELINE_PARAMS = "unintegrated", {"n_comps": 30}
    # Scenario 1 commits this result first, so the killed agent write is the second write on the branch.
    # The values differ from the harmonypy defaults (theta 2.0, sigma 0.1), which an agent often picks.
    SEED_PARAMS = {"theta": 0.5, "sigma": 0.2}
    REQC_QUANTILE = 0.99
    # The sidecar writer stops before this rename: the first and largest file, where a random kill most often lands.
    SIDECAR_KILL_FILE = "embedding.npy"
    BASE_PATHS = {"X", "layers", "obs", "var"}
    # Arraylake adds these personal fields to commit metadata. Records keep the keys but not the values.
    AUTHOR_KEYS = ("author_email", "author_name")
    SCENARIOS_POLL_S = 0.1
    SCENARIOS_WAIT_S = WAIT_S

    # NB2 measurements: commit latency against step time, manifests, open time, stored bytes and the 1M-cell projection.
    PROJECTED_CELLS = 1_000_000

    # Table rows and plots for the NB2 notebook. The notebook cells only display them.
    # One label set for the agent and evaluator exit reasons, which use two vocabularies.
    EXIT_LABELS = {
        "finished": "finished",
        "done": "finished",
        "refusal": "model refused",
        "refused": "model refused",
        "stopped": "stopped",
        "cost": "cost cap",
        "time": "time cap",
        "turn_limit": "turn limit",
        "max_turns": "turn limit",
        "end_turn": "ended without finish",
        "error": "error",
        "crashed": "crashed",
        "unknown": "no exit report",
    }


@app.cell
def _():
    mo.md(r"""
    # Branches for agentic method comparison

    This notebook compares 7 batch-integration methods on the human immune atlas.
    The data has 32,484 cells in 9 batches, with 16 cell types in `obs["final_annotation"]`.
    One LLM agent tunes each method.
    Each agent writes to its own Icechunk branch, and all 7 agents run at the same time.
    An evaluator agent then reads all branches and promotes the best result to `main`.
    It also deletes the other branches.

    The claim is that branches make this flow safe and cheap.
    Each agent writes only the paths of its method, and all branches share the base data.
    A failed or killed agent cannot change `main`.
    Promotion moves a branch ref and makes two small commits. It copies no data.

    The second part of the notebook runs four failure scenarios.
    Each scenario runs on Icechunk and on a careful sidecar-file baseline.

    Two caps apply to the whole notebook, failure scenarios included: US$5 of API cost and 90 minutes of wall-clock time.
    `ICB_MAX_MINUTES` changes the time cap.
    Code enforces both caps.
    You pay the API cost to your LLM provider.
    The notebook shows the actual cost and time per agent.
    The free tier of Arraylake is enough for the storage.

    The run needs at least 32 GiB of memory, because 7 agents hold the data at the same time.
    An AWS `m7g.4xlarge` (16 vCPU, 64 GiB, Graviton) is enough.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Design

    The diagram shows the flow of one run.
    Each agent and the by-hand iteration write only to their own branch.
    Only the promotion writes to the `main` branch of the run.
    """)
    return


@app.cell
def _():
    mo.mermaid(
        """
    flowchart TD
        data["Immune atlas: 32,484 cells, 9 batches"] --> base["Base snapshot, tag immune-base-v2: HVGs and a 50-PC PCA"]
        base --> main["Branch run-main"]
        base --> byhand["Branch run-by-hand: one Harmony iteration with no LLM, then deleted"]
        base --> agent
        subgraph agent["7 method agents in parallel, each on branch run-method"]
            llm["LLM picks the params"] --> fit["run_method"]
            fit --> scib["score with scib-metrics"]
            scib --> commit["write_method_output and commit"]
            commit --> llm
        end
        agent --> evaluator["Evaluator LLM: branch_scores, then picks one winner"]
        evaluator --> lineage["check_lineage"]
        lineage --> promote["promote: fast-forward to the best iteration, move the labels"]
        promote --> main
        promote --> tag["tag_winner"]
        tag --> discard["discard the other branches"]
        discard --> gc["collect_garbage"]
        main --> result["Result on main: UMAP, ARI and NMI"]
        gc --> scenarios["4 failure scenarios, each in its own run: Icechunk against a sidecar baseline"]
    """
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## Agent prompts

    Each method agent gets the system prompt that `method_system_prompt` makes for its method.
    The prompt gives the parameter space, the iteration limit, the two tools and the goal.
    The evaluator gets one fixed system prompt, `EVALUATOR_SYSTEM_PROMPT`, from the setup cell.
    The tabs below show the exact text that each agent gets.
    """)
    return


@app.function
def space_lines(method: str) -> str:
    return "\n".join(
        f"- {name}: {kind.__name__}, from {low} to {high} inclusive"
        for name, (kind, low, high) in PARAM_SPACES[method].items()
    )



@app.function
def method_system_prompt(method: str, limit: int) -> str:
    """Return the system prompt for the agent of `method` with at most `limit` iterations. It has no per-run text, so it caches."""
    return f"""You tune one single-cell integration method: {method}.
{NOTES[method]}

The data is the human immune atlas: 32,484 cells from several batches (studies and technologies), \
with expert cell-type labels. Each method runs on the 2,000 highly variable genes.

Parameter space. Every call must give every parameter, inside these limits:
{space_lines(method)}

Tools:
- run_and_score(params, reasoning): runs {method} with `params`, scores the embedding with scIB, \
and commits the result to your branch. `reasoning` is one short sentence that tells why you chose \
these values; it becomes the commit message. The result gives "Batch correction", "Bio conservation", \
"Total", and the snapshot id. Invalid params give an "error" and no commit.
- finish(summary): ends your work. `summary` gives your best params and their Total.

Goal: the highest "Total" score. Total = 0.6 * Bio conservation + 0.4 * Batch correction.

Rules:
- You have at most {limit} run_and_score calls. One call takes from 30 seconds to some minutes.
- Make one run_and_score call at a time, and use each result to choose the next params.
- Stop when the Total score stops improving, or when the limit is reached. Then call finish.
- Do not call run_and_score again with params that you already tried.
"""



@app.cell
def _(env):
    # A wrong ICB_ITERATION_SCALE stops the run in the settings cell; the tabs then show the default limits.
    try:
        _limits = iteration_limits(env)
    except ValueError:
        _limits = ITERATION_LIMITS
    _prompts = {method: method_system_prompt(method, n) for method, n in _limits.items()} | {"evaluator": EVALUATOR_SYSTEM_PROMPT}
    mo.ui.tabs({name: mo.md(f"```text\n{text}\n```") for name, text in _prompts.items()})
    return


@app.cell
def _():
    env = dict(os.environ)
    DRY_RUN = enabled(env)
    if DRY_RUN:
        print("DRY RUN: no LLM call. Scripts with fixed parameters replace the agents and the evaluator.")
    mo.callout(
        mo.md(
            "**Dry run: no LLM.** `ICB_NB2_DRY_RUN=1` is set. "
            "Scripts with fixed parameters replace the 7 method agents and the evaluator. "
            "The method fits, scores, commits, promotion and GC are real. "
            "The dry run needs no API key, and its API cost is zero."
        ),
        kind="warn",
    ) if DRY_RUN else None
    return DRY_RUN, env


@app.cell
def _():
    mo.md(r"""
    ## Settings

    The notebook reads its settings from environment variables:

    - `ICB_LLM`: `openai` (the default) or `anthropic`.
    - `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, for the provider in `ICB_LLM`.
      The key comes from the API console of the provider. A ChatGPT or claude.ai subscription does not give API access.
    - `ICB_STORE`: `arraylake` (the default) or `s3`.
    - For Arraylake: `ICB_ARRAYLAKE_ORG`. The notebook asks you to log in when it needs to.
      `ARRAYLAKE_TOKEN`, an Arraylake API token, skips the login.
    - For S3: `ICB_S3_BUCKET`, `ICB_S3_PREFIX` and `ICB_S3_REGION`. The AWS credentials come from the standard AWS environment.
    - `ICB_ACCELERATOR`: `cpu` (the default), `gpu` or `auto`. The value `gpu` trains scvi and scanvi on a GPU. The other methods run on the CPU.
    - `ICB_ITERATION_SCALE`: a whole number, 1 by default. It multiplies the iteration limit of each agent. The unintegrated baseline stays at 1 iteration.
    - `ICB_MAX_MINUTES`: the time cap of the notebook in minutes, 90 by default. A larger scale needs more time.

    The OpenAI backend chains the turns of a conversation with `previous_response_id`.
    Thus OpenAI stores the responses on its servers for 30 days by default.

    The cell below stops the notebook if a setting is missing.
    """)
    return


@app.cell
def _(DRY_RUN, env):
    missing = dry_run_missing_settings(env)
    if missing:
        print("Missing settings:", ", ".join(missing))
    mo.stop(
        bool(missing),
        mo.callout(mo.md(f"Set these environment variables, then run the notebook again: `{'`, `'.join(missing)}`"),
                   kind="danger"),
    )
    provider = llm_provider(env)
    models = MODELS[provider]
    _llm = "none (dry run)" if DRY_RUN else f"{provider}: agents `{models['method']}`, evaluator `{models['evaluator']}`"
    _limits = ", ".join(f"{m} {n}" for m, n in iteration_limits(env).items())
    mo.md(
        f"All settings are present. Store: `{env.get('ICB_STORE', 'arraylake')}`. LLM: {_llm}.\n\n"
        f"Iteration limits: {_limits}. Time cap: {max_minutes(env):g} minutes."
        + (" The dry run uses its fixed parameter lists, so it ignores the scale." if DRY_RUN else "")
    )
    return missing, models


@app.cell
def _(env, missing):
    # The reference to missing makes this cell run after the settings check.
    assert not missing
    logged_in = ensure_arraylake_login(env)
    mo.md("The notebook asks you to log in to Arraylake here if it has no token.")
    return (logged_in,)


@app.cell
def _():
    mo.md(r"""
    ## Base and run

    The base snapshot holds the immune data with HVG flags and a 50-PC PCA, under the tag `immune-base-v2`.
    The first run on a repo builds the base. Later runs find the tag and reuse the base.
    One budget holds both caps from this cell to the end, so a base build counts toward the time cap.

    Each run gets its own `main` branch, `<run id>-main`, at the base snapshot.
    Thus two runs in one repo never write to the same branch.

    Run one notebook per repo at a time.
    The evaluator runs GC with a cutoff of now.
    That GC deletes the uncommitted chunks of any other writer on the same repo.
    """)
    return


@app.class_definition
@dataclass
class Run:
    id: str
    main: str
    base: str



@app.function
def build_base():
    """Loads the immune data and adds the HVG flags and a batch-aware 50-PC PCA on the HVGs."""
    adata = load_immune()
    adata = adata[np.argsort(adata.obs[BATCH_KEY].to_numpy(), kind="stable")].copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, batch_key=BATCH_KEY, flavor="seurat")
    scaled = adata[:, adata.var["highly_variable"]].copy()
    scaled.X = scaled.X.toarray() if issparse(scaled.X) else scaled.X  # scale densifies anyway; avoids its warning
    sc.pp.scale(scaled, max_value=10)
    sc.pp.pca(scaled, n_comps=N_PCS, random_state=0)
    adata.obsm["X_pca"] = scaled.obsm["X_pca"]
    return adata



@app.function
def ensure_base(repo: icechunk.Repository) -> str:
    """Returns the snapshot id of BASE_TAG. Builds and tags the base if the tag is missing."""
    try:
        return repo.lookup_tag(BASE_TAG)
    except icechunk.RefNotFoundError:
        pass
    if BUILD_BRANCH in repo.list_branches():
        repo.delete_branch(BUILD_BRANCH)
    repo.create_branch(BUILD_BRANCH, repo.lookup_branch("main"))
    session = repo.writable_session(BUILD_BRANCH)
    write_adata(session, build_base())
    # Method branches write under /results; the base owns the parent group, so branches do not all create it.
    write_element(session, "results", {})
    snapshot = session.commit("Add immune base with HVGs and PCA")
    repo.create_tag(BASE_TAG, snapshot)
    return snapshot



@app.function
def start_run(repo: icechunk.Repository) -> Run:
    """Creates the main branch of a new run at the base snapshot."""
    run_id = fresh_name("r")
    base = ensure_base(repo)
    main = f"{run_id}-main"
    repo.create_branch(main, base)
    return Run(id=run_id, main=main, base=base)



@app.cell
def _(env, logged_in, missing):
    # The settings cell checks env. marimo runs this cell after the login cell because the body uses logged_in.
    assert not missing
    assert isinstance(logged_in, bool)
    budget = Budget(max_usd=5.0, max_seconds=max_minutes(env) * 60)
    repo = repository_from_env(env)
    _t0 = time.perf_counter()
    _base = ensure_base(repo)
    _base_s = time.perf_counter() - _t0
    run = start_run(repo)
    print(f"Run {run.id}: main {run.main} at base {run.base}")
    mo.md(f"Base snapshot `{_base}` ({_base_s:.0f} s). Run `{run.id}`, main branch `{run.main}`.")
    return budget, repo, run


@app.cell
def _():
    mo.md(r"""
    ## Methods and scoring

    Each agent tunes one method.
    One iteration of an agent calls `run_method` with the parameters that its LLM picks.
    Then it calls `score` and `write_method_output`, and it makes one commit on its branch.

    - `run_method` runs the method on the HVGs of the base. Then it builds the kNN graph and the Leiden labels.
    - `score` runs the default scib-metrics benchmark on the embedding.
    - `write_method_output` writes only the paths of the method: `obsm/X_<method>`, two `obsp` graphs, `results/<method>/labels` and `uns/<method>`.
    """)
    return


@app.class_definition
@dataclass
class MethodOutput:
    embedding: np.ndarray
    connectivities: scipy.sparse.csr_matrix
    distances: scipy.sparse.csr_matrix
    labels: pd.Categorical
    fit_s: float



@app.function
def hvg(adata: AnnData) -> AnnData:
    return adata[:, adata.var["highly_variable"]].copy()



@app.function
def pca(adata: AnnData, n_comps: int) -> np.ndarray:
    """Scaled PCA of `adata.X`, with the base settings."""
    scaled = AnnData(adata.X.toarray() if issparse(adata.X) else np.array(adata.X))
    sc.pp.scale(scaled, max_value=10)
    sc.pp.pca(scaled, n_comps=n_comps, random_state=0)
    return scaled.obsm["X_pca"]



@app.function
def harmony(work: AnnData, params: dict) -> np.ndarray:
    # The scanpy wrapper fails with harmonypy 2.1.0, so harmonypy is called directly.
    z = np.asarray(harmonypy.run_harmony(work.obsm["X_pca"], work.obs, BATCH_KEY, verbose=False, **params).Z_corr)
    return z if z.shape[0] == work.n_obs else z.T



@app.function
def combat(work: AnnData, params: dict) -> np.ndarray:
    # Some HVGs are all zero in one batch, so the first convergence check divides by a zero prior.
    # The next iteration has non-zero priors, and the output stays finite (the test checks this).
    with np.errstate(divide="ignore"):
        sc.pp.combat(work, key=BATCH_KEY)
    return pca(work, params["n_comps"])



@app.function
def scanorama(work: AnnData, params: dict) -> np.ndarray:
    sc.external.pp.scanorama_integrate(work, BATCH_KEY, basis="X_pca", verbose=0, **params)
    return work.obsm["X_scanorama"]



@app.function
def methods_bbknn(work: AnnData, params: dict) -> np.ndarray:
    # bbknn builds the graph; its 2-D UMAP is the embedding. The scanpy wrapper passes deprecated arguments.
    bbknn.bbknn(work, batch_key=BATCH_KEY, use_rep="X_pca", **params)
    sc.tl.umap(work, random_state=0)
    return work.obsm["X_umap"]



@app.function
def train_scvi(work: AnnData, max_epochs: int, **model_kwargs) -> scvi.model.SCVI:
    scvi.settings.seed = 0
    scvi.model.SCVI.setup_anndata(work, layer="counts", batch_key=BATCH_KEY)
    model = scvi.model.SCVI(work, **model_kwargs)
    train_model(model, max_epochs)
    return model



@app.function
def methods_scvi(work: AnnData, params: dict) -> np.ndarray:
    model = train_scvi(work, params["max_epochs"], n_latent=params["n_latent"], n_layers=params["n_layers"])
    return model.get_latent_representation()



@app.function
def pretrain_scvi(adata: AnnData, max_epochs: int = PRETRAIN_EPOCHS) -> scvi.model.SCVI:
    """Trains the scVI model, with scvi-tools default architecture, that scANVI starts from."""
    return train_scvi(hvg(adata), max_epochs)



@app.function
def scanvi_labels(obs: pd.DataFrame) -> pd.Categorical:
    """The training labels of scANVI: the expert labels, with "Unknown" for the cells of QUERY_BATCHES."""
    query = obs[BATCH_KEY].isin(QUERY_BATCHES).to_numpy()
    return pd.Categorical(np.where(query, "Unknown", obs[LABEL_KEY].astype(str)))


@app.function
def scanvi(work: AnnData, pretrained: scvi.model.SCVI, params: dict) -> np.ndarray:
    if pretrained is None:
        raise ValueError("scanvi needs a pretrained scVI model; call pretrain_scvi first")
    # scANVI embeds the pretrain cells, so their order must match the input cells.
    if not pretrained.adata.obs_names.equals(work.obs_names):
        raise ValueError("pretrain obs_names differ from the input obs_names; pretrain on the same cells and order")
    work.obs["scanvi_label"] = scanvi_labels(work.obs)
    model = scvi.model.SCANVI.from_scvi_model(
        pretrained, unlabeled_category="Unknown", labels_key="scanvi_label", adata=work
    )
    train_model(model, params["max_epochs"], n_samples_per_label=100)
    return model.get_latent_representation()



@app.function
def run_method(adata: AnnData, method: str, params: dict, pretrained: scvi.model.SCVI | None = None) -> MethodOutput:
    """Runs one method on the HVGs, then builds the graph and Leiden labels. `pretrained` is for scanvi only."""
    errors = validate_params(method, params)
    if errors:
        raise ValueError("; ".join(errors))
    batches = adata.obs[BATCH_KEY].to_numpy()
    if method == "scanorama" and (batches[1:] != batches[:-1]).sum() + 1 != len(set(batches)):
        raise ValueError("scanorama needs the cells of each batch to be contiguous")
    work = hvg(adata)
    embedders = {
        "unintegrated": lambda work, params: work.obsm["X_pca"][:, : params["n_comps"]],
        "harmony": harmony,
        "combat": combat,
        "scanorama": scanorama,
        "bbknn": methods_bbknn,
        "scvi": methods_scvi,
    }
    t0 = time.perf_counter()
    embedding = scanvi(work, pretrained, params) if method == "scanvi" else embedders[method](work, params)
    fit_s = time.perf_counter() - t0
    embedding = np.ascontiguousarray(embedding, dtype=np.float32)
    if method != "bbknn":
        work.obsm["X_emb"] = embedding
        sc.pp.neighbors(work, use_rep="X_emb", random_state=0)
    sc.tl.leiden(work, flavor="igraph", n_iterations=2, directed=False, random_state=0)
    return MethodOutput(
        embedding=embedding,
        connectivities=scipy.sparse.csr_matrix(work.obsp["connectivities"]),
        distances=scipy.sparse.csr_matrix(work.obsp["distances"]),
        labels=pd.Categorical(work.obs["leiden"]),
        fit_s=fit_s,
    )



@app.cell
def _():
    mo.md(r"""
    ### Scoring

    `score` uses [scib-metrics](https://github.com/YosefLab/scib-metrics), the scverse package for the scIB benchmark.
    scIB comes from Luecken et al., "Benchmarking atlas-level data integration in single-cell genomics" (Nature Methods, 2022).
    It is the usual way to compare batch-integration methods.

    The benchmark gives two scores, each the mean of a group of metrics.
    Each metric is from 0 to 1, and higher is better.

    | Score | Metric | What it measures |
    |---|---|---|
    | Bio conservation | isolated labels | Cell types that occur in few batches stay apart from the other types. |
    | | k-means NMI and ARI | k-means clusters of the embedding agree with the expert labels. |
    | | label silhouette | A cell is nearer to cells of its own type than to other types. |
    | | cLISI | The neighbors of a cell have few cell types. |
    | Batch correction | BRAS | Within each cell type, the batches mix in the embedding. |
    | | iLISI | The neighbors of a cell come from many batches. |
    | | kBET per label | Within each cell type, the batch mix of a neighborhood agrees with the batch mix of the type. |
    | | graph connectivity | The cells of one type form one connected part of the kNN graph. |
    | | PCR comparison | The batch explains less variance than in the unintegrated PCA. |

    The labels are the expert cell types in `obs["final_annotation"]`, and the batches are `obs["batch"]`.
    The base `X_pca` is the unintegrated reference for the PCR comparison.

    The 3 Oetjen batches (bone marrow, 9,581 cells) are the query batches.
    scANVI trains with the labels of the other batches only. The query cells are "Unknown" for it.
    Bio conservation scores only the query cells, for all 7 methods.
    Thus no method gets a score from labels that it trained on.
    The query batches contain all 16 cell types. Four of these types occur only there.
    Thus scANVI never sees the labels of these 4 types.
    Batch correction scores all cells.

    Total = 0.6 × Bio conservation + 0.4 × Batch correction.
    scib-metrics sets these weights. They count the biology more than the batch mix.
    A method that mixes the batches but merges cell types gets a low Total.

    `score` calls `get_results(min_max_scale=False)`, so the scores are not scaled between methods.
    Thus a score keeps its value from one iteration or run to the next.
    The tables of the scIB paper scale each metric between the methods, so their values differ from these.
    """)
    return


@app.function
def score(adata: AnnData, method: str, embedding: np.ndarray, n_jobs: int = -1) -> dict[str, float]:
    """Scores one embedding. Returns the three aggregates, `seconds` and `pcr_clipped`.

    Bio conservation uses only the cells of QUERY_BATCHES, whose labels scANVI never sees.
    Batch correction uses all cells, with the base `X_pca` as the unintegrated reference.
    Total = 0.6 * Bio conservation + 0.4 * Batch correction, the weights of scib-metrics.

    `pcr_clipped` is True when scib-metrics set a negative PCR comparison to 0; other warnings pass through.

    `n_jobs` limits the worker count when several agent processes share the CPU.
    """
    key = f"X_{method}"
    scored = AnnData(obs=adata.obs[[BATCH_KEY, LABEL_KEY]].copy(), obsm={"X_pca": adata.obsm["X_pca"], key: embedding})
    query = scored.obs[BATCH_KEY].isin(QUERY_BATCHES).to_numpy()
    t0 = time.perf_counter()
    # Full default metric set of scib-metrics, in two groups on two cell sets.
    benches = {
        "Bio conservation": Benchmarker(
            scored[query].copy(),
            batch_key=BATCH_KEY,
            label_key=LABEL_KEY,
            embedding_obsm_keys=[key],
            bio_conservation_metrics=BioConservation(),
            batch_correction_metrics=None,
            # Without this key, prepare runs a PCA of the empty X. The bio metrics do not use it.
            pre_integrated_embedding_obsm_key="X_pca",
            n_jobs=n_jobs,
            progress_bar=False,
        ),
        "Batch correction": Benchmarker(
            scored,
            batch_key=BATCH_KEY,
            label_key=LABEL_KEY,
            embedding_obsm_keys=[key],
            bio_conservation_metrics=None,
            batch_correction_metrics=BatchCorrection(),
            pre_integrated_embedding_obsm_key="X_pca",
            n_jobs=n_jobs,
            progress_bar=False,
        ),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for bench in benches.values():
            bench.benchmark()
    clipped = False
    for w in caught:
        if PCR_CLIPPED in str(w.message):
            clipped = True
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    result = {name: float(bench.get_results(min_max_scale=False).loc[key, name]) for name, bench in benches.items()}
    result["Total"] = 0.6 * result["Bio conservation"] + 0.4 * result["Batch correction"]
    return result | {"seconds": time.perf_counter() - t0, "pcr_clipped": clipped}


@app.function
def write_method_output(
    session: icechunk.Session, method: str, out: MethodOutput, params: dict, metrics: dict, reasoning: str
) -> None:
    """Writes only the paths of `method`. `iteration` counts the writes on this branch, from 1."""
    try:
        iteration = int(read_element(session, f"uns/{method}/iteration")) + 1
    except KeyError:
        iteration = 1
    write_element(session, f"obsm/X_{method}", out.embedding)
    write_element(session, f"obsp/{method}_connectivities", out.connectivities)
    write_element(session, f"obsp/{method}_distances", out.distances)
    write_element(session, f"results/{method}/labels", out.labels)
    record = {"params": params, "metrics": metrics, "reasoning": reasoning, "iteration": iteration}
    write_element(session, f"uns/{method}", record)



@app.cell
def _():
    mo.md(r"""
    ## One iteration by hand

    The function below does one iteration of an agent, with no LLM and fixed Harmony parameters.
    It creates a branch at the run base.
    On that branch, it fits Harmony and scores the embedding.
    Then it writes the method paths and makes one commit.
    It asks Icechunk for the diff against the base.
    Last, it deletes the branch. The evaluator GC later deletes the chunks of the branch.
    The iteration takes about 1 minute, and this time counts toward the time cap.
    """)
    return


@app.function
def one_iteration_by_hand(repo: icechunk.Repository, run: Run) -> dict:
    """One Harmony iteration on its own branch, with BY_HAND_PARAMS. `main` stays at the base."""
    branch = f"{run.id}-by-hand"
    repo.create_branch(branch, run.base)
    # The branch goes also on a failure, so a rerun can create it again.
    try:
        session = repo.writable_session(branch)
        adata = read_adata(session)

        out = run_method(adata, "harmony", BY_HAND_PARAMS)
        metrics = score(adata, "harmony", out.embedding)
        write_method_output(session, "harmony", out, BY_HAND_PARAMS, metrics, "by hand")
        snapshot = session.commit("harmony by hand", metadata={"params": BY_HAND_PARAMS, "metrics": metrics})

        diff = repo.diff(from_snapshot_id=run.base, to_snapshot_id=snapshot)
        changed = {
            *diff.new_arrays, *diff.new_groups, *diff.updated_arrays, *diff.updated_groups,
            *diff.deleted_arrays, *diff.deleted_groups,
        }
    finally:
        repo.delete_branch(branch)
    return {"branch": branch, "snapshot": snapshot, "metrics": metrics, "diff_paths": sorted(changed)}


@app.cell
def _(budget, repo, run):
    _skip = skip_reason(budget)
    by_hand = None if _skip else one_iteration_by_hand(repo, run)
    if by_hand:
        print(f"By hand: {by_hand['snapshot']}", {k: by_hand["metrics"][k] for k in SCORES})
    mo.md(_skip) if _skip else mo.vstack([
        mo.md(f"Commit `{by_hand['snapshot']}` on `{by_hand['branch']}` with `{BY_HAND_PARAMS}`. "
              "The branch is now deleted."),
        mo.ui.table([{k: by_hand["metrics"][k] for k in SCORES}], selection=None),
        mo.md("Paths that the commit changed against the base:"),
        mo.ui.table([{"path": p} for p in by_hand["diff_paths"]], selection=None),
        mo.md("Each agent does this loop on its own branch. The LLM picks the parameters, and the agent keeps its branch."),
    ])
    return (by_hand,)


@app.cell
def _():
    mo.md(r"""
    ## Shootout

    The cell below starts one agent process per method.
    Each agent gets a fixed parameter space and an iteration limit.
    Each iteration fits the method, scores it with scib-metrics, and makes one commit on the branch of the agent.
    The commit message holds the reasoning of the agent for that step.

    When a cap trips, all agents stop, and each branch keeps its last commit.
    A cap can stop an agent during an API call.
    The provider can bill that call, but the agent cannot report it.
    Then the shown cost is a lower bound.

    scib-metrics sets a negative PCR comparison score to 0.
    The column "PCR clipped" lists the iterations where this happened.

    The table updates while the agents run.
    """)
    return


@app.cell
def _(DRY_RUN, budget, by_hand, env, repo, run):
    # The reference to by_hand makes the agents start after the by-hand iteration ends.
    assert by_hand is None or by_hand["branch"] not in repo.list_branches()
    METHODS = list(ITERATION_LIMITS)

    def _show(progress):
        mo.output.replace(mo.ui.table(progress_rows(progress), selection=None))

    shootout = run_shootout(
        env, run, METHODS, budget,
        target=scripted_agent if DRY_RUN else run_method_agent, on_progress=_show,
    )
    _rows = agent_rows(shootout)
    for _row in _rows:
        print(_row)
    mo.output.replace(
        mo.vstack([
            mo.md(f"Shootout cost: US${budget.total_usd():.4f}." + (" Dry run: no LLM." if DRY_RUN else "")),
            mo.ui.table(_rows, selection=None),
        ])
    )
    return METHODS, shootout


@app.cell
def _(repo, shootout):
    # GC in the evaluator deletes the loser snapshots, so their manifests are read now.
    manifests = manifest_rows(repo, shootout)
    return (manifests,)


@app.cell
def _():
    mo.md(r"""
    ## Evaluator

    The evaluator reads the scores of all branches.
    For each branch, it uses the iteration with the highest Total, not only the last iteration.
    It compares bio conservation with batch correction and picks one winner.
    The functions below do the promotion steps in plain code:

    1. A lineage check. The branch must descend from the base, and `main` must still be at the base.
    2. A fast-forward of `main` to the best iteration of the winner. This is a ref update only.
    3. One commit moves the Leiden labels of the winner to `obs["leiden_integrated"]`.
    4. One small commit adds the new column to the `obs` column order.

    Then the evaluator tags `main`, deletes the other branches, and runs GC.
    GC deletes the chunks that no branch or tag can reach.
    The run does not expire snapshots, so the history of `main` stays complete.
    The clean run keeps `<run id>-main`, the winner branch and the winner tag.
    """)
    return


@app.function
def branch_scores(repo: icechunk.Repository, run: Run, methods: list[str]) -> dict[str, dict]:
    """Reads `uns/<method>` at each commit since the fork, read only. Reports the iteration with the highest Total.

    `fork` is the newest head ancestor that is on main. With no Total on the branch, the best snapshot is the head.
    """
    on_main = set(ancestry_ids(repo, repo.lookup_branch(run.main)))
    scores = {}
    for method in methods:
        branch = method_branch(run, method)
        try:
            head = repo.lookup_branch(branch)
        except icechunk.RefNotFoundError:
            scores[method] = {"branch": branch, "error": "branch not found"}
            continue
        ancestry = ancestry_ids(repo, head)
        fork = next((s for s in ancestry if s in on_main), None)
        records = {}
        for snapshot in ancestry[: ancestry.index(fork) if fork in ancestry else None]:
            try:
                records[snapshot] = plain_json(read_element(repo.readonly_session(snapshot_id=snapshot), f"uns/{method}"))
            except KeyError:
                records[snapshot] = {}
        scored = [s for s, r in records.items() if "Total" in (r.get("metrics") or {})]
        best = max(scored, key=lambda s: records[s]["metrics"]["Total"]) if scored else head
        record = records.get(best, {})
        scores[method] = {
            "branch": branch,
            "head": head,
            "fork": fork,
            "iterations": records.get(head, {}).get("iteration", 0),
            "best_snapshot": best,
            "best_iteration": record.get("iteration", 0),
            "metrics": record.get("metrics"),
            "params": record.get("params"),
        }
    return scores



@app.function
def check_lineage(repo: icechunk.Repository, run: Run, branch: str) -> str | None:
    """None if `branch` descends from `run.base` and `run.main` still points to `run.base`. Else the reason."""
    head = repo.lookup_branch(branch)
    if run.base not in ancestry_ids(repo, head):
        return f"{branch} at {head} does not descend from the run base {run.base}"
    main = repo.lookup_branch(run.main)
    if main != run.base:
        return f"{run.main} moved from the fork snapshot {run.base} to {main}"
    return None



@app.function
def append_column(repo: icechunk.Repository, branch: str) -> str:
    session = repo.writable_session(branch)
    obs = zarr.open_group(session.store, mode="a")["obs"]
    order = list(obs.attrs["column-order"])
    if PROMOTED_COLUMN not in order:
        obs.attrs["column-order"] = [*order, PROMOTED_COLUMN]
    return session.commit(f"Add {PROMOTED_COLUMN} to obs column-order")



@app.function
def promote(repo: icechunk.Repository, run: Run, method: str) -> dict:
    """Fast-forwards `run.main` to the best iteration of the method, moves its labels to `obs`, then lists the new column.

    The best iteration is the commit with the highest Total on the method branch.

    Raises `PromotionRefused` if the lineage check or the compare-and-swap reset fails.
    On a later failure it resets `run.main` to `run.base` and re-raises.
    """
    branch = method_branch(run, method)
    reason = check_lineage(repo, run, branch)
    if reason:
        raise PromotionRefused(reason)
    best = branch_scores(repo, run, [method])[method]
    head, target = best["head"], best["best_snapshot"]
    timings = {}

    t0 = time.perf_counter()
    try:
        repo.reset_branch(run.main, target, from_snapshot_id=run.base)
    except icechunk.ConflictError as error:
        raise PromotionRefused(f"{run.main} moved during promotion: {error}") from error
    timings["fast_forward_s"] = time.perf_counter() - t0

    # The last snapshot that this promotion put on main; the rollback is a compare-and-swap on it.
    tip = target
    try:
        t0 = time.perf_counter()
        rearrange = repo.rearrange_session(run.main)
        rearrange.move(f"/results/{method}/labels", f"/obs/{PROMOTED_COLUMN}")
        move_snapshot = tip = rearrange.commit(f"Promote {method}: move its labels to obs/{PROMOTED_COLUMN}")
        timings["move_s"] = time.perf_counter() - t0

        # Runs only after the move commit; read_adata hides an obs column that column-order does not list.
        t0 = time.perf_counter()
        attempts = 1
        try:
            attr_snapshot = append_column(repo, run.main)
        except icechunk.ConflictError:
            attempts = 2
            attr_snapshot = append_column(repo, run.main)
        timings["attr_s"] = time.perf_counter() - t0
    except Exception as failure:
        try:
            repo.reset_branch(run.main, run.base, from_snapshot_id=tip)
        except icechunk.ConflictError as rollback:
            raise RuntimeError(
                f"promotion of {method} failed ({failure!r}) and the rollback of {run.main} to {run.base} "
                f"failed ({rollback!r})"
            ) from failure
        raise
    timings["total_s"] = sum(timings.values())

    return {
        "method": method,
        "base": run.base,
        "head": head,
        "promoted": target,
        "iteration": best["best_iteration"],
        "move_snapshot": move_snapshot,
        "attr_snapshot": attr_snapshot,
        "attr_attempts": attempts,
        "timings_s": timings,
    }



@app.function
def tag_winner(repo: icechunk.Repository, run: Run, method: str) -> str:
    """Tags the tip of `run.main`. Refuses if no commit of the method branch after the base is on `run.main`."""
    head = repo.lookup_branch(method_branch(run, method))
    main = repo.lookup_branch(run.main)
    on_main = set(ancestry_ids(repo, main))
    promoted = next((s for s in ancestry_ids(repo, head) if s in on_main), None)
    if promoted in (None, run.base):
        raise PromotionRefused(f"{method} is not promoted: no commit of its branch is on {run.main}")
    tag = f"{run.id}-winner-{method}"
    repo.create_tag(tag, main)
    return tag



@app.function
def discard(repo: icechunk.Repository, run: Run, methods: list[str]) -> None:
    """Deletes the method branches. GC can then reclaim the chunks that only they reference."""
    for method in methods:
        repo.delete_branch(method_branch(run, method))



@app.function
def collect_garbage(repo: icechunk.Repository, open_writers: int) -> dict:
    """Deletes the objects older than now that no branch or tag can reach; history stays. Refuses with an open writer.

    A cutoff of now deletes the uncommitted chunks of any open writer, also of another notebook
    on the same repo. Do not run two notebooks on one repo at the same time.
    """
    if open_writers > 0:
        raise RuntimeError(f"{open_writers} writer sessions are open; GC with a cutoff of now deletes their chunks")
    summary = repo.garbage_collect(datetime.datetime.now(datetime.timezone.utc))
    fields = {f: getattr(summary, f) for f in GC_FIELDS}
    fields["skipped_phases"] = [str(p) for p in fields["skipped_phases"]]
    fields["delete_errors"] = [str(e) for e in fields["delete_errors"]]
    return fields



@app.cell
def _(DRY_RUN, METHODS, budget, env, manifests, run):
    # GC below deletes the loser snapshots. The reference to manifests makes this cell run after their read.
    assert manifests is not None
    _t0 = time.perf_counter()
    evaluator = (scripted_evaluator if DRY_RUN else run_evaluator)(env, run, METHODS, budget)
    evaluator_s = time.perf_counter() - _t0
    _rows = evaluator_rows(evaluator, evaluator_s)
    for _row in _rows:
        print(_row)
    mo.vstack([
        mo.md("### Report of the evaluator\n\n" + (evaluator["report"] or "_No report._")),
        mo.ui.table(_rows, selection=None),
    ])
    return evaluator, evaluator_s


@app.cell
def _():
    mo.md(r"""
    ## Result on `main`

    The cell below reads `main` as one AnnData.
    It compares the promoted Leiden labels with the ground-truth cell types, and it shows the UMAP of the winner.
    """)
    return


@app.cell
def _(budget, evaluator, repo, run):
    _skip = skip_reason(budget)
    if _skip:
        print(_skip)
    mo.stop(bool(_skip), mo.md(_skip or ""))
    mo.stop(evaluator["winner"] is None, mo.md("The evaluator promoted no winner, so `main` holds only the base."))
    _winner = evaluator["winner"]
    _adata = read_adata(repo.readonly_session(branch=run.main))
    _agreement, _crosstab = label_agreement(_adata)
    _fig = umap_figure(
        umap_coords(_adata, _winner), _adata.obs, [LABEL_KEY, PROMOTED_COLUMN], f"UMAP of X_{_winner}"
    )
    print(f"Winner {_winner}: ARI {_agreement['ARI']:.3f}, NMI {_agreement['NMI']:.3f}")
    mo.vstack([
        mo.md(f"Winner: **{_winner}**. ARI {_agreement['ARI']:.3f}, NMI {_agreement['NMI']:.3f} "
              f"of `obs[\"{PROMOTED_COLUMN}\"]` against `obs[\"{LABEL_KEY}\"]`."),
        _fig,
        mo.ui.table(_crosstab.reset_index(), selection=None, page_size=16),
    ])
    return


@app.cell
def _():
    mo.md(r"""
    ## Failure scenarios

    Each scenario uses its own run in the same repo.
    Thus the clean result on `main` stays as it is.
    Each scenario uses one real Harmony agent with 1 iteration.
    A scenario does not start when a cap has tripped.

    After a scenario records its result, it deletes its own branches and tags.
    No GC runs after the scenarios in the same run.
    The evaluator GC of the next notebook run reclaims the scenario chunks. This includes the 200 MB re-QC copy of scenario 2.

    The sidecar baseline models a careful scverse user, not a weak one.
    It writes each file to a temp path and then renames it into place.
    It stores `obs_names` next to each result and checks them on load.
    Its promotion copies the files to a staging folder and renames the folder.
    The baseline still has real gaps.
    It has no history, no branch or rollback, and no safety for concurrent writers.
    It also has no lineage check against the base.
    The sidecar files stay on local disk, because these failure modes do not depend on the storage medium.
    Each scenario writes them to a temporary folder and deletes the folder after it records its result.
    """)
    return


@app.cell
def _(DRY_RUN, budget, evaluator, repo):
    # The scenarios run after the evaluator GC. The reference to evaluator makes this cell run after it.
    assert evaluator is not None
    produce = scripted_agent if DRY_RUN else run_method_agent

    def run_scenario(name, scenario, rows):
        # Runs one scenario in isolation and shows its side-by-side table and the refs it deleted.
        result = run_isolated(repo, budget, scenario)
        print(f"Scenario {name}:", result)
        deleted = ", ".join(result.get("deleted_refs", [])) or "none"
        return result, mo.vstack([
            mo.ui.table(scenario_rows(result, rows), selection=None),
            mo.md(f"Refs deleted after the scenario: {deleted}."),
        ])

    return produce, run_scenario


@app.cell
def _():
    mo.md(r"""
    ### 1. Kill an agent during a write

    The parent process commits one Harmony result to the branch first. This commit is the seed.
    Then a Harmony agent starts on the same branch.
    The agent writes its chunks, and the scenario kills it with SIGKILL before its commit.
    The sidecar gets the same two results.
    Its writer gets SIGKILL before the rename of `embedding.npy`.

    This kill point is the best case for the sidecar: no rename happened, so the old result stays complete.
    A kill after the first rename gives a mix of old and new files.
    On Icechunk, the tip is the seed or the complete new commit, never a mix.
    The chunks of the killed write stay in storage until the evaluator GC of the next notebook run.
    """)
    return


@app.cell
def _(budget, env, produce, repo, run_scenario):
    scenario_1, _view = run_scenario(
        "1", lambda _run, _store: kill_mid_write(env, _run, _store, budget, produce, repo), kill_rows
    )
    _note = kill_note(scenario_1)
    if _note:
        print(_note)
    mo.vstack([_view, mo.callout(mo.md(_note), kind="warn") if _note else mo.md("")])
    return (scenario_1,)


@app.cell
def _():
    mo.md(r"""
    ### 2. Change `main` after the fork

    A Harmony agent forks from the base.
    Before its commit, a re-QC on the `main` of this run drops the cells in the top 1% of `n_counts`.
    The re-QC writes a full copy of the AnnData, about 200 MB, to `main`.
    Then each store tries to use the result of the agent.

    Icechunk refuses the promotion, because `main` moved after the fork.
    It refuses for any commit to `main`, not only for a change of cells.
    The careful sidecar user checks `obs_names` on load.
    That check detects only a change of `obs_names`.
    A change of values on the same cells passes it.
    """)
    return


@app.cell
def _(budget, env, produce, repo, run_scenario):
    scenario_2, _view = run_scenario(
        "2", lambda _run, _store: mutate_base_after_fork(env, _run, _store, budget, produce, repo), fork_rows
    )
    _view
    return (scenario_2,)


@app.cell
def _():
    mo.md(r"""
    ### 3. Promote and discard as named operations

    The parent process commits the unintegrated PCA result on one branch.
    A Harmony agent commits its result on a second branch.
    Then the scenario promotes Harmony and discards the unintegrated branch.
    On Icechunk, promote, tag and discard are named operations.
    The commit log and the operations log record each operation with its time.
    On the sidecar, promote and discard are file operations.
    A directory listing shows the result, but no record tells what happened or when.
    """)
    return


@app.cell
def _(budget, env, produce, repo, run_scenario):
    scenario_3, _view = run_scenario(
        "3", lambda _run, _store: promote_and_discard(env, _run, _store, budget, produce, repo), promote_rows
    )
    _ops = scenario_3.get("icechunk", {}).get("ops", [])
    mo.vstack([_view, mo.md("Operations log of the run, newest first:"), mo.ui.table(_ops, selection=None)])
    return (scenario_3,)


@app.cell
def _():
    mo.md(r"""
    ### 4. Ask the store what one agent changed

    A Harmony agent commits one result.
    Then it asks each store what changed since the base.
    Icechunk gives the changed paths, the commit message and the parameters from one diff.
    The sidecar gives a list of files by modification time.
    It cannot tell which data changed against the base.
    """)
    return


@app.cell
def _(budget, env, produce, repo, run_scenario):
    scenario_4, _view = run_scenario(
        "4", lambda _run, _store: diff_agent(env, _run, _store, budget, produce, repo), diff_rows
    )
    _view
    return (scenario_4,)


@app.cell
def _():
    mo.md(r"""
    ## Instrumentation

    One NB2 iteration writes about 15 MB of arrays, about 13.7 MB after compression.
    The arrays are the embedding (about 6 MB) and two sparse graphs (about 7.7 MB stored).
    The tables below show two times for each commit.
    `write_s` is the time to open the session and write the arrays.
    `commit_s` is the time of the commit call.
    The chunk uploads happen during the write, not during the commit.
    Thus `write_s` depends on the upload bandwidth of the machine.
    The commit writes one manifest for each new array, and one iteration adds 19 arrays.
    `repository_config()` sets the manifest concurrency to 32, so the commit writes the manifests in parallel.

    `step_s` is the time of one agent step.
    One step is the method fit, the graph, the score step, the write and the commit. The LLM time is not in it.
    `write_commit_share` is the write plus commit time divided by `step_s`.
    `commit_share` is the commit time alone divided by `step_s`.
    A small share means that storage does not limit the speed of the agents.

    The first table gives the median of each time and each share over all commits of the run.
    It also gives `max_commit_s`, the longest commit of the run.
    A single slow commit shows in `max_commit_s`, but not in the medians.
    The second table gives one row per commit.

    The cell also counts the commits per branch and reads the manifest files of each commit.
    It times the open of `main` at each snapshot of its history.
    It records the CPU count and the memory of the machine.
    It writes all of it to `data/nb2/nb2_run_<run id>.json`.
    """)
    return


@app.cell
def _(
    DRY_RUN,
    budget,
    evaluator,
    evaluator_s,
    manifests,
    models,
    repo,
    run,
    scenario_1,
    scenario_2,
    scenario_3,
    scenario_4,
    shootout,
):
    _latency = latency_rows(shootout)
    _summary = latency_summary(_latency)
    _commits = commits_per_branch(shootout, lambda m: method_branch(run, m))
    _cost_time = cost_time_rows(shootout, evaluator, evaluator_s)
    _scenario_runs = {k: s.get("run_id") for k, s in {"1": scenario_1, "2": scenario_2, "3": scenario_3,
                                                       "4": scenario_4}.items()}
    scenario_refs_left = run_refs(repo, [i for i in _scenario_runs.values() if i])
    _skip = skip_reason(budget)
    _winner = evaluator["winner"]
    if _skip:
        print(_skip, "No open times or stored bytes.")
        _open_main, base_bytes, result_bytes, n_obs = None, None, None, None
    else:
        _open_main = open_times(repo, run.main)
        _base_session = repo.readonly_session(snapshot_id=run.base)
        base_bytes = stored_bytes(_base_session)
        n_obs = cell_count(_base_session)
        result_bytes = (
            stored_bytes(
                repo.readonly_session(snapshot_id=evaluator["promotion"]["promoted"]), method_prefixes(_winner)
            )
            if _winner
            else None
        )
    _record = run_record(
        run,
        dry_run=DRY_RUN,
        models=None if DRY_RUN else models,
        total_cost_usd=budget.total_usd(),
        wall_s=budget.elapsed_s(),
        latency=_latency,
        latency_summary=_summary,
        commits_per_branch=_commits,
        manifests=manifests,
        open_main=_open_main,
        n_obs=n_obs,
        base_bytes=base_bytes,
        winner=_winner,
        winner_result_bytes=result_bytes,
        cost_time=_cost_time,
        shootout=shootout,
        evaluator=evaluator,
        scenarios={"1": scenario_1, "2": scenario_2, "3": scenario_3, "4": scenario_4},
        scenario_refs_left=scenario_refs_left,
    )
    _path = write_record(DATA_DIR / "nb2" / f"nb2_run_{run.id}.json", _record)
    print("Latency summary:", _summary)
    print("Scenario refs left:", scenario_refs_left or "none")
    print(f"Total cost US${budget.total_usd():.4f}, wall time {budget.elapsed_s():.0f} s. Record: {_path}")
    mo.vstack([
        mo.md(
            f"Record: `{_path}`. Total cost of the notebook: US${budget.total_usd():.4f}. "
            f"Wall time since the budget start: {budget.elapsed_s() / 60:.1f} minutes."
            + (" Dry run: no LLM." if DRY_RUN else "")
        ),
        mo.md(_skip or ""),
        mo.md("**Write and commit time**"),
        mo.ui.table([_summary], selection=None),
        mo.ui.table(_latency, selection=None),
        mo.md("**Commits per branch**"),
        mo.ui.table(_commits, selection=None),
        mo.md("**Manifest files per commit**"),
        mo.ui.table(manifests, selection=None),
        mo.md("**Time to open `main` and read `obs`, by snapshot depth**"),
        mo.ui.table(_open_main or [], selection=None),
        mo.md("**Cost and time per agent**"),
        mo.ui.table(_cost_time, selection=None),
        mo.md(f"Scenario refs left in the repo: {', '.join(scenario_refs_left) or 'none'}."),
    ])
    return base_bytes, n_obs, result_bytes


@app.cell
def _():
    mo.md(r"""
    ## Projection to 1M cells

    The cell below uses the stored bytes of the base and of one result of the winner in this run.
    It divides both by the cell count, and it projects the sizes to 1M cells.
    A branch per result stores the base one time and adds only the method paths.
    A full copy per result stores the base again for each result.
    """)
    return


@app.cell
def _(base_bytes, n_obs, result_bytes, shootout):
    mo.stop(result_bytes is None, mo.md("No promoted result or no measured bytes, so no projection."))
    _iterations = sum(_r["iterations"] for _r in shootout.values())
    _p = projection(n_obs, base_bytes, result_bytes, _iterations)
    print("Projection:", _p)
    mo.md(
        f"""
    Measured: {base_bytes / 1e6:.0f} MB for the base and {result_bytes / 1e6:.1f} MB for one result,
    at {n_obs:,} cells. This run made {_iterations} results.

    At {_p['cells_projected']:,} cells and {_iterations} results:

    - branches: {_p['branch_bytes_projected'] / 1e9:.1f} GB;
    - full copies: {_p['copy_bytes_projected'] / 1e9:.1f} GB.
    """
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## Limits

    - Bio conservation uses only the 9,581 cells of the 3 Oetjen batches. Thus it has less data than Batch correction.
      Totals from runs before this split are not comparable with Totals from later runs.
    - BBKNN gives a graph, not an embedding.
      Its embedding here is the 2-D UMAP of that graph, so its scIB scores use 2 dimensions and not 10 to 50.
      With some `trim` values the BBKNN graph splits into parts.
      UMAP then places the parts with a spectral layout, and sklearn warns that the graph is not fully connected.
    - All agents run on one machine and share its CPUs.
      Thus the step times depend on the machine and on the other agents.
    - scib-metrics 0.6.1 can print `RuntimeWarning: overflow encountered in divide` from `_diffusion_nn.py`.
      Its kBET metric computes a diffusion map in float32.
      A cell with very weak edges then gets an infinite inverse degree.
      That cell and its neighbor then get wrong kBET neighbors. The effect on Batch correction is small.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Appendix: code

    The cells below hold the code that the notebook calls.
    The setup cell at the top holds the imports and the constants.
    """)
    return


@app.class_definition(hide_code=True)
class RepositoryAccessError(RuntimeError):
    pass


@app.function(hide_code=True)
def repository_config() -> icechunk.RepositoryConfig:
    config = icechunk.RepositoryConfig.default()
    config.manifest = icechunk.ManifestConfig(max_concurrent_manifest_fetches_during_commit=COMMIT_MANIFEST_CONCURRENCY)
    return config


@app.function(hide_code=True)
def arraylake_repository(org: str, name: str) -> icechunk.Repository:
    try:
        return Client().create_repo(f"{org}/{name}", config=repository_config())
    except AuthException as error:
        raise RepositoryAccessError(
            f"Not authenticated to Arraylake org {org}. Run the Arraylake login cell, or set ARRAYLAKE_TOKEN."
        ) from error


@app.function(hide_code=True)
def ensure_arraylake_login(env: Mapping[str, str]) -> bool:
    """Logs in with the device-code flow if the org rejects the cached credentials. True if it logged in."""
    if store_kind(env) != "arraylake" or env.get("ARRAYLAKE_TOKEN"):
        return False
    client = al.Client()

    def check_access() -> None:
        # One request for the first repo page is the cheapest authenticated read.
        next(iter(client.list_repos_paginated(env["ICB_ARRAYLAKE_ORG"], page_size=1)), None)

    try:
        check_access()
    except ArraylakeClientError as error:
        # An invalid cached refresh token gives 401 or 403.
        if error.status_code not in (401, 403):
            raise
    except AuthException:
        pass
    else:
        return False
    client.login()
    check_access()
    return True


@app.function(hide_code=True)
def fresh_name(prefix: str) -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{secrets.token_hex(2)}"


@app.function(hide_code=True)
def store_kind(env: Mapping[str, str]) -> str:
    return env.get("ICB_STORE", "arraylake")


@app.function(hide_code=True)
def llm_provider(env: Mapping[str, str]) -> str:
    """The LLM provider from `ICB_LLM`; OpenAI is the default."""
    provider = env.get("ICB_LLM", "openai")
    if provider not in API_KEY_BY_PROVIDER:
        raise ValueError(f"ICB_LLM must be one of {sorted(API_KEY_BY_PROVIDER)}, not {provider!r}")
    return provider


@app.function(hide_code=True)
def training_accelerator(env: Mapping[str, str]) -> str:
    """The lightning accelerator for scvi and scanvi from `ICB_ACCELERATOR`; CPU is the default."""
    accelerator = env.get("ICB_ACCELERATOR", "cpu")
    if accelerator not in ACCELERATORS:
        raise ValueError(f"ICB_ACCELERATOR must be one of {list(ACCELERATORS)}, not {accelerator!r}")
    return accelerator


@app.function(hide_code=True)
def iteration_limits(env: Mapping[str, str]) -> dict[str, int]:
    """Iteration limits per agent. `ICB_ITERATION_SCALE` multiplies each one, but unintegrated has one parameter and stays at 1."""
    value = env.get("ICB_ITERATION_SCALE", "1")
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"ICB_ITERATION_SCALE must be a whole number of 1 or more, not {value!r}")
    return {m: n if m == "unintegrated" else n * int(value) for m, n in ITERATION_LIMITS.items()}


@app.function(hide_code=True)
def max_minutes(env: Mapping[str, str]) -> float:
    """The wall-clock cap of the notebook in minutes, from `ICB_MAX_MINUTES`; 90 is the default."""
    value = env.get("ICB_MAX_MINUTES", "90")
    try:
        minutes = float(value)
    except ValueError:
        minutes = 0.0
    if not minutes > 0:
        raise ValueError(f"ICB_MAX_MINUTES must be a number of minutes above 0, not {value!r}")
    return minutes


@app.function(hide_code=True)
def missing_settings(env: Mapping[str, str]) -> list[str]:
    """Sorted names of unset variables for the chosen store, plus the API key of the chosen LLM provider."""
    store = store_kind(env)
    if store not in REQUIRED_BY_STORE:
        raise ValueError(f"ICB_STORE must be one of {sorted(REQUIRED_BY_STORE)}, not {store!r}")
    training_accelerator(env)
    iteration_limits(env)
    max_minutes(env)
    required = (API_KEY_BY_PROVIDER[llm_provider(env)], *REQUIRED_BY_STORE[store])
    return sorted(name for name in required if not env.get(name))


@app.function(hide_code=True)
def repository_from_env(env: Mapping[str, str] = os.environ) -> icechunk.Repository:
    """Opens the NB2 repository of the reader. It is created on first use and reused later."""
    if store_kind(env) == "arraylake":
        try:
            return Client().get_or_create_repo(
                f"{env['ICB_ARRAYLAKE_ORG']}/icechunk-bio-nb2", config=repository_config()
            )
        except AuthException as error:
            raise RepositoryAccessError(
                f"Not authenticated to Arraylake org {env['ICB_ARRAYLAKE_ORG']}. Run the Arraylake login cell, or set ARRAYLAKE_TOKEN."
            ) from error
    storage = icechunk.s3_storage(
        bucket=env["ICB_S3_BUCKET"], prefix=env["ICB_S3_PREFIX"], region=env["ICB_S3_REGION"], from_env=True
    )
    return icechunk.Repository.open_or_create(storage, config=repository_config())


@app.function(hide_code=True)
def read_old_h5ad(path: str) -> anndata.AnnData:
    # These files predate anndata encoding metadata; each element warns.
    with warnings.catch_warnings():
        for message in (
            "Element '/.*' was written without encoding metadata",
            "'/.*' was written with a very old version of AnnData",
        ):
            warnings.filterwarnings(
                "ignore", message=message, category=anndata.OldFormatWarning
            )
        return anndata.read_h5ad(path)


@app.function(hide_code=True)
def load_immune() -> anndata.AnnData:
    """scIB human immune benchmark without the Villani batch. Batch key "batch", label key "final_annotation"."""
    path = pooch.retrieve(
        url=IMMUNE_URL,
        known_hash=IMMUNE_SHA256,
        fname=IMMUNE_FILENAME,
        path=DATA_DIR,
    )
    adata = read_old_h5ad(path)
    return adata[adata.obs["batch"] != IMMUNE_EXCLUDED_BATCH].copy()


@app.function(hide_code=True)
def anndata_store_root(session: icechunk.Session) -> zarr.Group:
    return zarr.open_group(session.store, mode="r" if session.read_only else "a")


@app.function(hide_code=True)
def write_adata(session: icechunk.Session, adata: anndata.AnnData) -> None:
    anndata.io.write_elem(anndata_store_root(session), "/", adata)


@app.function(hide_code=True)
def read_adata(session: icechunk.Session) -> anndata.AnnData:
    return anndata.io.read_elem(anndata_store_root(session))


@app.function(hide_code=True)
def write_element(session: icechunk.Session, key: str, value) -> None:
    # write_elem rejects "/" in keys, so write into the parent group, which is created if missing.
    parent, _, name = key.rpartition("/")
    root = anndata_store_root(session)
    anndata.io.write_elem(root.require_group(parent) if parent else root, name, value)


@app.function(hide_code=True)
def read_element(session: icechunk.Session, key: str):
    return anndata.io.read_elem(anndata_store_root(session)[key])


@app.function(hide_code=True)
def plain_json(value):
    # Converts numpy values to Python types so they serialize as JSON and zarr attrs.
    if isinstance(value, dict):
        return {str(k): plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@app.function(hide_code=True)
def changed_paths(
    repo: icechunk.Repository, from_snapshot: str, to_snapshot: str
) -> dict[str, list[str]]:
    d = repo.diff(from_snapshot_id=from_snapshot, to_snapshot_id=to_snapshot)
    new = {*d.new_arrays, *d.new_groups}
    deleted = {*d.deleted_arrays, *d.deleted_groups}
    # anndata write_elem deletes a node and writes it again, so the diff lists it as new and deleted.
    rewritten = new & deleted
    return {
        "new": sorted(new - rewritten),
        "rewritten": sorted(rewritten),
        "updated": sorted([*d.updated_arrays, *d.updated_groups]),
        "deleted": sorted(deleted - rewritten),
    }


@app.function(hide_code=True)
def method_branch(run: Run, method: str) -> str:
    return f"{run.id}-{method}"


@app.function(hide_code=True)
def type_ok(kind: type, value) -> bool:
    # bool is an int subclass; a float parameter also takes an int.
    if isinstance(value, bool):
        return False
    return isinstance(value, int) if kind is int else isinstance(value, (int, float))


@app.function(hide_code=True)
def validate_params(method: str, params: dict) -> list[str]:
    """Returns one error message per problem. An empty list means the params are valid."""
    if method not in PARAM_SPACES:
        return [f"unknown method {method!r}; known: {sorted(PARAM_SPACES)}"]
    space = PARAM_SPACES[method]
    errors = [f"unknown parameter {k!r} for {method}; allowed: {sorted(space)}" for k in params if k not in space]
    for name, (kind, low, high) in space.items():
        if name not in params:
            errors.append(f"missing parameter {name!r} for {method}")
        elif not type_ok(kind, params[name]):
            errors.append(f"{name} must be {kind.__name__}, got {params[name]!r}")
        elif not low <= params[name] <= high:
            errors.append(f"{name}={params[name]} is outside [{low}, {high}]")
    return errors


@app.function(hide_code=True)
def train_model(model, max_epochs: int, **train_kwargs) -> None:
    # In-process data loading is deliberate, and the CPU is the default. Lightning's hints about them are expected.
    # lightning also calls a torch pytree API that torch deprecates. Its INFO lines and the scvi
    # "Training for N epochs" INFO line stay out of notebook output.
    loggers = [logging.getLogger(name) for name in ("lightning.pytorch", "scvi")]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="GPU available but not used")
            warnings.filterwarnings("ignore", message="The 'train_dataloader' does not have many workers")
            warnings.filterwarnings(
                "ignore", message=r"`isinstance\(treespec, LeafSpec\)` is deprecated", category=FutureWarning
            )
            model.train(max_epochs=max_epochs, accelerator=training_accelerator(os.environ), enable_progress_bar=False, **train_kwargs)
    finally:
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)


@app.function(hide_code=True)
def payload_bytes(out: MethodOutput) -> int:
    """Uncompressed bytes of the arrays that one iteration writes: embedding, both graphs and the label codes."""
    graphs = sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in (out.connectivities, out.distances))
    return out.embedding.nbytes + graphs + out.labels.codes.nbytes


@app.function(hide_code=True)
def budget_prices(model: str) -> dict:
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    if model not in warned:
        warned.add(model)
        log.warning("no price for model %r; it is charged at the highest rates in the table", model)
    return MAX_PRICES


@app.function(hide_code=True)
def tokens_cost(model: str, usage: dict) -> float:
    prices = budget_prices(model)
    cost = sum(usage[f] * prices[p] for f, p in USAGE_FIELDS.items() if usage.get(f))
    cost -= (usage.get("cached_input_tokens") or 0) * prices["input"]
    return cost / 1e6


@app.function(hide_code=True)
def cost_usd(model: str, usage: dict) -> float:
    """Return the US$ cost of one API usage object; missing or None fields count as 0.

    With `iterations` (one entry per attempt, fallback hops included), each entry costs at its own model.
    """
    iterations = usage.get("iterations")
    if iterations:
        return sum(tokens_cost(entry.get("model") or model, entry) for entry in iterations)
    return tokens_cost(model, usage)


@app.class_definition(hide_code=True)
class Budget:
    """Parent-process tally of cost and time. Sets `stop_event` when a cap is reached."""

    def __init__(self, max_usd: float = 5.0, max_seconds: float = 5400.0):
        self.max_usd = max_usd
        self.max_seconds = max_seconds
        self.start = time.monotonic()
        # Spawn context: agent processes use the spawn start method.
        self.stop_event = multiprocessing.get_context("spawn").Event()
        self._agents: dict[str, dict] = {}

    def add(self, agent: str, model: str, usage: dict) -> None:
        stats = self._agents.setdefault(agent, {"cost": 0.0, "calls": 0})
        stats["cost"] += cost_usd(model, usage)
        stats["calls"] += 1
        self.exceeded()

    def total_usd(self) -> float:
        return sum(s["cost"] for s in self._agents.values())

    def exceeded(self) -> str | None:
        """Return "cost" or "time" when a cap is reached, else None."""
        reason = None
        if self.total_usd() > self.max_usd:
            reason = "cost"
        elif time.monotonic() - self.start > self.max_seconds:
            reason = "time"
        if reason:
            self.stop_event.set()
        return reason

    def elapsed_s(self) -> float:
        return time.monotonic() - self.start

    def per_agent(self) -> dict[str, dict]:
        seconds = self.elapsed_s()
        return {a: {**s, "seconds": seconds} for a, s in self._agents.items()}


@app.function(hide_code=True)
def pause(prefix: str, step: str, info: dict) -> None:
    """Writes `info` to `<prefix>.<step>`, then blocks until `<prefix>.<step>.go` exists."""
    path = Path(f"{prefix}.{step}")
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(info))
    os.replace(tmp, path)
    go = Path(f"{prefix}.{step}.go")
    deadline = time.monotonic() + WAIT_S
    while not go.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"no release of checkpoint {path} after {WAIT_S} s")
        time.sleep(POLL_S)


@app.function(hide_code=True)
def release(prefix: str, step: str) -> None:
    Path(f"{prefix}.{step}.go").touch()


@app.function(hide_code=True)
def pause_at_write(prefix: str, iteration: int, params: dict) -> None:
    """The method-tool checkpoint: the chunks of `iteration` are written and its commit is not."""
    pause(prefix, str(iteration), {"iteration": iteration, "params": params})


@app.class_definition(hide_code=True)
@dataclasses.dataclass(frozen=True)
class Tool:
    """One function tool. `parameters` is a JSON schema; the backend sends the dict that `func` returns as JSON."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., dict]


@app.class_definition(hide_code=True)
class Refusal(RuntimeError):
    """The model refused the request."""


@app.function(hide_code=True)
def run_conversation(
    provider: str,
    model: str,
    system: str,
    user: str,
    tools: list[Tool],
    effort: str,
    stop_event,
    on_usage: Callable[[str, dict], None],
    max_turns: int,
    *,
    done: Callable[[], bool] | None = None,
    api_key: str | None = None,
) -> str:
    """Runs the tool loop and returns the final text. Raises `Refusal` when the model refuses.

    The loop ends when the model calls no tool, `stop_event` is set, `done()` is true after a tool turn,
    or `max_turns` API calls are made. `on_usage(model, usage)` runs after each API response.
    """
    if provider not in MODELS:
        raise ValueError(f"provider must be one of {sorted(MODELS)}, not {provider!r}")
    backend = {"openai": run_openai_conversation, "anthropic": run_anthropic_conversation}[provider]
    return backend(
        model, system, user, tools, effort, stop_event, on_usage, max_turns, done=done, api_key=api_key
    )


@app.function(hide_code=True)
def function_tool(tool: Tool) -> dict:
    # Not strict: strict mode limits the JSON schema keywords; the tools validate their own input.
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": False,
    }


@app.function(hide_code=True)
def usage_dict(usage) -> dict:
    """Budget keys. `input_tokens` includes the cached tokens; `output_tokens` includes the reasoning tokens."""
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.input_tokens_details.cached_tokens,
        "output_tokens": usage.output_tokens,
    }


@app.function(hide_code=True)
def call_tool(tools: dict[str, Tool], call) -> str:
    """Runs one `function_call` item. An error goes back to the model as the output, as the Anthropic runner does."""
    try:
        return json.dumps(tools[call.name].func(**json.loads(call.arguments)))
    except Exception as error:
        return json.dumps({"error": f"{type(error).__name__}: {error}"})


@app.function(hide_code=True)
def llm_openai_refusal(response) -> str | None:
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "refusal":
                    return part.refusal
    return None


@app.function(hide_code=True)
def run_openai_conversation(
    model, system, user, tools, effort, stop_event, on_usage, max_turns, *, done=None, api_key=None, create=None
) -> str:
    """See `run_conversation`. `create` is the client seam: tests pass a callable that returns SDK `Response`s."""
    create = create or openai.OpenAI(api_key=api_key).responses.create
    by_name = {tool.name: tool for tool in tools}
    request = {
        "model": model,
        # `instructions` do not carry over with `previous_response_id`, so each turn sends them.
        "instructions": system,
        "tools": [function_tool(tool) for tool in tools],
        "reasoning": {"effort": effort},
        "max_output_tokens": MAX_TOKENS,
        # `previous_response_id` needs stored responses; OpenAI keeps them for at least 30 days.
        "store": True,
        "input": [{"role": "user", "content": user}],
    }
    text = ""
    for _ in range(max_turns):
        response = create(**request)
        text = response.output_text
        # The requested model is the billed model: no server fallback, and `response.model` can be a dated snapshot.
        on_usage(model, usage_dict(response.usage) if response.usage is not None else {})
        if stop_event.is_set():
            return text
        if response.status == "failed":
            raise RuntimeError(f"response {response.id} failed: {response.error}")
        if response.status == "incomplete":
            raise RuntimeError(f"response {response.id} is incomplete: {response.incomplete_details}")
        refusal = llm_openai_refusal(response)
        if refusal is not None:
            raise Refusal(refusal)
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return text
        # Server-side state: the server keeps earlier items and reasoning, so each turn sends only the tool outputs.
        request["previous_response_id"] = response.id
        request["input"] = [
            {"type": "function_call_output", "call_id": call.call_id, "output": call_tool(by_name, call)}
            for call in calls
        ]
        if (done and done()) or stop_event.is_set():
            return text
    return text


@app.function(hide_code=True)
def runner_tool(tool: Tool):
    """Wraps `tool` for the SDK tool runner. The runner validates the input against the signature of `tool.func`."""
    @functools.wraps(tool.func)
    def call(**kwargs) -> str:
        return json.dumps(tool.func(**kwargs))

    return beta_tool(call, name=tool.name, description=tool.description, input_schema=tool.parameters)


@app.function(hide_code=True)
def llm_anthropic_text(message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


@app.function(hide_code=True)
def run_anthropic_conversation(
    model, system, user, tools, effort, stop_event, on_usage, max_turns, *, done=None, api_key=None, client=None
) -> str:
    """See `run_conversation`. `client` is an `anthropic.Anthropic`; the default uses `api_key`."""
    client = client or anthropic.Anthropic(api_key=api_key)
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=MAX_TOKENS,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        # Explicit marker caches tools + system; top-level auto caching covers the growing conversation.
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        cache_control={"type": "ephemeral"},
        tools=[runner_tool(tool) for tool in tools],
        messages=[{"role": "user", "content": user}],
        max_iterations=max_turns,
    )
    text = ""
    for message in runner:
        text = llm_anthropic_text(message)
        # The full dump keeps `usage.iterations`, so the budget prices each fallback hop.
        on_usage(message.model, message.usage.model_dump())
        if stop_event.is_set():
            return text
        if message.stop_reason == "refusal":
            raise Refusal(text)
        # Runs this turn's tools now; the runner reuses the cached result for the next request.
        runner.generate_tool_call_response()
        if (done and done()) or stop_event.is_set():
            return text
    return text


@app.function(hide_code=True)
def threads_per_agent(n_agents: int) -> int:
    return max(1, (os.cpu_count() or 1) // n_agents)


@app.function(hide_code=True)
def ensure_method_branch(repo: icechunk.Repository, run: Run, method: str) -> str:
    """Creates the method branch at the run base if it is missing. Returns the branch name."""
    branch = method_branch(run, method)
    if branch not in repo.list_branches():
        repo.create_branch(branch, run.base)
    return branch


@app.class_definition(hide_code=True)
class MethodTools:
    """The agent tools as plain callables. Each `run_and_score` that succeeds is one commit."""

    def __init__(
        self,
        repo: icechunk.Repository,
        run: Run,
        method: str,
        adata,
        stop_event,
        max_iterations: int,
        n_jobs: int = -1,
        pretrained=None,
        report=None,
        checkpoint=None,
    ):
        self.repo, self.method, self.adata = repo, method, adata
        self.branch = method_branch(run, method)
        self.stop_event, self.max_iterations = stop_event, max_iterations
        self.n_jobs, self.pretrained = n_jobs, pretrained
        self.report = report or (lambda item: None)
        self.checkpoint = checkpoint
        self.iterations = 0
        self.last_snapshot: str | None = None
        self.summary: str | None = None

    def _refusal(self) -> str | None:
        if self.stop_event.is_set():
            return "stopped: the run budget is used up; call finish"
        if self.iterations >= self.max_iterations:
            return f"iteration limit {self.max_iterations} reached; call finish"
        return None

    def run_and_score(self, params: dict, reasoning: str) -> dict:
        refusal = self._refusal()
        if refusal:
            return {"error": refusal}
        errors = validate_params(self.method, params)
        if errors:
            return {"error": "; ".join(errors)}
        step_start = time.perf_counter()
        try:
            out = run_method(self.adata, self.method, params, pretrained=self.pretrained)
            metrics = score(self.adata, self.method, out.embedding, n_jobs=self.n_jobs)
        except Exception as error:  # a fit can fail for some params; the agent sees the error and continues
            return {"error": f"{type(error).__name__}: {error}"}
        metrics["fit_seconds"] = out.fit_s
        # The cap can trip during the fit; the branch tip then stays at the last commit.
        refusal = self._refusal()
        if refusal:
            return {"error": refusal}
        iteration = self.iterations + 1
        t0 = time.perf_counter()
        session = self.repo.writable_session(self.branch)
        write_method_output(session, self.method, out, params, metrics, reasoning)
        write_s = time.perf_counter() - t0
        wait_start = time.perf_counter()
        if self.checkpoint:
            self.checkpoint(iteration, params)
        t0 = time.perf_counter()
        wait_s = t0 - wait_start
        snapshot = session.commit(
            f"{self.method} iter {iteration}: {reasoning}",
            metadata={"params": params, "metrics": metrics, "iteration": iteration},
        )
        commit_s = time.perf_counter() - t0
        committed_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        self.iterations, self.last_snapshot = iteration, snapshot
        # One step is fit, graph, score, write and commit. It excludes the checkpoint wait.
        step_s = time.perf_counter() - step_start - wait_s
        self.report({
            "agent": self.method,
            "commit": {
                "iteration": iteration, "snapshot": snapshot, "total": metrics.get("Total"),
                "pcr_clipped": metrics.get("pcr_clipped"), "payload_bytes": payload_bytes(out),
                "write_s": write_s, "commit_s": commit_s, "write_commit_s": write_s + commit_s, "step_s": step_s,
                "committed_at": committed_at,
            },
        })
        return {**metrics, "iteration": iteration, "snapshot": snapshot}

    def finish(self, summary: str) -> dict:
        self.summary = summary
        return {"finished": True, "iterations": self.iterations, "snapshot": self.last_snapshot}


@app.function(hide_code=True)
def write_checkpoint(env: dict):
    """The `MethodTools` checkpoint that `env[WRITE_SIGNAL_VAR]` asks for, or None."""
    prefix = env.get(WRITE_SIGNAL_VAR)
    return functools.partial(pause_at_write, prefix) if prefix else None


@app.function(hide_code=True)
def params_schema(method: str) -> dict:
    kinds = {int: "integer", float: "number"}
    return {
        "type": "object",
        "properties": {
            name: {"type": kinds[kind], "minimum": low, "maximum": high}
            for name, (kind, low, high) in PARAM_SPACES[method].items()
        },
        "required": list(PARAM_SPACES[method]),
        "additionalProperties": False,
    }


@app.function(hide_code=True)
def runner_tools(tools: MethodTools) -> list[Tool]:
    """The plain tool callables with their JSON schemas, for `run_conversation`."""
    run_schema = {
        "type": "object",
        "properties": {
            "params": params_schema(tools.method),
            "reasoning": {"type": "string", "description": "One sentence: why these values. It is the commit message."},
        },
        "required": ["params", "reasoning"],
        "additionalProperties": False,
    }
    finish_schema = {
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "Best params and their Total."}},
        "required": ["summary"],
        "additionalProperties": False,
    }
    return [
        Tool(
            "run_and_score",
            f"Run {tools.method} with params, score it with scIB, and commit the result.",
            run_schema,
            tools.run_and_score,
        ),
        Tool("finish", "End the tuning.", finish_schema, tools.finish),
    ]


@app.function(hide_code=True)
def agent_exit_reason(refused: bool, summary: str | None, stopped: bool, turns: int, max_turns: int) -> str:
    """The exit reason of one agent conversation. Earlier checks win."""
    if refused:
        return "refusal"
    if summary is not None:
        return "finished"
    if stopped:
        return "stopped"
    return "turn_limit" if turns >= max_turns else "end_turn"


@app.function(hide_code=True)
def converse(env: dict, tools: MethodTools, usage_queue, stop_event) -> str:
    """Runs the tool loop until the agent finishes or a stop condition holds. Returns the exit reason."""
    provider = llm_provider(env)
    # Each successful iteration needs one turn; the extra turns cover rejected params and finish.
    max_turns = 2 * tools.max_iterations + 4
    turns = 0

    def on_usage(model: str, usage: dict) -> None:
        nonlocal turns
        turns += 1
        usage_queue.put({"agent": tools.method, "model": model, "usage": usage})

    refused = False
    try:
        run_conversation(
            provider,
            MODELS[provider]["method"],
            method_system_prompt(tools.method, tools.max_iterations),
            f"Tune {tools.method}. Begin with your first run_and_score call.",
            runner_tools(tools),
            "low",
            stop_event,
            on_usage,
            max_turns,
            done=lambda: tools.summary is not None,
            api_key=env.get(API_KEY_BY_PROVIDER[provider]),
        )
    except Refusal:
        refused = True
    return agent_exit_reason(refused, tools.summary, stop_event.is_set(), turns, max_turns)


@app.function(hide_code=True)
def run_method_agent(
    env: dict, run: Run, method: str, usage_queue, stop_event, max_iterations: int, n_threads: int | None = None
) -> None:
    """Process entry point of one method agent. Puts usage, commit and exit reports on `usage_queue`.

    Each run has exactly one agent per method branch; the commit iteration and `uns/<method>/iteration` assume it.
    """
    tools = None
    reason = "error"
    detail = None
    try:
        if n_threads:
            torch.set_num_threads(n_threads)
        repo = repository_from_env(env)
        adata = read_adata(repo.readonly_session(snapshot_id=run.base))
        ensure_method_branch(repo, run, method)
        pretrained = pretrain_scvi(adata) if method == "scanvi" else None
        tools = MethodTools(
            repo, run, method, adata, stop_event, max_iterations,
            n_jobs=n_threads or -1, pretrained=pretrained, report=usage_queue.put,
            checkpoint=write_checkpoint(env),
        )
        reason = converse(env, tools, usage_queue, stop_event)
    except Exception as exc:
        detail = repr(exc)
        raise
    finally:
        usage_queue.put({
            "agent": method,
            "exit": reason,
            "detail": detail,
            "iterations": tools.iterations if tools else 0,
            "snapshot": tools.last_snapshot if tools else None,
        })


@app.function(hide_code=True)
@contextlib.contextmanager
def thread_env(n_threads: int):
    """Sets the thread variables in this process while children start; spawn children copy them."""
    saved = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
    os.environ.update({name: str(n_threads) for name in THREAD_ENV_VARS})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@app.function(hide_code=True)
def supervise(
    procs: dict, usage_queue, budget: Budget, poll_s: float = 1.0, grace_s: float = STOP_GRACE_S, on_progress=None
) -> dict[str, dict]:
    """Drains reports into `budget` until all processes end or a cap trips; then stops the rest.

    `on_progress(rows)` runs after each poll with the iterations, last Total, cost and seconds per agent.

    The loop polls the caps on every timeout, so a hung agent still meets the time cap.
    On a stop it sets `stop_event` and drains for up to `grace_s`, so agents report their last calls.
    The in-flight call of an agent that is then terminated can be billed but not reported:
    `budget` is a lower bound of the spend.
    """
    start = time.monotonic()
    state = {
        m: {"exit": None, "detail": None, "iterations": 0, "snapshot": None, "seconds": None, "commits": []}
        for m in procs
    }
    cap = None

    def record(item: dict) -> None:
        agent = state[item["agent"]]
        if "usage" in item:
            budget.add(item["agent"], item["model"], item["usage"])
        elif "commit" in item:
            agent["iterations"] = item["commit"]["iteration"]
            agent["snapshot"] = item["commit"]["snapshot"]
            agent["commits"].append(item["commit"])
        elif "exit" in item:
            agent.update({k: item.get(k) for k in ("exit", "detail", "iterations", "snapshot")})

    def drain_once() -> None:
        try:
            record(usage_queue.get(timeout=poll_s))
        except queue.Empty:
            pass
        for m, p in procs.items():
            if not p.is_alive() and state[m]["seconds"] is None:
                state[m]["seconds"] = time.monotonic() - start

    def alive() -> list:
        return [p for p in procs.values() if p.is_alive()]

    def progress() -> None:
        if on_progress is None:
            return
        costs, now = budget.per_agent(), time.monotonic() - start
        on_progress({
            m: {"iterations": s["iterations"], "total": s["commits"][-1].get("total") if s["commits"] else None,
                "cost": costs.get(m, {}).get("cost", 0.0), "seconds": s["seconds"] if s["seconds"] is not None else now,
                "running": procs[m].is_alive()}
            for m, s in state.items()
        })

    terminated = []
    try:
        while alive():
            drain_once()
            progress()
            cap = budget.exceeded() or ("stopped" if budget.stop_event.is_set() else None)
            if cap:
                break
    finally:
        try:
            if alive():
                budget.stop_event.set()
                deadline = time.monotonic() + grace_s
                while alive() and time.monotonic() < deadline:
                    drain_once()
        finally:
            terminated = alive()
            for p in terminated:
                p.terminate()
            for p in procs.values():
                p.join(TERMINATE_GRACE_S)
                if p.is_alive():
                    p.kill()
                    p.join()
    # A process killed during a put can leave a partial message that blocks a read, so only clean exits drain.
    if not terminated:
        while True:
            try:
                record(usage_queue.get_nowait())
            except queue.Empty:
                break
    costs = budget.per_agent()
    results = {}
    for m, p in procs.items():
        s = state[m]
        exit_reason = s["exit"]
        if exit_reason in (None, "stopped"):
            exit_reason = cap or exit_reason or ("crashed" if p.exitcode else "unknown")
        results[m] = {
            "exit": exit_reason,
            "detail": s["detail"],
            "iterations": s["iterations"],
            "snapshot": s["snapshot"],
            "cost": costs.get(m, {}).get("cost", 0.0),
            "seconds": s["seconds"] if s["seconds"] is not None else time.monotonic() - start,
            "commits": s["commits"],
        }
    return results


@app.function(hide_code=True)
def run_shootout(
    env: dict, run: Run, methods: list[str], budget: Budget, poll_s: float = 1.0,
    target=run_method_agent, on_progress=None,
) -> dict[str, dict]:
    """Runs one `target` process per method under `budget`. `target` has the signature of `run_method_agent`.

    Returns, per agent: exit reason, iterations, last snapshot, cost, seconds, threads, and per commit
    the iteration, snapshot, Total, write plus commit seconds and step seconds.
    """
    ctx = multiprocessing.get_context("spawn")
    usage_queue = ctx.Queue()
    n_threads = threads_per_agent(len(methods))
    limits = iteration_limits(env)
    procs = {}
    with thread_env(n_threads):
        for method in methods:
            procs[method] = ctx.Process(
                target=target,
                args=(env, run, method, usage_queue, budget.stop_event, limits[method]),
                kwargs={"n_threads": n_threads},
                name=f"agent-{method}",
            )
            procs[method].start()
    results = supervise(procs, usage_queue, budget, poll_s, on_progress=on_progress)
    return {m: {**r, "threads": n_threads} for m, r in results.items()}


@app.class_definition(hide_code=True)
class PromotionRefused(RuntimeError):
    """The lineage check failed, so `main` stays as it is."""


@app.function(hide_code=True)
def ancestry_ids(repo: icechunk.Repository, snapshot_id: str) -> list[str]:
    return [s.id for s in repo.ancestry(snapshot_id=snapshot_id)]


@app.class_definition(hide_code=True)
class EvaluatorTools:
    """The evaluator tools as plain callables. They record the winner, tag, discards and GC summary."""

    def __init__(self, repo: icechunk.Repository, run: Run, methods: list[str]):
        self.repo, self.run, self.methods = repo, run, list(methods)
        self.winner: str | None = None
        self.promotion: dict | None = None
        self.refusal: str | None = None
        self.failure: str | None = None
        self.tag: str | None = None
        self.discarded: list[str] = []
        self.gc: dict | None = None

    def _unknown(self, methods: list[str]) -> dict | None:
        bad = [m for m in methods if m not in self.methods]
        return {"error": f"{', '.join(bad)} is not a method of this run: {self.methods}"} if bad else None

    def branch_scores(self) -> dict:
        return branch_scores(self.repo, self.run, self.methods)

    def check_lineage(self, method: str) -> dict:
        if error := self._unknown([method]):
            return error
        reason = check_lineage(self.repo, self.run, method_branch(self.run, method))
        return {"method": method, "ok": reason is None, "reason": reason}

    def promote(self, method: str) -> dict:
        if error := self._unknown([method]):
            return error
        if self.winner:
            return {"error": f"{self.winner} is already promoted"}
        scores = branch_scores(self.repo, self.run, [method])[method]
        if not scores.get("metrics"):
            return {"error": f"{method} has no metrics; pick a method with a result"}
        try:
            result = promote(self.repo, self.run, method)
        except PromotionRefused as refusal:
            self.refusal = str(refusal)
            return {"error": f"refused: {refusal}"}
        except Exception as failure:  # promote resets main to the base, or its error names the failed reset
            self.failure = f"{type(failure).__name__}: {failure}"
            return {"error": f"promotion failed: {self.failure}"}
        self.winner, self.promotion = method, result
        return result

    def tag_winner(self, method: str) -> dict:
        if error := self._unknown([method]):
            return error
        if method != self.winner:
            return {"error": f"{method} is not the promoted winner ({self.winner})"}
        try:
            self.tag = tag_winner(self.repo, self.run, method)
        except PromotionRefused as refusal:
            return {"error": f"refused: {refusal}"}
        return {"tag": self.tag}

    def discard(self, methods: list[str]) -> dict:
        if error := self._unknown(methods):
            return error
        # A discard before the promotion would let GC delete every result.
        if self.winner is None:
            return {"error": "promote a winner first"}
        if self.winner in methods:
            return {"error": f"{self.winner} is the winner; discard only the losers"}
        discard(self.repo, self.run, methods)
        self.discarded += methods
        return {"deleted": [method_branch(self.run, m) for m in methods]}

    def collect_garbage(self) -> dict:
        # The shootout joins or terminates every agent process before the evaluator starts.
        self.gc = collect_garbage(self.repo, open_writers=0)
        return self.gc


@app.function(hide_code=True)
def evaluator_tools(tools: EvaluatorTools) -> list[Tool]:
    """The evaluator callables with their JSON schemas, for `run_conversation`."""
    method = {"type": "string", "enum": tools.methods}

    def schema(properties: dict) -> dict:
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    return [
        Tool("branch_scores", "Read the metrics, fork and iterations of every method branch.", schema({}),
                 tools.branch_scores),
        Tool("check_lineage", "Check that the branch descends from the base and main is still at the base.",
                 schema({"method": method}), tools.check_lineage),
        Tool("promote", "Promote the winning method to main.", schema({"method": method}), tools.promote),
        Tool("tag_winner", "Tag the promoted main.", schema({"method": method}), tools.tag_winner),
        Tool("discard", "Delete the branches of the losing methods.",
                 schema({"methods": {"type": "array", "items": method}}), tools.discard),
        Tool("collect_garbage", "Delete the objects that no branch or tag can reach. Gives bytes_deleted.",
                 schema({}), tools.collect_garbage),
    ]


@app.function(hide_code=True)
def evaluator_exit_reason(refused: bool, cap: str | None, stopped: bool, turns: int, done: bool) -> str:
    """The exit reason of the evaluator conversation. Earlier checks win."""
    if refused:
        return "refused"
    if cap:
        return cap
    if stopped:
        return "stopped"
    if done:
        return "done"
    return "max_turns" if turns >= MAX_TURNS else "end_turn"


@app.function(hide_code=True)
def run_evaluator(env: dict, run: Run, methods: list[str], budget: Budget) -> dict:
    """Runs the evaluator conversation under `budget`. Returns the winner, promotion, tag, discards, GC and report.

    If a cap has tripped already, it makes no API call.
    """
    provider = llm_provider(env)
    model = MODELS[provider]["evaluator"]
    tripped = budget.exceeded() or budget.stop_event.is_set()
    tools = EvaluatorTools(None if tripped else repository_from_env(env), run, methods)
    report, refused, turns = None, False, 0

    def on_usage(m: str, usage: dict) -> None:
        nonlocal turns
        turns += 1
        budget.add(AGENT, m, usage)

    if not tripped:
        try:
            report = run_conversation(
                provider,
                model,
                EVALUATOR_SYSTEM_PROMPT,
                f"Run {run.id}. Methods: {', '.join(methods)}. Begin with branch_scores.",
                evaluator_tools(tools),
                "low",
                budget.stop_event,
                on_usage,
                MAX_TURNS,
                api_key=env.get(API_KEY_BY_PROVIDER[provider]),
            )
        except Refusal:
            refused = True
    exit_ = evaluator_exit_reason(refused, budget.exceeded(), budget.stop_event.is_set(), turns, tools.gc is not None)
    return evaluator_result(tools, model, exit_, report, budget.per_agent().get(AGENT, {}).get("cost", 0.0))


@app.function(hide_code=True)
def evaluator_result(tools: EvaluatorTools, model: str, exit_: str, report: str | None, cost: float) -> dict:
    """The result dict of an evaluator run, from the state that `tools` recorded."""
    return {
        "model": model,
        "exit": exit_,
        "winner": tools.winner,
        "promotion": tools.promotion,
        "lineage_refusal": tools.refusal,
        "promotion_failure": tools.failure,
        "tag": tools.tag,
        "discarded": tools.discarded,
        "gc": tools.gc,
        "report": report,
        "cost": cost,
    }


@app.function(hide_code=True)
def enabled(env: Mapping[str, str]) -> bool:
    return env.get(ENV_VAR) == "1"


@app.function(hide_code=True)
def dry_run_missing_settings(env: Mapping[str, str]) -> list[str]:
    """`missing_settings`, without the API key in a dry run."""
    missing = missing_settings(env)
    if enabled(env):
        missing = [name for name in missing if name not in API_KEY_BY_PROVIDER.values()]
    return missing


@app.function(hide_code=True)
def open_repo(env: Mapping[str, str]) -> icechunk.Repository:
    if env.get(LOCAL_REPO_VAR):
        return icechunk.Repository.open(icechunk.local_filesystem_storage(env[LOCAL_REPO_VAR]))
    return repository_from_env(env)


@app.function(hide_code=True)
def scripted_agent(
    env: dict, run: Run, method: str, usage_queue, stop_event, max_iterations: int, n_threads: int | None = None
) -> None:
    """Same signature and reports as `run_method_agent`. It runs PARAMS[method] and makes no API call."""
    tools = None
    reason, detail = "error", None
    try:
        if n_threads:
            import torch

            torch.set_num_threads(n_threads)
        repo = open_repo(env)
        adata = read_adata(repo.readonly_session(snapshot_id=run.base))
        ensure_method_branch(repo, run, method)
        pretrained = pretrain_scvi(adata, DRY_RUN_PRETRAIN_EPOCHS) if method == "scanvi" else None
        tools = MethodTools(
            repo, run, method, adata, stop_event, max_iterations,
            n_jobs=n_threads or -1, pretrained=pretrained, report=usage_queue.put, checkpoint=write_checkpoint(env),
        )
        reason = "finished"
        for params in PARAMS[method][:max_iterations]:
            result = tools.run_and_score(params, f"dry run, fixed params {params}")
            if "error" in result:
                if stop_event.is_set():
                    reason = "stopped"
                    break
                raise RuntimeError(result["error"])
    except Exception as exc:
        reason, detail = "error", repr(exc)
        raise
    finally:
        usage_queue.put({
            "agent": method,
            "exit": reason,
            "detail": detail,
            "iterations": tools.iterations if tools else 0,
            "snapshot": tools.last_snapshot if tools else None,
        })


@app.function(hide_code=True)
def scripted_evaluator(env: dict, run: Run, methods: list[str], budget: Budget) -> dict:
    """Same result as `run_evaluator`. It promotes the branch with the highest Total and makes no API call."""
    tripped = budget.exceeded() or ("stopped" if budget.stop_event.is_set() else None)
    if tripped:
        return evaluator_result(EvaluatorTools(None, run, methods), MODEL, tripped, None, 0.0)
    tools = EvaluatorTools(open_repo(env), run, methods)
    scores = tools.branch_scores()
    scored = {m: s["metrics"]["Total"] for m, s in scores.items() if s.get("metrics")}
    if not scored:
        return evaluator_result(tools, MODEL, "end_turn", "Dry run: no branch has metrics.", 0.0)
    winner = max(scored, key=scored.get)
    promoted = tools.promote(winner)
    if "error" in promoted:
        return evaluator_result(tools, MODEL, "end_turn", f"Dry run: {promoted['error']}", 0.0)
    tools.tag_winner(winner)
    tools.discard([m for m, s in scores.items() if m != winner and "error" not in s])
    tools.collect_garbage()
    report = (
        f"Dry run with no LLM. The script promoted {winner}, the branch with the highest Total "
        f"({scored[winner]:.3f}). It then tagged main, deleted the other branches and ran GC."
    )
    return evaluator_result(tools, MODEL, "done", report, 0.0)


@app.function(hide_code=True)
def atomic_write(path: Path, write, before_rename=None) -> None:
    """Calls `write(file)` on a temp file in the same directory, then renames it to `path`.
    `before_rename(name)` runs between the two; the kill scenario stops the writer there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as f:
        write(f)
    if before_rename:
        before_rename(path.name)
    os.replace(tmp, path)


@app.class_definition(hide_code=True)
class SidecarStore:
    """Careful sidecar files. Gaps: no history, no branch or rollback, no safe concurrent writers
    (fixed temp names), and no lineage check against the base."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, method: str) -> Path:
        return self.root / "results" / method

    def write_base(self, adata: AnnData) -> None:
        """Writes `base.h5ad` and makes it read-only."""
        path = self.root / "base.h5ad"
        # h5ad writes to a path, not a file object, so `atomic_write` does not fit.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        adata.write_h5ad(tmp)
        os.replace(tmp, path)
        path.chmod(0o444)

    def write_result(self, method: str, out: MethodOutput, obs_names, before_rename=None) -> None:
        """Writes the result files, `obs_names.parquet` last. It marks a complete result only for a first write.
        A killed overwrite leaves new files next to the old labels and `obs_names`."""
        d = self._dir(method)
        atomic_write(d / "embedding.npy", lambda f: np.save(f, out.embedding), before_rename)
        atomic_write(
            d / "connectivities.npz",
            lambda f: scipy.sparse.save_npz(f, out.connectivities),
            before_rename,
        )
        atomic_write(
            d / "distances.npz", lambda f: scipy.sparse.save_npz(f, out.distances), before_rename
        )
        atomic_write(
            d / "labels.parquet",
            lambda f: pd.DataFrame({"label": out.labels}).to_parquet(f),
            before_rename,
        )
        atomic_write(
            d / "fit_s.json", lambda f: f.write(json.dumps(out.fit_s).encode()), before_rename
        )
        atomic_write(
            d / "obs_names.parquet",
            lambda f: pd.DataFrame({"obs_name": list(obs_names)}).to_parquet(f),
            before_rename,
        )

    def read_result(self, method: str, obs_names) -> MethodOutput:
        """Raises `FileNotFoundError` for a missing file and `ValueError` for different `obs_names`."""
        d = self._dir(method)
        stored = pd.read_parquet(d / "obs_names.parquet")["obs_name"].tolist()
        if stored != list(obs_names):
            raise ValueError(
                f"stored obs_names for {method!r} differ from the requested obs_names"
            )
        return MethodOutput(
            embedding=np.load(d / "embedding.npy"),
            connectivities=scipy.sparse.load_npz(d / "connectivities.npz").tocsr(),
            distances=scipy.sparse.load_npz(d / "distances.npz").tocsr(),
            labels=pd.read_parquet(d / "labels.parquet")["label"].array,
            fit_s=json.loads((d / "fit_s.json").read_text()),
        )

    def promote(self, method: str) -> None:
        """Copies the result files to a staging directory, then renames it to `promoted/<method>`."""
        src = self._dir(method)
        if not (src / "obs_names.parquet").exists():
            raise FileNotFoundError(
                f"no complete result for {method!r}: obs_names.parquet is missing"
            )
        staging = self.root / f".promote-{method}.staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for f in src.iterdir():
            if not (f.name.startswith(".") and f.name.endswith(".tmp")):
                shutil.copy2(f, staging / f.name)
        target = self.root / "promoted" / method
        target.parent.mkdir(parents=True, exist_ok=True)
        old = self.root / f".promote-{method}.old"
        if target.exists():
            shutil.rmtree(old, ignore_errors=True)
            os.rename(target, old)
        os.rename(staging, target)
        shutil.rmtree(old, ignore_errors=True)

    def discard(self, method: str) -> None:
        shutil.rmtree(self._dir(method))

    def history(self) -> None:
        """Sidecar files keep no history."""
        return None


@app.class_definition(hide_code=True)
class ScenarioStopped(RuntimeError):
    """The writer ended, or a cap tripped, before the scenario reached its failure point."""


@app.class_definition(hide_code=True)
class AgentProcess:
    """One method agent in a spawn process, with write checkpoints. Its usage reports go to `budget`."""

    def __init__(self, produce, env: dict, run: Run, max_iterations: int, budget: Budget, signal_dir: str):
        self.prefix = str(Path(signal_dir) / "write")
        self.budget = budget
        self.commits: list[str] = []
        self.exit: dict | None = None
        ctx = multiprocessing.get_context("spawn")
        self.queue = ctx.Queue()
        self.proc = ctx.Process(
            target=produce,
            args=({**env, WRITE_SIGNAL_VAR: self.prefix}, run, METHOD, self.queue, budget.stop_event, max_iterations),
            name=f"scenario-{METHOD}",
        )
        self.proc.start()

    def _record(self, item: dict) -> None:
        if "usage" in item:
            self.budget.add(item["agent"], item["model"], item["usage"])
        elif "commit" in item:
            self.commits.append(item["commit"]["snapshot"])
        elif "exit" in item:
            self.exit = item

    def _drain(self, timeout: float) -> bool:
        try:
            self._record(self.queue.get(timeout=timeout))
            return True
        except queue.Empty:
            return False

    def _drain_all(self) -> None:
        while self._drain(SCENARIOS_POLL_S):
            pass

    def _poll(self) -> None:
        self._drain(SCENARIOS_POLL_S)
        cap = self.budget.exceeded() or ("stopped" if self.budget.stop_event.is_set() else None)
        if cap:
            self.kill()
            raise ScenarioStopped(f"the {cap} cap tripped")

    def await_write(self, write: int) -> dict | None:
        """Waits until the agent pauses before the commit of its write number `write`, counted from 1.
        Returns the checkpoint info, or None if the agent ended before that write."""
        path = Path(f"{self.prefix}.{write}")
        deadline = time.monotonic() + SCENARIOS_WAIT_S
        while not path.exists():
            if not self.proc.is_alive():
                self._drain_all()
                return None
            if time.monotonic() > deadline:
                self.kill()
                raise ScenarioStopped(f"no write {write} after {SCENARIOS_WAIT_S} s")
            self._poll()
        # The agent blocks in its tool, so no report is in flight after this drain.
        self._drain_all()
        return json.loads(path.read_text())

    def release(self, write: int) -> None:
        release(self.prefix, str(write))

    def wait(self) -> None:
        deadline = time.monotonic() + SCENARIOS_WAIT_S
        while self.proc.is_alive():
            if time.monotonic() > deadline:
                self.kill()
                raise ScenarioStopped(f"the agent did not end after {SCENARIOS_WAIT_S} s")
            self._poll()
        self._drain_all()

    def kill(self) -> None:
        """SIGKILL, as an out-of-memory kill or a lost machine. The queue is not read after it."""
        if self.proc.is_alive():
            os.kill(self.proc.pid, signal.SIGKILL)
        self.proc.join()


@app.function(hide_code=True)
def agent_commit(produce, env: dict, run: Run, budget: Budget) -> tuple[str, str | None]:
    """Runs a METHOD agent with 1 iteration to its end. Returns its commit and its exit reason."""
    with tempfile.TemporaryDirectory() as signals:
        agent = AgentProcess(produce, env, run, 1, budget, signals)
        try:
            if agent.await_write(1) is None:
                raise ScenarioStopped(f"the agent ended before its write: {agent.exit}")
            agent.release(1)
            agent.wait()
        finally:
            agent.kill()
    if not agent.commits:
        raise ScenarioStopped(f"the agent made no commit: {agent.exit}")
    return agent.commits[-1], (agent.exit or {}).get("exit")


@app.function(hide_code=True)
def commit_output(
    repo: icechunk.Repository, run: Run, method: str, out: MethodOutput, params: dict,
    metrics: dict | None = None, reasoning: str = "fixed params",
) -> str:
    """Commits `out` as iteration 1 on the method branch, which starts at the run base."""
    session = repo.writable_session(ensure_method_branch(repo, run, method))
    write_method_output(session, method, out, params, metrics or {}, reasoning)
    return session.commit(f"{method} iter 1: {reasoning}", metadata={"params": params, "iteration": 1})


@app.function(hide_code=True)
def read_output(repo: icechunk.Repository, snapshot: str, method: str) -> MethodOutput:
    session = repo.readonly_session(snapshot_id=snapshot)
    record = plain_json(read_element(session, f"uns/{method}"))
    return MethodOutput(
        embedding=read_element(session, f"obsm/X_{method}"),
        connectivities=read_element(session, f"obsp/{method}_connectivities"),
        distances=read_element(session, f"obsp/{method}_distances"),
        labels=pd.Categorical(read_element(session, f"results/{method}/labels")),
        fit_s=record["metrics"].get("fit_seconds", 0.0),
    )


@app.function(hide_code=True)
def same_parts(a: MethodOutput, b: MethodOutput) -> dict[str, bool]:
    return {
        "embedding": np.array_equal(a.embedding, b.embedding),
        "connectivities": (a.connectivities != b.connectivities).nnz == 0,
        "distances": (a.distances != b.distances).nnz == 0,
        "labels": np.array_equal(np.asarray(a.labels, dtype=str), np.asarray(b.labels, dtype=str)),
    }


@app.function(hide_code=True)
def classify_read(store: SidecarStore, method: str, obs_names, first: MethodOutput) -> dict:
    """What `read_result` returns after the kill: iteration 1, iteration 2, a mix, or nothing."""
    try:
        loaded = store.read_result(method, obs_names)
    except FileNotFoundError as error:
        return {"read": "missing", "read_error": str(error), "parts_from_iteration_1": []}
    same = same_parts(loaded, first)
    kind = "iteration 1" if all(same.values()) else "iteration 2" if not any(same.values()) else "mixed"
    return {"read": kind, "read_error": None, "parts_from_iteration_1": sorted(k for k, v in same.items() if v)}


@app.function(hide_code=True)
def overwrite_until_kill(root: str, method: str, out: MethodOutput, obs_names: list, prefix: str) -> None:
    """Spawn target: a sidecar overwrite that pauses before the rename of SIDECAR_KILL_FILE."""

    def before_rename(name: str) -> None:
        if name == SIDECAR_KILL_FILE:
            pause(prefix, name, {})

    SidecarStore(Path(root)).write_result(method, out, obs_names, before_rename=before_rename)


@app.function(hide_code=True)
def kill_sidecar_overwrite(store: SidecarStore, method: str, out: MethodOutput, obs_names: list, signal_dir: str) -> int:
    """Overwrites the result of `method` in a child and SIGKILLs it at its checkpoint. Returns the exit code."""
    prefix = str(Path(signal_dir) / "sidecar")
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=overwrite_until_kill, args=(str(store.root), method, out, obs_names, prefix))
    proc.start()
    path = Path(f"{prefix}.{SIDECAR_KILL_FILE}")
    deadline = time.monotonic() + SCENARIOS_WAIT_S
    try:
        while not path.exists():
            if not proc.is_alive():
                raise ScenarioStopped(f"the sidecar writer ended before its checkpoint, exit code {proc.exitcode}")
            if time.monotonic() > deadline:
                raise ScenarioStopped(f"no sidecar checkpoint after {SCENARIOS_WAIT_S} s")
            time.sleep(SCENARIOS_POLL_S)
    finally:
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
        proc.join()
    return proc.exitcode


@app.function(hide_code=True)
def kill_mid_write(
    env: dict, run: Run, store: SidecarStore, budget: Budget, produce=run_method_agent,
    repo: icechunk.Repository | None = None,
) -> dict:
    """Scenario 1. The parent commits a Harmony result with SEED_PARAMS. A Harmony agent with 1 iteration
    then gets SIGKILL after it writes its chunks and before it commits them. The sidecar gets the same
    two outputs; its writer gets SIGKILL before the rename of SIDECAR_KILL_FILE."""
    repo = repo or repository_from_env(env)
    branch = method_branch(run, METHOD)
    base = read_adata(repo.readonly_session(snapshot_id=run.base))
    obs_names = list(base.obs_names)
    first_out = run_method(base, METHOD, SEED_PARAMS)
    first = commit_output(repo, run, METHOD, first_out, SEED_PARAMS, {"fit_seconds": first_out.fit_s}, "seed")
    with tempfile.TemporaryDirectory() as signals:
        agent = AgentProcess(produce, env, run, 1, budget, signals)
        try:
            killed = agent.await_write(1)
        finally:
            agent.kill()

        tip = repo.lookup_branch(branch)
        tip_record = plain_json(read_element(repo.readonly_session(snapshot_id=tip), f"uns/{METHOD}"))
        icechunk_obs = {
            "agent_wrote": killed is not None,
            "agent_exit": (agent.exit or {}).get("exit"),
            "exitcode": agent.proc.exitcode,
            "first_commit": first,
            "tip": tip,
            "tip_is_first_commit": tip == first,
            "tip_iteration": tip_record["iteration"],
            "tip_params": tip_record["params"],
        }
        if killed is None:
            return {"icechunk": icechunk_obs, "sidecar": {"skipped": "the agent ended before its write"}}

        second_out = run_method(base, METHOD, killed["params"])
        tip_out = read_output(repo, tip, METHOD)
        icechunk_obs |= {
            "killed_params": killed["params"],
            # Harmony is deterministic: with the seed params, the killed write holds the seed data.
            "killed_params_equal_seed": killed["params"] == SEED_PARAMS,
            "tip_equals_first": all(same_parts(tip_out, first_out).values()),
            "tip_equals_killed": all(same_parts(tip_out, second_out).values()),
        }
        store.write_result(METHOD, first_out, obs_names)
        exitcode = kill_sidecar_overwrite(store, METHOD, second_out, obs_names, signals)

    result_dir = store.root / "results" / METHOD
    sidecar_obs = {
        "exitcode": exitcode,
        "killed_before_rename_of": SIDECAR_KILL_FILE,
        "temp_files": sorted(p.name for p in result_dir.glob(".*.tmp")),
        **classify_read(store, METHOD, obs_names, first_out),
    }
    return {"icechunk": icechunk_obs, "sidecar": sidecar_obs}


@app.function(hide_code=True)
def requalify(adata: anndata.AnnData) -> anndata.AnnData:
    """The re-QC: drops the cells above the REQC_QUANTILE of `n_counts`."""
    counts = adata.obs["n_counts"].to_numpy()
    return adata[counts <= np.quantile(counts, REQC_QUANTILE)].copy()


@app.function(hide_code=True)
def commit_requalified(repo: icechunk.Repository, run: Run, adata: anndata.AnnData) -> str:
    session = repo.writable_session(run.main)
    write_adata(session, adata)
    # The root rewrite drops /results; method branches expect it, as in the base.
    write_element(session, "results", {})
    return session.commit(f"Re-QC: drop the cells above the {REQC_QUANTILE:.0%} quantile of n_counts")


@app.function(hide_code=True)
def h5ad_obs_names(path: Path) -> list:
    with h5py.File(path, "r") as f:
        return list(anndata.io.read_elem(f["obs"]).index)


@app.function(hide_code=True)
def mutate_base_after_fork(
    env: dict, run: Run, store: SidecarStore, budget: Budget, produce=run_method_agent,
    repo: icechunk.Repository | None = None,
) -> dict:
    """Scenario 2. A Harmony agent with 1 iteration forks from the base. Before its commit, a re-QC commit
    on `run.main` drops the cells with the top 1% of `n_counts`. Then each store tries to use the result."""
    repo = repo or repository_from_env(env)
    branch = method_branch(run, METHOD)
    base = read_adata(repo.readonly_session(snapshot_id=run.base))
    store.write_base(base)
    with tempfile.TemporaryDirectory() as signals:
        agent = AgentProcess(produce, env, run, 1, budget, signals)
        try:
            if agent.await_write(1) is None:
                raise ScenarioStopped(f"the agent ended before its write: {agent.exit}")
            requalified = requalify(base)
            requalified_main = commit_requalified(repo, run, requalified)
            store.write_base(requalified)
            agent.release(1)
            agent.wait()
        finally:
            agent.kill()

    head = repo.lookup_branch(branch)
    try:
        promote(repo, run, METHOD)
        refused = None
    except PromotionRefused as error:
        refused = str(error)
    main_after = repo.lookup_branch(run.main)
    icechunk_obs = {
        "fork": run.base,
        "requalified_main": requalified_main,
        "cells_fork": base.n_obs,
        "cells_main": requalified.n_obs,
        "agent_head": head,
        "agent_exit": (agent.exit or {}).get("exit"),
        "promote_refused": refused,
        "main_after": main_after,
        "main_unchanged": main_after == requalified_main,
    }

    # The careful user stores the agent result with the obs_names it ran on, and checks them against the base on disk.
    store.write_result(METHOD, read_output(repo, head, METHOD), list(base.obs_names))
    current = h5ad_obs_names(store.root / "base.h5ad")
    try:
        store.read_result(METHOD, current)
        read_error = None
    except ValueError as error:
        read_error = str(error)
    sidecar_obs = {"cells_base_file": len(current), "read_error": read_error}
    return {"icechunk": icechunk_obs, "sidecar": sidecar_obs}


@app.function(hide_code=True)
def delete_run_refs(repo: icechunk.Repository, run: Run) -> list[str]:
    """Deletes every branch and tag of `run`, so a later GC reclaims its chunks. Returns the deleted refs."""
    branches = [b for b in repo.list_branches() if b.startswith(f"{run.id}-")]
    tags = [t for t in repo.list_tags() if t.startswith(f"{run.id}-")]
    for branch in branches:
        repo.delete_branch(branch)
    for tag in tags:
        repo.delete_tag(tag)
    return sorted([*branches, *tags])


@app.function(hide_code=True)
def run_isolated(repo: icechunk.Repository, budget: Budget, scenario: Callable[[Run, SidecarStore], dict]) -> dict:
    """Runs `scenario(run, store)` on a new run and a temporary sidecar folder, then deletes the run refs and folder.

    Returns the result plus `run_id` and `deleted_refs`. With a tripped cap it returns `{"skipped": cap}` and
    runs nothing; a `ScenarioStopped` gives `{"stopped": reason}`.
    """
    cap = budget.exceeded() or ("stopped" if budget.stop_event.is_set() else None)
    if cap:
        return {"skipped": cap}
    run = start_run(repo)
    try:
        # Scenario 2 writes two base .h5ad files of about 1.6 GB each, so the folder goes after each scenario.
        with tempfile.TemporaryDirectory(prefix="nb2-sidecar-") as sidecar:
            result = scenario(run, SidecarStore(Path(sidecar)))
    except ScenarioStopped as error:
        result = {"stopped": str(error)}
    finally:
        deleted = delete_run_refs(repo, run)
    return {**result, "run_id": run.id, "deleted_refs": deleted}


@app.function(hide_code=True)
def run_refs(repo: icechunk.Repository, run_ids: list[str]) -> list[str]:
    """The branches and tags that are left of the runs `run_ids`."""
    refs = [*repo.list_branches(), *repo.list_tags()]
    return sorted(r for r in refs if any(r.startswith(f"{i}-") for i in run_ids))


@app.function(hide_code=True)
def iso(t: datetime.datetime) -> str:
    return t.isoformat(timespec="seconds")


@app.function(hide_code=True)
def main_log(repo: icechunk.Repository, run: Run) -> list[dict]:
    """The commits on `run.main` after the fork, newest first."""
    log = []
    for snap in repo.ancestry(branch=run.main):
        if snap.id == run.base:
            break
        log.append({"snapshot": snap.id, "message": snap.message, "written_at": iso(snap.written_at)})
    return log


@app.function(hide_code=True)
def run_ops(repo: icechunk.Repository, run: Run) -> list[dict]:
    """The ops log entries of the run refs, newest first, back to the creation of `run.main`."""
    ops = []
    for update in repo.ops_log():
        op = type(update.kind).__name__
        ref = getattr(update.kind, "branch", None) or getattr(update.kind, "name", None)
        if ref and ref.startswith(run.id):
            ops.append({"op": op, "ref": ref, "at": iso(update.updated_at)})
        if op == "BranchCreated" and ref == run.main:
            break
    return ops


@app.function(hide_code=True)
def listing(directory: Path) -> list[dict]:
    if not directory.exists():
        return []
    return [
        {"name": d.name, "modified": iso(datetime.datetime.fromtimestamp(d.stat().st_mtime, datetime.timezone.utc))}
        for d in sorted(directory.iterdir())
    ]


@app.function(hide_code=True)
def promote_and_discard(
    env: dict, run: Run, store: SidecarStore, budget: Budget, produce=run_method_agent,
    repo: icechunk.Repository | None = None,
) -> dict:
    """Scenario 3. The parent commits the BASELINE result, and a Harmony agent with 1 iteration commits its result.
    The parent then promotes Harmony and discards BASELINE: named operations on Icechunk, file operations on the sidecar.
    Records what a reader can list afterwards, and when."""
    repo = repo or repository_from_env(env)
    adata = read_adata(repo.readonly_session(snapshot_id=run.base))
    obs_names = list(adata.obs_names)
    baseline = run_method(adata, BASELINE, BASELINE_PARAMS)
    commit_output(repo, run, BASELINE, baseline, BASELINE_PARAMS)
    store.write_result(BASELINE, baseline, obs_names)
    head, agent_exit = agent_commit(produce, env, run, budget)
    store.write_result(METHOD, read_output(repo, head, METHOD), obs_names)

    promote(repo, run, METHOD)
    tag = tag_winner(repo, run, METHOD)
    discard(repo, run, [BASELINE])
    store.promote(METHOD)
    store.discard(BASELINE)

    icechunk_obs = {
        "agent_head": head,
        "agent_exit": agent_exit,
        "tag": tag,
        "tag_snapshot": repo.lookup_tag(tag),
        "main_log": main_log(repo, run),
        "ops": run_ops(repo, run),
        "run_branches": sorted(b for b in repo.list_branches() if b.startswith(run.id)),
    }
    sidecar_obs = {
        "promoted": listing(store.root / "promoted"),
        "results": listing(store.root / "results"),
        "history": store.history(),
    }
    return {"icechunk": icechunk_obs, "sidecar": sidecar_obs}


@app.function(hide_code=True)
def redact_authors(metadata: dict) -> dict:
    """A copy of `metadata` with the AUTHOR_KEYS values set to "redacted"."""
    return {k: "redacted" if k in AUTHOR_KEYS else v for k, v in metadata.items()}


@app.function(hide_code=True)
def files_by_mtime(root: Path) -> list[dict]:
    files = [p for p in root.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "path": str(p.relative_to(root)),
            "modified": iso(datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc)),
            "bytes": p.stat().st_size,
        }
        for p in files
    ]


@app.function(hide_code=True)
def diff_agent(
    env: dict, run: Run, store: SidecarStore, budget: Budget, produce=run_method_agent,
    repo: icechunk.Repository | None = None,
) -> dict:
    """Scenario 4. A Harmony agent with 1 iteration commits one result. Then each store reports what changed."""
    repo = repo or repository_from_env(env)
    head, agent_exit = agent_commit(produce, env, run, budget)
    obs_names = list(read_element(repo.readonly_session(snapshot_id=run.base), "obs").index)
    store.write_result(METHOD, read_output(repo, head, METHOD), obs_names)

    changed = changed_paths(repo, run.base, head)
    snap = next(iter(repo.ancestry(snapshot_id=head)))
    tops = {p.split("/")[1] for paths in changed.values() for p in paths}
    icechunk_obs = {
        "head": head,
        "agent_exit": agent_exit,
        "changed": changed,
        "message": snap.message,
        "metadata": redact_authors(plain_json(snap.metadata)),
        "base_paths_untouched": tops.isdisjoint(BASE_PATHS),
    }
    sidecar_obs = {"files_by_mtime": files_by_mtime(store.root)}
    return {"icechunk": icechunk_obs, "sidecar": sidecar_obs}


@app.function(hide_code=True)
def latency_rows(shootout: dict[str, dict]) -> list[dict]:
    """One row per commit: payload, write (upload) and commit seconds, step seconds, and the commit share."""
    return [
        {
            "method": m,
            "iteration": c["iteration"],
            "committed_at": c["committed_at"],
            "payload_mb": c["payload_bytes"] / 1e6,
            "write_s": c["write_s"],
            "commit_s": c["commit_s"],
            "write_commit_s": c["write_commit_s"],
            "step_s": c["step_s"],
            "commit_share": c["commit_s"] / c["step_s"],
            "write_commit_share": c["write_commit_s"] / c["step_s"],
        }
        for m, r in shootout.items()
        for c in r["commits"]
    ]


@app.function(hide_code=True)
def latency_summary(rows: list[dict]) -> dict:
    """Medians of the write, commit and step times and of their shares. The maximum commit shows the tail."""
    if not rows:
        return {"commits": 0}

    def median(key: str) -> float:
        return statistics.median(r[key] for r in rows)

    return {
        "commits": len(rows),
        "median_payload_mb": median("payload_mb"),
        "median_write_s": median("write_s"),
        "median_commit_s": median("commit_s"),
        "median_write_commit_s": median("write_commit_s"),
        "median_step_s": median("step_s"),
        "median_commit_share": median("commit_share"),
        "median_write_commit_share": median("write_commit_share"),
        "max_commit_s": max(r["commit_s"] for r in rows),
    }


@app.function(hide_code=True)
def commits_per_branch(shootout: dict[str, dict], branch_of) -> list[dict]:
    """The commit count of each method branch. `branch_of(method)` gives the branch name."""
    return [{"branch": branch_of(m), "commits": r["iterations"]} for m, r in shootout.items()]


@app.function(hide_code=True)
def machine() -> dict:
    """CPU count, total memory and training accelerator, so step times from different machines compare."""
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {"cpu_count": os.cpu_count(), "memory_gib": round(memory / 2**30, 1), "accelerator": training_accelerator(os.environ)}


@app.function(hide_code=True)
def run_record(run, **fields) -> dict:
    """The JSON record of one notebook run: the run, the machine, and `fields`."""
    return {"run": vars(run), "machine": machine(), **fields}


@app.function(hide_code=True)
def cell_count(session: icechunk.Session) -> int:
    return len(read_element(session, "obs"))


@app.function(hide_code=True)
def manifest_rows(repo: icechunk.Repository, shootout: dict[str, dict]) -> list[dict]:
    """Manifest files, chunk refs and manifest bytes of each shootout commit. Run it before GC deletes losers."""
    rows = []
    for m, r in shootout.items():
        for c in r["commits"]:
            files = repo.list_manifest_files(c["snapshot"])
            rows.append({
                "method": m,
                "iteration": c["iteration"],
                "manifest_files": len(files),
                "chunk_refs": sum(f.num_chunk_refs for f in files),
                "manifest_bytes": sum(f.size_bytes for f in files),
            })
    return rows


@app.function(hide_code=True)
def open_times(repo: icechunk.Repository, branch: str) -> list[dict]:
    """Seconds to open each snapshot of `branch` and read `obs`, from the oldest snapshot to the head."""
    snapshots = list(repo.ancestry(branch=branch))[::-1]
    rows = []
    for depth, snap in enumerate(snapshots, start=1):
        t0 = time.perf_counter()
        session = repo.readonly_session(snapshot_id=snap.id)
        try:
            read_element(session, "obs")
            has_obs = True
        except (KeyError, zarr.errors.GroupNotFoundError):  # the first snapshot of a repo is empty
            has_obs = False
        rows.append({"depth": depth, "snapshot": snap.id, "message": snap.message, "has_obs": has_obs,
                     "seconds": time.perf_counter() - t0})
    return rows


@app.function(hide_code=True)
def stored_bytes(session: icechunk.Session, prefixes: list[str] | None = None) -> int:
    """Stored bytes of the arrays under `prefixes`, or of all arrays."""
    root = zarr.open_group(session.store, mode="r")
    return sum(
        array.nbytes_stored()
        for path, array in root.members(max_depth=None)
        if isinstance(array, zarr.Array) and (prefixes is None or any(path.startswith(p) for p in prefixes))
    )


@app.function(hide_code=True)
def method_prefixes(method: str) -> list[str]:
    """The paths that one method iteration writes."""
    return [f"obsm/X_{method}", f"obsp/{method}_", f"results/{method}/", f"uns/{method}"]


@app.function(hide_code=True)
def projection(n_obs: int, base_bytes: int, branch_bytes: int, iterations: int, cells: int = PROJECTED_CELLS) -> dict:
    """Bytes for `iterations` method results at `cells` cells. Branches: one base plus the method paths per result.
    Copies: one full AnnData, base plus result, per result. Both scale with cells at fixed genes and neighbours."""
    base_per_cell, branch_per_cell = base_bytes / n_obs, branch_bytes / n_obs
    return {
        "cells_measured": n_obs,
        "cells_projected": cells,
        "iterations": iterations,
        "base_bytes_per_cell": base_per_cell,
        "result_bytes_per_cell": branch_per_cell,
        "branch_bytes_projected": (base_per_cell + iterations * branch_per_cell) * cells,
        "copy_bytes_projected": iterations * (base_per_cell + branch_per_cell) * cells,
    }


@app.function(hide_code=True)
def write_record(path: Path, record: dict) -> Path:
    """Writes `record` as JSON. Creates the parent folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain_json(record), indent=2, default=str))
    return path


@app.function(hide_code=True)
def exit_label(reason: str | None) -> str:
    return EXIT_LABELS.get(reason, reason) if reason else "running"


@app.function(hide_code=True)
def views_round(value, digits: int):
    return None if value is None else round(value, digits)


@app.function(hide_code=True)
def progress_rows(progress: dict[str, dict]) -> list[dict]:
    """Rows of the live shootout table, from the `on_progress` argument of `run_shootout`."""
    return [
        {
            "Method": m,
            "State": "running" if p["running"] else "ended",
            "Iterations": p["iterations"],
            "Total (last)": views_round(p["total"], 3),
            "Cost US$": round(p["cost"], 4),
            "Time s": round(p["seconds"]),
        }
        for m, p in progress.items()
    ]


@app.function(hide_code=True)
def agent_rows(shootout: dict[str, dict]) -> list[dict]:
    """One row per method agent: exit, iterations, last and best Total, cost and time."""
    rows = []
    for m, r in shootout.items():
        totals = [c["total"] for c in r["commits"] if c.get("total") is not None]
        rows.append({
            "Method": m,
            "Exit": exit_label(r["exit"]),
            "Iterations": r["iterations"],
            "Total (head)": views_round(totals[-1] if totals else None, 3),
            "Total (best)": views_round(max(totals) if totals else None, 3),
            "Cost US$": round(r["cost"], 4),
            "Time s": round(r["seconds"]),
            # scib-metrics sets a negative PCR comparison to 0 for these iterations.
            "PCR clipped": ", ".join(str(c["iteration"]) for c in r["commits"] if c.get("pcr_clipped")),
            "Detail": r["detail"],
        })
    return rows


@app.function(hide_code=True)
def skip_reason(budget) -> str | None:
    """A note for a cell that a tripped cap skips, or None."""
    cap = budget.exceeded() or ("stopped" if budget.stop_event.is_set() else None)
    if not cap:
        return None
    return f"Skipped: the {EXIT_LABELS[cap] if cap in ('cost', 'time') else 'stop signal'} tripped before this cell."


@app.function(hide_code=True)
def evaluator_rows(evaluator: dict, seconds: float) -> list[dict]:
    """Field and value rows of the evaluator result."""
    promotion, gc = evaluator["promotion"] or {}, evaluator["gc"] or {}
    fields = [
        ("Model", evaluator["model"]),
        ("Exit", exit_label(evaluator["exit"])),
        ("Winner", evaluator["winner"]),
        ("Fork (base)", promotion.get("base")),
        ("Winner head", promotion.get("head")),
        ("Promoted iteration", promotion.get("iteration")),
        ("Promoted commit", promotion.get("promoted")),
        ("Move commit", promotion.get("move_snapshot")),
        ("Column-order commit", promotion.get("attr_snapshot")),
        ("Promotion seconds", views_round(promotion.get("timings_s", {}).get("total_s"), 2)),
        ("Tag", evaluator["tag"]),
        ("Deleted branches", ", ".join(evaluator["discarded"])),
        ("GC bytes deleted", gc.get("bytes_deleted")),
        ("GC chunks deleted", gc.get("chunks_deleted")),
        ("Lineage refusal", evaluator["lineage_refusal"]),
        ("Promotion failure", evaluator["promotion_failure"]),
        ("Cost US$", round(evaluator["cost"], 4)),
        ("Time s", round(seconds)),
    ]
    return [{"Field": f, "Value": v} for f, v in fields]


@app.function(hide_code=True)
def cost_time_rows(shootout: dict[str, dict], evaluator: dict, evaluator_s: float) -> list[dict]:
    """Cost and time of each method agent and of the evaluator."""
    rows = [
        {"Agent": m, "Exit": exit_label(r["exit"]), "Cost US$": round(r["cost"], 4), "Time s": round(r["seconds"])}
        for m, r in shootout.items()
    ]
    rows.append({"Agent": "evaluator", "Exit": exit_label(evaluator["exit"]), "Cost US$": round(evaluator["cost"], 4),
                 "Time s": round(evaluator_s)})
    return rows


@app.function(hide_code=True)
def label_agreement(adata: AnnData) -> tuple[dict, pd.DataFrame]:
    """ARI and NMI of the promoted Leiden labels against the ground truth, and their crosstab."""
    truth, found = adata.obs[LABEL_KEY], adata.obs[PROMOTED_COLUMN]
    scores = {"ARI": adjusted_rand_score(truth, found), "NMI": normalized_mutual_info_score(truth, found)}
    return scores, pd.crosstab(truth, found)


@app.function(hide_code=True)
def umap_coords(adata: AnnData, method: str) -> np.ndarray:
    """2-D UMAP of the `X_<method>` embedding. The BBKNN embedding is a 2-D UMAP already."""
    embedding = adata.obsm[f"X_{method}"]
    if embedding.shape[1] == 2:
        return np.asarray(embedding)
    work = AnnData(obs=adata.obs[[]].copy(), obsm={"X_emb": np.asarray(embedding)})
    sc.pp.neighbors(work, use_rep="X_emb", random_state=0)
    sc.tl.umap(work, random_state=0)
    return work.obsm["X_umap"]


@app.function(hide_code=True)
def umap_figure(coords: np.ndarray, obs: pd.DataFrame, keys: list[str], title: str):
    """One UMAP scatter panel per `obs` key, coloured by its categories."""
    fig, axes = plt.subplots(1, len(keys), figsize=(6.5 * len(keys), 5.5), squeeze=False)
    for ax, key in zip(axes[0], keys):
        labels = obs[key].astype("category")
        for i, cat in enumerate(labels.cat.categories):
            mask = (labels == cat).to_numpy()
            ax.scatter(coords[mask, 0], coords[mask, 1], s=0.5, color=plt.cm.tab20(i % 20), label=cat, rasterized=True)
        ax.set_title(f"{title}: {key}")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][0].legend(markerscale=10, fontsize=6, loc="center left", bbox_to_anchor=(-0.55, 0.5), frameon=False)
    fig.tight_layout()
    return fig


@app.function(hide_code=True)
def yes_no(flag) -> str:
    return "yes" if flag else "no"


@app.function(hide_code=True)
def scenario_rows(result: dict, rows) -> list[dict]:
    """`rows(result)`, or one row that says why the scenario did not finish."""
    if "skipped" in result:
        note = f"skipped: the {EXIT_LABELS.get(result['skipped'], result['skipped'])} tripped"
    elif "stopped" in result:
        note = f"stopped: {result['stopped']}"
    else:
        return rows(result)
    return [{"Question": "Did the scenario finish?", "Icechunk": note, "Sidecar": note}]


@app.function(hide_code=True)
def kill_rows(result: dict) -> list[dict]:
    """Scenario 1, side by side."""
    ice, side = result["icechunk"], result["sidecar"]
    if "skipped" in side:
        return [{"Question": "Did the agent reach its write?", "Icechunk": "no", "Sidecar": side["skipped"]}]
    if ice["tip_is_first_commit"] and ice["tip_equals_first"]:
        reader_gets = f"the seed commit, complete (iteration {ice['tip_iteration']})"
    else:
        reader_gets = f"tip {ice['tip']}, iteration {ice['tip_iteration']}"
    sidecar_gets = side["read"] if not side["read_error"] else f"{side['read']}: {side['read_error']}"
    if side["read"] == "mixed":
        sidecar_gets += f" (parts from iteration 1: {', '.join(side['parts_from_iteration_1'])})"
    return [
        {"Question": "Where did the kill land?", "Icechunk": "after the chunk writes, before the commit",
         "Sidecar": f"before the rename of {side['killed_before_rename_of']}"},
        {"Question": "Exit code of the writer", "Icechunk": ice["exitcode"], "Sidecar": side["exitcode"]},
        {"Question": "What does a reader get?", "Icechunk": reader_gets, "Sidecar": sidecar_gets},
        {"Question": "What stays on disk?", "Icechunk": "unreferenced chunks, until GC deletes them",
         "Sidecar": ", ".join(side["temp_files"]) or "nothing"},
    ]


@app.function(hide_code=True)
def kill_note(result: dict) -> str | None:
    """Scenario 1: a note when the agent chose the seed params, or None."""
    if not result.get("icechunk", {}).get("killed_params_equal_seed"):
        return None
    return (
        "Note: the agent chose the seed params. Harmony is deterministic, so the killed write holds the seed data. "
        "Thus the data checks cannot tell the seed from the killed write. The snapshot check still holds."
    )


@app.function(hide_code=True)
def fork_rows(result: dict) -> list[dict]:
    """Scenario 2, side by side."""
    ice, side = result["icechunk"], result["sidecar"]
    return [
        {"Question": "Cells at the fork", "Icechunk": ice["cells_fork"], "Sidecar": ice["cells_fork"]},
        {"Question": "Cells in the base after the re-QC", "Icechunk": ice["cells_main"],
         "Sidecar": side["cells_base_file"]},
        {"Question": "What detects the change?", "Icechunk": "the lineage check: main moved after the fork",
         "Sidecar": "the obs_names check only"},
        {"Question": "Result of the attempt", "Icechunk": f"promotion refused: {ice['promote_refused']}",
         "Sidecar": f"read refused: {side['read_error']}"},
        {"Question": "Is main unchanged by the attempt?", "Icechunk": yes_no(ice["main_unchanged"]),
         "Sidecar": "not applicable: no main"},
    ]


@app.function(hide_code=True)
def promote_rows(result: dict) -> list[dict]:
    """Scenario 3, side by side."""
    ice, side = result["icechunk"], result["sidecar"]
    return [
        {"Question": "Record of the winner", "Icechunk": f"tag {ice['tag']} at {ice['tag_snapshot']}",
         "Sidecar": ", ".join(d["name"] for d in side["promoted"]) + " in promoted/"},
        {"Question": "Commits on main since the fork", "Icechunk": len(ice["main_log"]), "Sidecar": "no commits"},
        {"Question": "Logged operations of the run", "Icechunk": len(ice["ops"]),
         "Sidecar": "none" if side["history"] is None else len(side["history"])},
        {"Question": "What is left", "Icechunk": ", ".join(ice["run_branches"]),
         "Sidecar": ", ".join(d["name"] for d in side["results"]) + " in results/"},
    ]


@app.function(hide_code=True)
def diff_rows(result: dict) -> list[dict]:
    """Scenario 4, side by side."""
    ice, side = result["icechunk"], result["sidecar"]
    changed = ice["changed"]
    return [
        {"Question": "What changed since the base?",
         "Icechunk": "; ".join(f"{k}: {', '.join(v)}" for k, v in changed.items() if v),
         "Sidecar": ", ".join(f["path"] for f in side["files_by_mtime"])},
        {"Question": "Are X, layers, obs and var untouched?", "Icechunk": yes_no(ice["base_paths_untouched"]),
         "Sidecar": "unknown: no history to compare"},
        {"Question": "Why did it change?", "Icechunk": ice["message"], "Sidecar": "no record"},
        {"Question": "With which params?", "Icechunk": str(ice["metadata"].get("params")), "Sidecar": "no record"},
    ]


if __name__ == "__main__":
    app.run()

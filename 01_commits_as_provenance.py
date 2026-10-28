# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anndata==0.13.*",
#     "icechunk==2.3.*",
#     "leidenalg",
#     "marimo",
#     "numpy",
#     "pandas",
#     "pooch",
#     "scanpy==1.12.*",
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
# zarr = "3.4.*"
# anndata = "0.13.*"
# scanpy = "1.12.*"
# leidenalg = "*"
# pooch = "*"
# numpy = "*"
# pandas = "*"
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App()

with app.setup(hide_code=True):
    import datetime
    import json
    import pathlib
    import platform
    import secrets
    import tempfile
    import warnings
    from importlib.metadata import version
    from pathlib import Path

    import anndata
    import icechunk
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pooch
    import scanpy as sc
    import zarr
    from anndata import AnnData

    DATA_DIR = Path(__file__).resolve().parent / "data"

    PBMC3K_URL = "https://exampledata.scverse.org/scanpy/pbmc3k_raw.h5ad"
    PBMC3K_FILENAME = "pbmc3k_raw.h5ad"
    PBMC3K_SHA256 = "sha256:89a96f1beaa2dd83a687666d3f19a4513ac27a2a2d12581fcd77afed7ea653a1"

    MARKERS_V1 = {
        "CD4 T": ["IL7R", "CD3E"],
        "CD14 Monocytes": ["CD14", "LYZ"],
        "B": ["MS4A1", "CD79A"],
        "NK": ["GNLY", "NKG7"],
        "FCGR3A Monocytes": ["FCGR3A", "MS4A7"],
        "Dendritic": ["FCER1A", "CST3"],
        "Megakaryocytes": ["PPBP"],
    }
    # MARKERS_V1 has no CD8 marker, so CD8 T cells get the "CD4 T" label.
    MARKERS_FIXED = {**MARKERS_V1, "CD8 T": ["CD8A", "CD8B"]}

    # Commit writes one manifest per new array, serially by default.
    COMMIT_MANIFEST_CONCURRENCY = 32


@app.cell
def _():
    mo.md(r"""
    # Commits as provenance

    A single-cell result depends on exact inputs: the data, the method and the parameters.
    This notebook keeps each analysis stage of PBMC3k as one Icechunk commit.
    Each commit records the method, the parameters and the package versions.
    Thus the history is a data structure that code can read and query.

    The notebook runs six stages and makes one commit for each stage.
    Then it reads the audit log and opens an earlier snapshot.
    Next, it fixes a wrong cell type label.
    A tag keeps the old labels.
    Last, it compares the size of the repository with `.h5ad` copies.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Repository

    Each run makes a new, empty repository in a new folder under `data/nb1/`.
    The run never writes into a repository from an earlier run.
    """)
    return


@app.cell
def _():
    # Mutes the readback and local-storage WARN lines. Other WARN lines stay visible.
    icechunk.set_logs_filter("warn,icechunk_storage::readback=error,icechunk_arrow_object_store=error")
    repo_path = DATA_DIR / "nb1" / fresh_name("pbmc3k")
    repo = local_repository(repo_path)
    mo.md(f"Repository path: `{repo_path}`")
    return repo, repo_path


@app.cell
def _():
    mo.md(r"""
    ## Six stages, one commit each

    The stages are ingest, QC, normalize, embed, cluster and annotate.
    Each cell below runs the scanpy calls of one stage.
    Then it writes the changed AnnData elements with `anndata.io.write_elem` and makes one commit.
    The commit metadata holds the stage name, the parameters and the package versions.

    Ingest and QC write the full AnnData.
    The later stages write only the elements that they change.
    Icechunk stores only the new chunks, so the history stays small.

    Below each stage, the table shows the Zarr nodes that the commit adds, rewrites or deletes.
    anndata deletes an element and writes it again, so the diff shows a rewrite and not an update.
    """)
    return


@app.cell
def _(repo):
    def show_commit(parent, snapshot):
        # Shows the commit message, its parameters and the nodes that changed since the parent snapshot.
        info = next(iter(repo.ancestry(snapshot_id=snapshot)))
        diff = repo.diff(from_snapshot_id=parent, to_snapshot_id=snapshot)
        return mo.vstack(
            [
                mo.md(f"**{info.message}** (`{snapshot}`)"),
                mo.md(f"```json\n{json.dumps(info.metadata['params'], indent=1)}\n```"),
                mo.ui.table(node_changes(diff), selection=None, page_size=8),
            ]
        )

    return (show_commit,)


@app.cell
def _(repo, show_commit):
    raw = load_pbmc3k()

    snap_init = repo.lookup_branch("main")
    _session = repo.writable_session("main")
    anndata.io.write_elem(zarr.open_group(_session.store), "/", raw)
    snap_ingest = _session.commit(
        "ingest: pooch download, SHA-256 verified",
        metadata={
            "stage": "ingest",
            "params": {"url": PBMC3K_URL, "sha256": PBMC3K_SHA256, "n_obs": raw.n_obs, "n_vars": raw.n_vars},
            "versions": package_versions(),
        },
    )
    show_commit(snap_init, snap_ingest)
    return raw, snap_ingest


@app.cell
def _(raw, repo, show_commit, snap_ingest):
    adata = raw.copy()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False)
    adata = adata[(adata.obs["n_genes_by_counts"] < 2500) & (adata.obs["pct_counts_mt"] < 5)].copy()

    _session = repo.writable_session("main")
    anndata.io.write_elem(zarr.open_group(_session.store), "/", adata)
    snap_qc = _session.commit(
        "qc: scanpy.pp.filter_cells, filter_genes, calculate_qc_metrics",
        metadata={
            "stage": "qc",
            "params": {"min_genes": 200, "min_cells": 3, "max_genes": 2500, "max_pct_mt": 5},
            "versions": package_versions(),
        },
    )
    show_commit(snap_ingest, snap_qc)
    return adata, snap_qc


@app.cell
def _(adata, repo, show_commit, snap_qc):
    # X keeps the raw counts; the normalized values go to a layer.
    _norm = sc.pp.normalize_total(adata, target_sum=1e4, copy=True)
    sc.pp.log1p(_norm)
    adata.layers["norm"] = _norm.X

    _session = repo.writable_session("main")
    _root = zarr.open_group(_session.store)
    anndata.io.write_elem(_root["layers"], "norm", adata.layers["norm"])
    snap_normalize = _session.commit(
        "normalize: scanpy.pp.normalize_total, log1p",
        metadata={"stage": "normalize", "params": {"target_sum": 1e4}, "versions": package_versions()},
    )
    show_commit(snap_qc, snap_normalize)
    return (snap_normalize,)


@app.cell
def _(adata, repo, show_commit, snap_normalize):
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", layer="norm")
    sc.pp.pca(adata, n_comps=50, layer="norm", mask_var="highly_variable")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)

    _session = repo.writable_session("main")
    _root = zarr.open_group(_session.store)
    anndata.io.write_elem(_root, "var", adata.var)
    for _name in ("X_pca", "X_umap"):
        anndata.io.write_elem(_root["obsm"], _name, adata.obsm[_name])
    anndata.io.write_elem(_root["varm"], "PCs", adata.varm["PCs"])
    for _name in ("distances", "connectivities"):
        anndata.io.write_elem(_root["obsp"], _name, adata.obsp[_name])
    for _name in ("hvg", "pca", "neighbors", "umap"):
        anndata.io.write_elem(_root["uns"], _name, adata.uns[_name])
    snap_embed = _session.commit(
        "embed: scanpy.pp.highly_variable_genes, pca, neighbors; scanpy.tl.umap",
        metadata={
            "stage": "embed",
            "params": {"n_top_genes": 2000, "n_comps": 50, "n_neighbors": 10, "n_pcs": 40},
            "versions": package_versions(),
        },
    )
    show_commit(snap_normalize, snap_embed)
    return (snap_embed,)


@app.cell
def _(adata, repo, show_commit, snap_embed):
    sc.tl.leiden(adata, resolution=0.9, flavor="igraph", n_iterations=2, directed=False, key_added="leiden")

    _session = repo.writable_session("main")
    _root = zarr.open_group(_session.store)
    anndata.io.write_elem(_root, "obs", adata.obs)
    anndata.io.write_elem(_root["uns"], "leiden", adata.uns["leiden"])
    snap_cluster = _session.commit(
        "cluster: scanpy.tl.leiden",
        metadata={
            "stage": "cluster",
            "params": {"resolution": 0.9, "n_iterations": 2, "flavor": "igraph"},
            "versions": package_versions(),
        },
    )
    show_commit(snap_embed, snap_cluster)
    return (snap_cluster,)


@app.cell
def _():
    mo.md(r"""
    ### Annotate

    The function below gives each Leiden cluster the cell type of its marker genes.
    It z-scores each marker gene across the cluster means of `layers["norm"]`.
    Then it picks the cell type with the highest mean z-score.
    The annotate cell writes the labels to `obs["cell_type"]`, commits them, and tags the commit as `v1`.
    """)
    return


@app.function
def annotate_clusters(adata: AnnData, markers: dict[str, list[str]]) -> pd.Categorical:
    """One cell type label per cell: the type of highest mean z-scored marker expression in its Leiden cluster."""
    genes = list(dict.fromkeys(g for gs in markers.values() for g in gs))
    expr = pd.DataFrame(
        adata.layers["norm"][:, [adata.var_names.get_loc(g) for g in genes]].toarray(),
        columns=genes,
    )
    means = expr.groupby(adata.obs["leiden"].to_numpy(), observed=True).mean()
    spread = means.std(ddof=0).replace(0, np.nan)
    z = ((means - means.mean()) / spread).fillna(0)
    scores = pd.DataFrame({t: z[gs].mean(axis=1) for t, gs in markers.items()})
    labels = scores.idxmax(axis=1)
    return pd.Categorical(adata.obs["leiden"].astype(str).map(labels.rename(index=str)))


@app.cell
def _(adata, repo, show_commit, snap_cluster):
    adata.obs["cell_type"] = annotate_clusters(adata, MARKERS_V1)

    _session = repo.writable_session("main")
    anndata.io.write_elem(zarr.open_group(_session.store), "obs", adata.obs)
    snap_annotate = _session.commit(
        "annotate: highest mean z-scored layers['norm'] marker expression per Leiden cluster",
        metadata={"stage": "annotate", "params": {"markers": MARKERS_V1}, "versions": package_versions()},
    )
    repo.create_tag("v1", snap_annotate)
    labels_v1 = adata.obs["cell_type"].copy()
    show_commit(snap_cluster, snap_annotate)
    return labels_v1, snap_annotate


@app.cell
def _(
    snap_annotate,
    snap_cluster,
    snap_embed,
    snap_ingest,
    snap_normalize,
    snap_qc,
):
    snapshots = {
        "ingest": snap_ingest,
        "qc": snap_qc,
        "normalize": snap_normalize,
        "embed": snap_embed,
        "cluster": snap_cluster,
        "annotate": snap_annotate,
    }
    return (snapshots,)


@app.cell
def _():
    mo.md(r"""
    ## Audit log

    The audit log is the ancestry of the `main` branch, newest first.
    Each row comes from the commit metadata, not from a separate log file.
    The last row is the empty snapshot that Icechunk makes when it creates the repository.
    The log also includes the `annotate_fix` commit from the fix section below.
    Under the table, the notebook shows the package versions from the newest commit.
    """)
    return


@app.cell
def _(repo, snap_fix):
    def key_params(params):
        # Shows the keys of a nested dict parameter, such as the marker sets.
        return ", ".join(
            f"{k}={list(v) if isinstance(v, dict) else v}" for k, v in params.items()
        )

    log = list(repo.ancestry(branch="main"))
    assert log[0].id == snap_fix
    log_table = pd.DataFrame(
        [
            {
                "time": snapshot.written_at.isoformat(),
                "message": snapshot.message,
                "stage": snapshot.metadata.get("stage", ""),
                "key params": key_params(snapshot.metadata.get("params", {})),
            }
            for snapshot in log
        ]
    )
    mo.vstack(
        [
            log_table,
            mo.md("Package versions in the newest commit:"),
            pd.Series(log[0].metadata["versions"], name="version").to_frame(),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## Time travel

    A snapshot id opens the exact state of an earlier stage.
    Here the notebook opens the snapshot after the embed stage.
    At that point, the data has a UMAP but no Leiden clusters yet.
    The plot colors the UMAP by `CD8A` expression from `layers["norm"]`.
    """)
    return


@app.cell
def _(repo, snapshots):
    _session = repo.readonly_session(snapshot_id=snapshots["embed"])
    at_embed = anndata.io.read_elem(zarr.open_group(_session.store, mode="r"))
    assert "leiden" not in at_embed.obs
    print("obs columns at embed:", list(at_embed.obs.columns))
    umap_ax = sc.pl.umap(at_embed, color="CD8A", layer="norm", show=False)
    umap_ax.figure
    return


@app.cell
def _():
    mo.md(r"""
    ## Fix a mislabeled cluster

    `MARKERS_V1` has no CD8 marker gene.
    So the rule gives the "CD4 T" label to the CD8 T cells too.
    `MARKERS_FIXED` adds `CD8A` and `CD8B` as markers for "CD8 T".
    The fix is a new commit with the stage name `annotate_fix`.
    The tag `v1` keeps the old labels.
    A reader can open both versions.
    """)
    return


@app.cell
def _(adata, labels_v1, repo, show_commit, snap_annotate):
    adata.obs["cell_type"] = annotate_clusters(adata, MARKERS_FIXED)

    _session = repo.writable_session("main")
    anndata.io.write_elem(zarr.open_group(_session.store), "obs", adata.obs)
    snap_fix = _session.commit(
        "annotate_fix: add CD8A and CD8B as CD8 T markers",
        metadata={"stage": "annotate_fix", "params": {"markers": MARKERS_FIXED}, "versions": package_versions()},
    )
    labels_fixed = adata.obs["cell_type"]
    assert "CD8 T" in set(labels_fixed)
    assert (labels_v1[labels_fixed == "CD8 T"] == "CD4 T").all()
    mo_fix = show_commit(snap_annotate, snap_fix)
    crosstab = pd.crosstab(
        labels_v1.rename("v1 label"), labels_fixed.rename("fixed label")
    )
    return crosstab, mo_fix, snap_fix


@app.cell
def _(crosstab, mo_fix):
    mo.vstack(
        [mo_fix, mo.md("Old labels (rows) against fixed labels (columns):"), crosstab]
    )
    return


@app.cell
def _(repo, snap_annotate):
    # The annotate cell creates tag v1. The reference to snap_annotate makes this cell run after it.
    assert repo.lookup_tag("v1") == snap_annotate
    _session = repo.readonly_session(tag="v1")
    at_v1 = anndata.io.read_elem(zarr.open_group(_session.store, mode="r"))
    assert "CD8 T" not in set(at_v1.obs["cell_type"])
    at_v1.obs["cell_type"].value_counts().rename("cells at tag v1")
    return


@app.cell
def _():
    mo.md(r"""
    ## Storage

    The notebook writes the state after each of the six stages as an `.h5ad` file.
    It writes each state twice: with no compression and with gzip compression.
    Each `.h5ad` file is a full copy of the data.
    So the `.h5ad` totals are the sum of six full copies.
    The repository stores only the elements that each stage writes.

    Zarr compresses the repository chunks with Zstandard by default.
    So the gzip row is the fair comparison, not the row with no compression.

    anndata rewrites the whole `obs` group when one `obs` column changes.
    Icechunk does not deduplicate the unchanged columns.
    So the cluster, annotate and fix commits each store the whole `obs` group again.
    The embed commit stores the whole `var` group again, because embed adds `var` columns.
    The repository also holds the `annotate_fix` commit, which the `.h5ad` totals do not include.
    """)
    return


@app.cell
def _(repo, repo_path, snap_fix, snapshots):
    _log = list(repo.ancestry(branch="main"))
    assert _log[0].id == snap_fix
    # Icechunk makes the initial snapshot, so n_commits does not count it.
    n_commits = len(_log) - 1
    stage_snapshots = list(snapshots.values())
    sizes = {
        f"repository ({n_commits} commits)": directory_bytes(repo_path),
        "six .h5ad, no compression": h5ad_snapshot_bytes(repo, stage_snapshots),
        "six .h5ad, gzip": h5ad_snapshot_bytes(repo, stage_snapshots, "gzip"),
    }
    return (sizes,)


@app.cell
def _(sizes):
    mo.vstack(
        [
            mo.md("Sizes in MB:"),
            pd.Series(sizes, name="MB").div(1e6).round(1).to_frame(),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## What this shows, and its limits

    - The audit log gives the method, the parameters and the package versions of each stage. Code reads it from the repository.
    - A snapshot id or a tag opens the exact data of a stage. A fix does not delete the old result.
    - The commit record is only as accurate as the code that writes it. Icechunk stores the metadata that it gets, and it does not check it.
    - Each `obs` change stores the whole `obs` group again. Each `var` change stores the whole `var` group again.
    - An `.h5ad` export does not keep the commit metadata. To keep it, copy it into `uns` before the export.
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


@app.function(hide_code=True)
def repository_config() -> icechunk.RepositoryConfig:
    config = icechunk.RepositoryConfig.default()
    config.manifest = icechunk.ManifestConfig(max_concurrent_manifest_fetches_during_commit=COMMIT_MANIFEST_CONCURRENCY)
    return config


@app.function(hide_code=True)
def local_repository(path: pathlib.Path) -> icechunk.Repository:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} exists and is not empty")
    return icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)), config=repository_config())


@app.function(hide_code=True)
def fresh_name(prefix: str) -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{secrets.token_hex(2)}"


@app.function(hide_code=True)
def directory_bytes(path: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


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
def load_pbmc3k() -> anndata.AnnData:
    """Raw counts, 2700 cells by 32738 genes."""
    path = pooch.retrieve(
        url=PBMC3K_URL,
        known_hash=PBMC3K_SHA256,
        fname=PBMC3K_FILENAME,
        path=DATA_DIR,
    )
    return read_old_h5ad(path)


@app.function(hide_code=True)
def anndata_store_root(session: icechunk.Session) -> zarr.Group:
    return zarr.open_group(session.store, mode="r" if session.read_only else "a")


@app.function(hide_code=True)
def read_adata(session: icechunk.Session) -> anndata.AnnData:
    return anndata.io.read_elem(anndata_store_root(session))


@app.function(hide_code=True)
def package_versions() -> dict[str, str]:
    names = ["icechunk", "anndata", "zarr", "scanpy", "numpy"]
    return {n: version(n) for n in names} | {"python": platform.python_version()}


@app.function(hide_code=True)
def node_changes(diff: icechunk.Diff) -> list[dict]:
    """One row per changed node: new, rewritten, updated or deleted."""
    new = {*diff.new_arrays, *diff.new_groups}
    deleted = {*diff.deleted_arrays, *diff.deleted_groups}
    # anndata write_elem deletes a node and writes it again, so the diff lists it as new and deleted.
    rewritten = new & deleted
    changes = {
        "new": new - rewritten,
        "rewritten": rewritten,
        "updated": {*diff.updated_arrays, *diff.updated_groups},
        "deleted": deleted - rewritten,
    }
    return [{"change": change, "path": path} for change, paths in changes.items() for path in sorted(paths)]


@app.function(hide_code=True)
def h5ad_snapshot_bytes(
    repo: icechunk.Repository, snapshot_ids: list[str], compression: str | None = None
) -> int:
    """Total size of one full `.h5ad` copy of each snapshot."""
    with tempfile.TemporaryDirectory() as tmp:
        for i, snapshot in enumerate(snapshot_ids):
            adata = read_adata(repo.readonly_session(snapshot_id=snapshot))
            adata.write_h5ad(Path(tmp) / f"{i}.h5ad", compression=compression)
        return directory_bytes(Path(tmp))


if __name__ == "__main__":
    app.run()

# icechunk-bio

Two marimo notebooks show Icechunk with single-cell data:

- **Notebook 1** (`01_commits_as_provenance.py`) follows the scanpy PBMC3k tutorial and makes one Icechunk commit per stage. The commit history is the provenance record. It runs in about 1 minute and needs no account.
- **Notebook 2** (`02_agentic_integration_shootout.py`) runs 7 LLM agents at the same time, one per batch-integration method. Each agent works on its own Icechunk branch. An evaluator agent promotes the best result to `main`. Four failure scenarios then compare Icechunk with a careful sidecar-file baseline. It needs an LLM API key and a store, and it costs a few cents of API use.

Each notebook is one file. It holds all of its code and its dependency list. You can run it on your laptop or on molab.

## Choose where to run

| | Your laptop | molab |
|---|---|---|
| What you install | pixi 0.80 or later | nothing; molab runs in the browser |
| Python | 3.12 | 3.13 |
| Notebook 2 hardware | your CPU, at least 32 GiB of memory | molab CPU, or a molab GPU for scvi and scanvi |
| Notebook 2 secrets | environment variables | the molab secrets pane |

## Run on your laptop

Install pixi 0.80 or later: <https://pixi.sh/latest/#installation>. Then clone this repository.

The first run of each notebook installs its environment. That takes a few minutes and some GB of disk space.

### Notebook 1

```bash
pixi run --script 01_commits_as_provenance.py
```

This command runs the notebook to the end. To open it in marimo instead, use this command and then click "Run all":

```bash
pixi exec marimo edit --sandbox=pixi 01_commits_as_provenance.py
```

Each run writes a new Icechunk repository to `data/nb1/`.
To run a stage again, restart the kernel and run all cells again.

### Notebook 2

1. Get an API key from the OpenAI API console. A ChatGPT subscription does not give API access. To use Anthropic instead, see "Notebook 2 settings" below.
2. Get a store. The default is Arraylake, and its free tier is enough. You need the name of your Arraylake org.
3. Set the API key. It is a secret, so it comes only from an environment variable:

   ```bash
   export OPENAI_API_KEY=<your key>
   ```

4. Open the notebook in marimo and click "Run all" once:

   ```bash
   pixi exec marimo edit --sandbox=pixi 02_agentic_integration_shootout.py
   ```

5. In the settings form, type your Arraylake org. Change the other settings if you want to. Then click "Start the run". Each click starts a new paid run.
6. Log in to Arraylake. If the notebook has no Arraylake login, its login cell prints a link and a code. Open the link, check that the code matches, and approve. The run then continues. To skip the login, set `ARRAYLAKE_TOKEN` to an Arraylake API token.

To run the notebook to the end with no browser, set the settings as environment variables and use script mode. Script mode does not wait for a click:

```bash
export ICB_ARRAYLAKE_ORG=<your org>
pixi run --script 02_agentic_integration_shootout.py
```

A run takes about 25 to 50 minutes. With the default models, a run costs about US$0.015. A larger iteration scale can make a run longer.
The notebook stops all agents at US$5 of API cost or after 90 minutes. The "Time cap in minutes" setting changes the time cap.

To try the notebook with no API key and no API cost, select "Dry run" in the settings form. Scripts with fixed parameters then replace the agents and the evaluator. The dry run still writes to the store.

## Run on molab

[molab](https://molab.marimo.io) is a free cloud service for marimo notebooks.
molab reads the dependency list at the top of the notebook and installs it.

### Notebook 1 on molab

1. Open molab.marimo.io and create a new notebook.
2. Upload `01_commits_as_provenance.py`.
3. Click "Run all".

### Notebook 2 on molab

1. Open molab.marimo.io, create a new notebook, and upload `02_agentic_integration_shootout.py`.
2. In the secrets pane of the developer panel, set `OPENAI_API_KEY`.
3. Optional: to train scvi and scanvi on a GPU, click the notebook specs button in the app header and switch on the GPU. With the GPU, a run is about one third shorter.
4. Click "Run all" once.
5. In the settings form, type your Arraylake org.
   - If you switched on the GPU, set "Device for scvi and scanvi" to `gpu`.
   - To give the agents more iterations, raise "Iteration scale", for example to `2`. Also raise "Time cap in minutes", for example to `120`.
6. Click "Start the run". Each click starts a new paid run.
7. The login cell prints a link and a code. Open the link, check the code, and approve.
8. Wait for the run to end. Do not run cells while the run continues, because a new cell run can stop the agents.

The login cell prints your Arraylake email and user ID. An HTML export of the notebook keeps them.

molab shows two kinds of harmless messages for Notebook 2:

- `Jax plugin configuration error` with a traceback. molab has a JAX CUDA plugin in its system packages that does not match our JAX version. JAX continues on the CPU.
- Information lines from TensorFlow, which molab also has in its system packages.

## Notebook 2 settings

Secrets come only from environment variables. On molab, set them in the secrets pane.

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | yes, with OpenAI | the LLM agents and the evaluator |
| `ANTHROPIC_API_KEY` | yes, with Anthropic | the LLM agents and the evaluator |
| `ARRAYLAKE_TOKEN` | no | an Arraylake API token; it skips the login cell |
| AWS credentials | yes, with S3 | from the standard AWS environment |

You set the other settings in the settings form of the notebook. The store is a dropdown above the form. The form shows only the fields of the chosen store.
An environment variable with the name in the table sets the start value of its field. Script mode uses the start values.

| Form field | Variable | Values | Purpose |
|---|---|---|---|
| LLM provider | `ICB_LLM` | `openai` (default), `anthropic` | the LLM provider |
| Store | `ICB_STORE` | `arraylake` (default), `s3` | where the Icechunk repository lives |
| Arraylake org | `ICB_ARRAYLAKE_ORG` | your org; required with Arraylake | the notebook uses the repo `<org>/icechunk-bio-nb2` and creates it on first use |
| S3 bucket, S3 prefix, S3 region | `ICB_S3_BUCKET`, `ICB_S3_PREFIX`, `ICB_S3_REGION` | your bucket; required with S3 | the S3 location |
| Device for scvi and scanvi | `ICB_ACCELERATOR` | `cpu` (default), `gpu`, `auto` | the other methods and the scIB scoring use the CPU |
| Iteration scale | `ICB_ITERATION_SCALE` | a whole number of 1 or more, `1` (default) | multiplies the iteration limit of each agent; the unintegrated baseline stays at 1 |
| Time cap in minutes | `ICB_MAX_MINUTES` | minutes, `90` (default) | the time cap of the notebook; a larger scale needs more time |
| Dry run | `ICB_NB2_DRY_RUN` | off (default), on (`1`) | a run with no LLM call |

## Measured runs

Notebook 2 runs on 2026-10-23 with the default models.
These runs scored Bio conservation on all cells. The notebook now scores it on the 3 Oetjen query batches only, so new Totals differ.

| Where | Wall time | API cost | Median commit | Write plus commit share of an agent step |
|---|---|---|---|---|
| Laptop (Apple silicon) | 1403 s | US$0.0138 | 3.07 s | 12.0% |
| AWS `m7g.4xlarge` | 3002 s | US$0.0146 | 2.21 s | 6.7% |
| molab CPU | 2094 s | US$0.0144 | 1.30 s | 3.3% |
| molab GPU, first run | 1397 s | US$0.0149 | 1.54 s | 4.1% |
| molab GPU, second run | 1439 s | US$0.0146 | 1.32 s | 3.5% |

On the molab GPU, the scvi agent took about 650 s and the scanvi agent about 580 s. On the molab CPU, they took 1316 s and 1146 s.
The molab session had a 20 CPU quota and 160 GiB of memory. The molab docs state 4 CPUs and 32 GB, so your session can have other limits.

## Notes on the Notebook 2 repository

- Run one notebook per repository at a time. The evaluator runs garbage collection, which deletes the uncommitted chunks of any other writer on the repository.
- A clean run keeps its `<run id>-main` branch, the winner branch and the winner tag.
- Each failure scenario deletes its own branches and tags after it records its result. The garbage collection of the next run reclaims their chunks.
- A stopped run keeps its branches. Delete them by hand if you do not need them.
- Each run writes a JSON record to `data/nb2/`.

## License

Apache License 2.0. See `LICENSE`.

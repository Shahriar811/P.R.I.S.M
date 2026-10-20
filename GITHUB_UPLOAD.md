# Uploading this project to GitHub

Publish the source code and training notebook. Keep credentials, trained models,
private datasets, feedback, logs, and report drafts on your computer. The checked-in
`.gitignore` excludes them; `.env.example` contains empty credential fields.

## Findings from the local inspection

The original folder contained 310 files totaling about 12.75 GiB:

| File or directory | Finding | Upload policy |
| --- | --- | --- |
| `.env` | Contains a Groq API key | Excluded; upload only `.env.example` |
| `ngrok_token.txt` | Contains an ngrok credential and was not previously ignored | Excluded |
| `scratch/test_guardrail.py` | Contained a hardcoded Groq API key | Changed to read local environment configuration; all `scratch/` excluded |
| `Saved model/TM/extra_trees.joblib` | 12,479,796,721 bytes, about 11.62 GiB | Excluded |
| `Saved model/TM/random_forest.joblib` | 1,070,361,473 bytes, about 1.00 GiB | Excluded |
| `Saved model/` | About 12.67 GiB of generated model/preprocessing artifacts | Excluded in full |
| `federated/data/`, `federated/feedback/` | Training/test shards and locally collected feedback | Excluded |
| `federated/runtime_logs/`, status JSON, `catboost_info/` | Generated runtime/training state | Excluded |
| `REport/`, `scratch/` | Report drafts, reviews, extracted document text, exploratory scripts | Excluded |
| `IDS_FL_RL_Combined.ipynb` | Originally 51 saved outputs, about 1.46 MiB serialized | Outputs/execution state cleared; source cells preserved |

No Git repository existed in this folder at inspection time. These instructions
are for a first upload to an empty remote repository.

The inspection looked for common credential formats and local credential copies.
It is not a complete application security audit. If either credential has already
been shared, committed elsewhere, or exposed publicly, revoke/rotate it at its
provider and update your local files. Ignoring it does not revoke it.

GitHub warns about regular Git files above 50 MiB and blocks files above 100 MiB.
Git LFS has plan-dependent per-file limits up to 5 GB, so it cannot hold the
11.62 GiB Extra Trees file as one object. Keep model artifacts in separate,
access-controlled file/object storage and document access instructions if needed.
Do not upload private training data just because it fits below a size limit.

Sources: [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
[Git LFS limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

## First upload with Git Bash

1. [Create a new GitHub repository](https://github.com/new). Choose its visibility;
   private is a reasonable starting point. Leave README, license, and `.gitignore`
   initialization unchecked so the remote is empty.
2. Replace the repository URL below. If Git asks for your identity, use your
   preferred name and GitHub email (or your GitHub-provided noreply email).
3. Open **Git Bash** and paste this block. The subshell stops at the first failed
   step. All excluded files stay on disk.

```bash
(
  set -e
  cd '/e/CSE499B/Latest One(FL+RL)'

  git init -b main
  # Only if Git has no commit identity configured:
  # git config user.name 'Your Name'
  # git config user.email 'YOUR_GITHUB_NOREPLY_EMAIL'

  python scripts/github_safety.py prepare
  git add .
  python scripts/github_safety.py check
  git diff --cached --stat

  git commit -m 'Initial commit: SDN-IoT IDS with federated and reinforcement learning'
  git remote add origin 'https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git'
  git push -u origin main
)
```

The `check` command inspects the **complete staged snapshot**, rejects local-only
paths and files at least 50 MiB, scans for common credentials and exact copies of
local Groq/ngrok credentials, and rejects notebooks with saved outputs. It reports
paths and reasons without printing credential values. A pass is a useful check,
not a guarantee that arbitrary secrets or personal data are absent.

Authenticate through Git Credential Manager's browser sign-in when prompted.
Never put an access token in the repository URL or a script. GitHub account
passwords cannot be used as Git HTTPS credentials. See
[GitHub authentication guidance](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git?platform=windows).

The notebook's original outputs are preserved under the ignored
`.local-backups/github/` directory, with a content hash in the backup filename.
Running `prepare` again on a clean notebook does not create another backup.

## Later updates

From the project root in Git Bash:

```bash
(
  set -e
  python scripts/github_safety.py prepare
  git add .
  python scripts/github_safety.py check
  git diff --cached --stat
  git commit -m 'Describe your changes'
  git push
)
```

Do not use `git add -f` to bypass exclusions. `.gitignore` does not untrack files
that have already been committed. The checker inspects the current index, not
earlier commits. If you already committed a secret or oversized artifact in
another repository, clean that history before pushing and rotate exposed
credentials. Do not bypass GitHub secret protection or force-push to resolve it.

If a new remote rejects the push because it already has commits, clone that
remote into a separate directory and copy only the reviewed source files into
it. Keep its existing history rather than replacing it.

## Running a fresh clone

A source-only clone needs the original dataset and regenerated artifacts before
the dashboard can start:

1. Install `requirements.txt` in a Python virtual environment. The optional live
   packet sniffer also needs your platform's packet-capture driver and privileges.
2. Copy `.env.example` to `.env` and set your own key if using Groq features.
3. Obtain `ASEADOS_SDN_IoT.csv` from an authorized source and put it in the project
   root or its parent, as the notebook's path configuration expects. The dataset
   is not bundled. Update `DATA_PATH` if storing it elsewhere.
4. Run the notebook's training/export sections to regenerate `Saved model/` and
   `federated/data/`. Run `federated/run_rl_federated.bat` for the RL-to-FL stage.
   Finish exporting the preprocessing artifacts in Section 11.
5. Start the dashboard with `python ids_server.py`.

The preprocessing artifacts (`scaler.joblib`, `label_encoder.joblib`, and
`feature_cols.joblib`) and matching trained models must come from the same
pipeline. Restore their original relative paths if obtaining reviewed artifacts
separately. Load only trusted model files: joblib/pickle and PyTorch checkpoints
loaded with `weights_only=False` can execute code when deserialized.

## Separate deployment issues observed

The original application used `host='0.0.0.0'` and `debug=True`. The published
version now starts on localhost with debugging/reloading disabled and a 1 MiB
request-body limit. Its settings, training-trigger, feedback, and Groq-backed
endpoints still need authentication before any public deployment. The browser's
role picker is only a view preference. `expose_ngrok.ps1` can make the service
reachable through a public tunnel, so add access controls and use an appropriate
production server before operating a public demo. Source-publication checks are
not a complete application security audit.

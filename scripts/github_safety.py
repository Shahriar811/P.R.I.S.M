"""Prepare the training notebook and check the complete Git index before upload.

Uses Python's standard library and Git only. Never prints credential values.
This checks the staged snapshot; it does not audit previous commits or certify
application security.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 50 * 1024 * 1024  # Conservative project policy, below GitHub's hard limit.
SECRET_NAMES = re.compile(r"(?:key|token|password|secret)", re.IGNORECASE)
SECRET_PATTERNS = {
    "Groq API key": rb"\bgsk_[A-Za-z0-9_-]{15,}",
    "GitHub token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    "API token": rb"\bsk-[A-Za-z0-9_-]{20,}",
    "AWS access key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
    "credential in URL": rb"https?://[^\s/:@]+:[^\s/@]+@",
}
LITERAL_SECRET = re.compile(
    rb"(?i)\b[\w-]*(?:api[_-]?key|auth[_-]?token|access[_-]?token|"
    rb"client[_-]?secret|secret[_-]?key|password)\b\s*[=:]\s*['\"]([^'\"\r\n]{8,})['\"]"
)
LOCAL_DIRECTORIES = {
    "saved model", "scratch", "report", ".local-backups", "data", "datasets",
    "feedback", "runtime_logs", "catboost_info", "__pycache__",
    ".ipynb_checkpoints", ".venv", "venv", "env", ".aws", ".ssh",
}
LOCAL_SUFFIXES = {
    ".joblib", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".h5", ".hdf5",
    ".onnx", ".safetensors", ".npz", ".npy", ".csv", ".tsv", ".parquet",
    ".pcap", ".pcapng", ".db", ".sqlite", ".sqlite3", ".log", ".pyc",
    ".pem", ".key", ".p12", ".pfx", ".exe", ".zip", ".7z",
}


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def env_pairs(text: str):
    for line in text.splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and not name.startswith("#"):
            yield name.strip(), value.strip().strip("\"'")


def local_secrets(root: Path) -> set[bytes]:
    """Also recognize exact copies of this project's untracked credentials."""
    values = set()
    for path in root.glob(".env*"):
        if path.name == ".env.example" or not path.is_file():
            continue
        for name, value in env_pairs(path.read_text(encoding="utf-8-sig")):
            if SECRET_NAMES.search(name) and len(value) >= 8:
                values.add(value.encode("utf-8"))
    token_file = root / "ngrok_token.txt"
    if token_file.is_file():
        token = token_file.read_text(encoding="utf-8-sig").strip()
        if len(token) >= 8:
            values.add(token.encode("utf-8"))
    return values


def prepare_notebook(root: Path) -> None:
    path = root / "IDS_FL_RL_Combined.ipynb"
    original = path.read_bytes()
    notebook = json.loads(original)
    clean = json.loads(original)
    for cell in clean.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        metadata = cell.get("metadata", {})
        for key in ("execution", "ExecuteTime", "collapsed", "scrolled"):
            metadata.pop(key, None)
    for key in ("widgets", "signature"):
        clean.get("metadata", {}).pop(key, None)
    if clean == notebook:
        print("Notebook already has no saved outputs or execution state.")
        return
    backup_dir = root / ".local-backups" / "github"
    backup_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(original).hexdigest()[:12]
    backup = backup_dir / f"{path.stem}.{digest}.ipynb"
    if not backup.exists():
        backup.write_bytes(original)
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Cleared notebook outputs; source cells preserved. Backup: {backup.relative_to(root)}")


def private_path(name: str) -> bool:
    path = PurePosixPath(name.lower())
    if any(part in LOCAL_DIRECTORIES for part in path.parts[:-1]):
        return True
    if path.suffix in LOCAL_SUFFIXES:
        return True
    if path.name.startswith(".env") and path.name != ".env.example":
        return True
    if path.name in {
        "ngrok_token.txt", "ngrok.yml", "ngrok.yaml", "credentials.json",
        "secrets.json", "id_rsa", "id_ed25519",
    }:
        return True
    return name.lower().startswith("federated/") and (
        path.name == "status.json" or re.fullmatch(r"client_\d+_status\.json", path.name) is not None
    )


def inspect_content(name: str, content: bytes, secrets: set[bytes]) -> list[str]:
    reasons = [label for label, pattern in SECRET_PATTERNS.items() if re.search(pattern, content)]
    if any(value in content for value in secrets):
        reasons.append("copy of a local credential")
    for match in LITERAL_SECRET.finditer(content):
        value = match.group(1).lower()
        if not any(marker in value for marker in (b"your_", b"your-", b"example", b"placeholder", b"changeme", b"<")):
            reasons.append("possible hardcoded credential")
            break
    if PurePosixPath(name).name == ".env.example":
        if any(SECRET_NAMES.search(key) and value for key, value in env_pairs(content.decode("utf-8-sig"))):
            reasons.append("credential fields in .env.example must be empty")
    if name.lower().endswith(".ipynb"):
        try:
            notebook = json.loads(content)
            cells = notebook["cells"]
            if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in cells):
                reasons.append("notebook has saved outputs/execution counts; run prepare and git add again")
            if notebook.get("metadata", {}).get("widgets"):
                reasons.append("notebook has saved widget state")
            # Scan decoded source too, including tokens split across JSON source lines.
            source = "\n".join("".join(cell.get("source", [])) for cell in cells).encode("utf-8")
            reasons.extend(inspect_content("notebook-source.py", source, secrets))
        except (ValueError, KeyError, TypeError, AttributeError):
            reasons.append("invalid notebook JSON")
    return sorted(set(reasons))


def check_index() -> int:
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    entries = git(root, "ls-files", "--stage", "-z").split(b"\0")
    ignored = set(git(root, "ls-files", "-ci", "--exclude-standard", "-z").split(b"\0"))
    secrets = local_secrets(root)
    errors = []
    total = count = 0
    for entry in filter(None, entries):
        info, raw_name = entry.split(b"\t", 1)
        mode, oid, stage = info.split()
        name = raw_name.decode("utf-8", errors="replace")
        if stage != b"0" or mode not in (b"100644", b"100755"):
            errors.append((name, "unmerged file, symlink, or submodule requires manual review"))
            continue
        if raw_name in ignored or private_path(name):
            errors.append((name, "local/private/generated file must not be staged"))
            continue
        size = int(git(root, "cat-file", "-s", oid.decode("ascii")))
        total += size
        count += 1
        if size >= MAX_FILE_BYTES:
            errors.append((name, "file is at least 50 MiB; keep large artifacts outside this source repository"))
            continue
        content = git(root, "cat-file", "blob", oid.decode("ascii"))
        errors.extend((name, reason) for reason in inspect_content(name, content, secrets))
    if not count and not errors:
        errors.append(("Git index", "no files are staged; run git add . first"))
    if errors:
        print("UPLOAD CHECK FAILED. Credential values are deliberately hidden.")
        for name, reason in errors:
            print(f"  {name}: {reason}")
        print("Fix the reported files, stage the fixes, and run this check again.")
        return 1
    print(f"PASS: {count} staged files, {total / 1024:.1f} KiB total; no blocked files or detected credentials.")
    print("Checks cover the complete index, including unchanged tracked files; previous commits are not scanned.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check"))
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_notebook(PROJECT_ROOT)
            return 0
        return check_index()
    except subprocess.CalledProcessError:
        print("Git check could not run. Initialize Git in this project and resolve Git errors first.", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        # Avoid echoing file contents or exception text that could contain credentials.
        print(f"Upload preparation/check could not finish ({type(error).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Fetch bounded, pinned public sources; never uses the application's credentials."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import httpx

from .parsing import parse_jats, parse_markup

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "catalog.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get(client: httpx.Client, url: str, cache: Path, *, expected_hash: str | None = None, refresh: bool = False) -> bytes:
    """Content-address URL cache with optional pinned SHA validation."""
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / digest(url.encode())
    if target.exists() and not refresh:
        raw = target.read_bytes()
    else:
        for attempt in range(3):
            try:
                response = client.get(url)
                response.raise_for_status()
                raw = response.content
                if len(raw) > 25_000_000:
                    raise ValueError(f"Source exceeds 25MB bound: {url}")
                if expected_hash and digest(raw) != expected_hash:
                    raise ValueError(f"Pinned SHA256 mismatch: {url}")
                # Readers never observe a partially written cache file.
                with tempfile.NamedTemporaryFile(dir=cache, prefix="fetch-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(raw)
                temporary.replace(target)
                break
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
    if expected_hash and digest(raw) != expected_hash:
        raise ValueError(f"Pinned SHA256 mismatch: {url}")
    return raw


def resolve_lock(client: httpx.Client, catalog: dict, cache: Path, lock_path: Path, *, refresh: bool = False) -> dict:
    catalog_hash = digest(CATALOG.read_bytes())
    if lock_path.exists() and not refresh:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["catalog_sha256"] != catalog_hash:
            raise ValueError("Catalog changed: use --refresh-lock explicitly")
        return lock
    lock = {"version": 1, "catalog_sha256": catalog_hash, "github": [], "europe_pmc": [], "tokenizer": catalog["tokenizer"]}
    for source in catalog["github"]:
        repo = source["repo"]
        try:
            commit = json.loads(get(client, f"https://api.github.com/repos/{repo}/commits/{source['ref']}", cache, refresh=refresh))["sha"]
            tree = json.loads(get(client, f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1", cache))
        except httpx.HTTPError:
            # Public API quotas are shared by network egress. Git transport needs
            # no API token and fetches only commit/tree metadata, no checkout.
            checkout = cache / "git" / repo.replace("/", "--")
            checkout.parent.mkdir(parents=True, exist_ok=True)
            if not (checkout / ".git").exists():
                subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--filter=blob:none", "--no-checkout", "--single-branch", "--branch", source["ref"], f"https://github.com/{repo}.git", str(checkout)], check=True, capture_output=True, timeout=180)
            elif refresh:
                subprocess.run(["git", "-C", str(checkout), "fetch", "--quiet", "--depth", "1", "origin", source["ref"]], check=True, capture_output=True, timeout=180)
            revision = "FETCH_HEAD" if refresh and (checkout / ".git" / "FETCH_HEAD").exists() else "HEAD"
            commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", revision], text=True).strip()
            entries = subprocess.check_output(["git", "-C", str(checkout), "ls-tree", "-rz", commit]).decode("utf-8").split("\0")
            tree = {"tree": [{"type": info.split()[1], "sha": info.split()[2], "path": path}
                              for entry in entries if entry for info, path in [entry.split("\t", 1)]]}
        if tree.get("truncated"):
            raise ValueError(f"Truncated source tree: {repo}")
        minimum = 6_000 if source["language"] == "zh" else 12_000
        candidates = [e for e in tree["tree"] if e["type"] == "blob" and e["path"].startswith(source["prefix"])
                      and e["path"].endswith(source["suffix"]) and ("size" not in e or minimum <= e["size"] <= 2_000_000)
                      and not any(s in e["path"] for s in ("whatsnew/", "changelog", "_origin.md", "release/", "releasenotes/"))]
        # Large natural files first; deterministic ties. Limits are per source.
        candidates.sort(key=lambda e: (-e.get("size", 0), e["path"]))
        limit = source["max_files"] if all("size" in e for e in candidates) else source["max_files"] * 2
        selected = [{"path": e["path"], "git_blob": e["sha"], "bytes": e.get("size")} for e in candidates[:limit]]
        licenses = []
        for path in source["license_paths"]:
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
            try:
                value = get(client, url, cache)
            except httpx.HTTPError:
                continue
            licenses.append({"url": url, "sha256": digest(value), "text": value.decode("utf-8")})
        if not licenses:
            raise ValueError(f"No license observation fetched: {repo}")
        lock["github"].append({**source, "commit": commit, "files": selected, "license_evidence": licenses})
        print(f"Pinned {repo}@{commit[:12]}: {len(selected)} files", flush=True)
    for source in catalog.get("europe_pmc", []):
        request = httpx.URL("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"query": source["query"], "format": "json", "pageSize": str(source["max_articles"] * 2)})
        response = json.loads(get(client, str(request), cache))
        records = [r for r in response["resultList"]["result"] if r.get("pmcid")]
        lock["europe_pmc"].append({**source, "search_url": str(request), "articles": records})
    write_json(lock_path, lock)
    return lock


def normalize_cc(links: list[str], text: str) -> str | None:
    joined = " ".join(links) + " " + text
    m = re.search(r"creativecommons.org/licenses/(by(?:-nc)?(?:-nd|-sa)?)/(\d\.\d)", joined, re.I)
    if m:
        return "CC-" + m.group(1).upper() + "-" + m.group(2)
    if "creativecommons.org/publicdomain/zero" in joined:
        return "CC0-1.0"
    return None


def acquire(args) -> int:
    implementation_hashes = {p.name: digest(p.read_bytes()) for p in HERE.glob("*.py")}
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "raw"
    client = httpx.Client(timeout=45, follow_redirects=True, headers={"User-Agent": "Ariadne-longdoc-research/1.0"}, limits=httpx.Limits(max_connections=12))
    lock = resolve_lock(client, catalog, cache, Path(args.lock), refresh=args.refresh_lock)
    tokenizer_spec = lock["tokenizer"]
    tokenizer_url = f"https://huggingface.co/{tokenizer_spec['repo']}/resolve/{tokenizer_spec['revision']}/{tokenizer_spec['filename']}"
    tokenizer_bytes = get(client, tokenizer_url, cache, expected_hash=tokenizer_spec.get("sha256"))
    tokenizer_path = output / "tokenizer.json"
    tokenizer_path.write_bytes(tokenizer_bytes)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    documents: list[dict] = []
    audit: list[dict] = []
    records: list[dict] = []

    def finish(doc: dict, raw: bytes, url: str) -> None:
        text = "\n\n".join(b["text"] for b in doc["blocks"])
        tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
        record = {"id": doc["id"], "source_url": url, "raw_sha256": digest(raw), "tokens": tokens,
                  "tokenizer_sha256": digest(tokenizer_bytes)}
        records.append(record)
        if tokens < args.min_tokens:
            audit.append({**record, "status": "below_min_tokens"})
            return
        documents.append({**doc, "raw_sha256": digest(raw), "token_count": tokens,
                          "tokenizer": {**tokenizer_spec, "sha256": digest(tokenizer_bytes), "add_special_tokens": False},
                          "review_status": "unreviewed", "annotation_status": "unreviewed", "source_format": doc.pop("source_format", "markup")})

    jobs = []
    for source in lock["github"]:
        for file in source["files"]:
            jobs.append((source, file))

    def fetch_one(job):
        source, file = job
        path, repo, commit = file["path"], source["repo"], source["commit"]
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        raw = get(client, url, cache, expected_hash=file.get("sha256"))
        git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if git_blob != file["git_blob"]:
            raise ValueError(f"Git blob hash mismatch: {url}")
        blocks = parse_markup(raw.decode("utf-8"), rst=path.endswith(".rst"))
        title = next((b["text"] for b in blocks if b["type"] == "heading"), path)
        family = source.get("family_prefix", repo) + ":" + path
        return {"id": repo + ":" + path, "family_id": family, "title": title, "source_url": url,
                "license": source["license"], "license_evidence": [{"url": x["url"], "sha256": x["sha256"]} for x in source["license_evidence"]],
                "revision": commit, "language": source["language"], "domain": source["domain"], "kind": source["kind"],
                "blocks": blocks, "source_format": "rst" if path.endswith(".rst") else "markdown", "source_path": path,
                "parser": "source-markup-v1", "parser_limits": ["includes_not_expanded", "not_visual_layout_gold"]}, raw, url

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            source, file = futures[future]
            try:
                finish(*future.result())
            except (httpx.HTTPError, UnicodeError, ValueError) as exc:
                audit.append({"source": source["repo"], "path": file["path"], "status": "fetch_or_parse_error", "error": str(exc)[:500]})
            if i % 50 == 0:
                print(f"GitHub {i}/{len(jobs)}; retained {len(documents)}", flush=True)
    seen_pmc: set[str] = set()
    for source in lock["europe_pmc"]:
        retained = 0
        for article in source["articles"]:
            pmcid = article["pmcid"]
            if pmcid in seen_pmc or retained >= source["max_articles"]:
                continue
            seen_pmc.add(pmcid)
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            try:
                raw = get(client, url, cache, expected_hash=article.get("raw_sha256"))
                blocks, meta = parse_jats(raw.decode("utf-8"))
                license_id = normalize_cc(meta["license_links"], meta["license_text"])
                if license_id not in {"CC-BY-4.0", "CC-BY-3.0", "CC0-1.0", "CC-BY-SA-4.0"}:
                    audit.append({"source": pmcid, "status": "license_not_in_assessment_allowlist", "license_observed": license_id})
                    continue
                before = len(documents)
                finish({"id": pmcid, "family_id": article.get("doi", pmcid).lower(), "title": meta["title"] or article["title"],
                        "source_url": url, "license": license_id, "license_evidence": meta["license_links"], "license_text": meta["license_text"],
                        "revision": digest(raw), "language": "en", "domain": source["domain"], "kind": "research_article",
                        "blocks": blocks, "source_format": "jats_xml", "parser": meta["parser"],
                        "parser_limits": ["not_visual_layout_gold", "inline_structures_preserved_as_metadata_not_rendered"], "excluded_elements": meta["excluded_elements"]}, raw, url)
                retained += int(len(documents) > before)
                article["raw_sha256"] = digest(raw)
            except (httpx.HTTPError, ValueError, ET.ParseError, UnicodeError) as exc:
                audit.append({"source": pmcid, "status": "fetch_or_parse_error", "error": str(exc)[:500]})
            time.sleep(0.35)
        print(f"Europe PMC {source['domain']}: retained {retained}", flush=True)
    documents.sort(key=lambda d: d["id"])
    records.sort(key=lambda d: d["id"])
    write_jsonl(output / "documents.jsonl", documents)
    write_jsonl(output / "acquisition_audit.jsonl", audit)
    write_jsonl(output / "source_records.jsonl", records)
    # Lock the observed mutable XML bodies and actual tokenizer, not just search IDs.
    lock["tokenizer"]["sha256"] = digest(tokenizer_bytes)
    write_json(Path(args.lock), lock)
    counts = [d["token_count"] for d in documents]
    manifest = {"schema_version": "longdoc-acquisition-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "catalog_sha256": digest(CATALOG.read_bytes()), "source_lock_sha256": digest(Path(args.lock).read_bytes()),
                "documents_sha256": digest((output / "documents.jsonl").read_bytes()), "min_tokens": args.min_tokens,
                "implementation_sha256": implementation_hashes,
                "documents": len(documents), "unique_families": len({d["family_id"] for d in documents}),
                "tokens": sum(counts), "tokenizer": {**tokenizer_spec, "sha256": digest(tokenizer_bytes)},
                "by_language": dict(Counter(d["language"] for d in documents)), "by_domain": dict(Counter(d["domain"] for d in documents)),
                "by_kind": dict(Counter(d["kind"] for d in documents)),
                "by_length_tokens": {"4096-8191": sum(4096 <= t < 8192 for t in counts), "8192-16383": sum(8192 <= t < 16384 for t in counts),
                                     "16384-32767": sum(16384 <= t < 32768 for t in counts), "32768+": sum(t >= 32768 for t in counts)},
                "audit_statuses": dict(Counter(a["status"] for a in audit)), "human_verified_documents": 0,
                "limitations": ["source_markup_and_XML_only_not_real_PDF_layout_benchmark", "all_document_quality_reviews_pending", "not_CPT_training_results"]}
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if documents else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch pinned, public long-document sources (no LLM calls)")
    sp = ap.add_subparsers(dest="command", required=True)
    fetch = sp.add_parser("fetch")
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--lock", default=str(HERE / "sources.lock.json"))
    fetch.add_argument("--refresh-lock", action="store_true")
    fetch.add_argument("--min-tokens", type=int, default=4096)
    fetch.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.min_tokens < 1 or not 1 <= args.workers <= 12:
        ap.error("min-tokens must be positive; workers must be 1..12")
    return acquire(args)


if __name__ == "__main__":
    raise SystemExit(main())

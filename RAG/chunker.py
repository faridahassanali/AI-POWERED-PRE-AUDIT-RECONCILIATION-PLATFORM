"""
rag/chunker.py

Parses policy markdown files (data/0X_*.md) into structured chunks.
Chunking strategy: one chunk per `##` section, not per fixed word count.
This keeps each chunk semantically whole and traceable back to its
policy_id + section name, which the AI Output Validation step (Step 7)
will need to confirm cited policy sources actually exist.

Each chunk dict:
{
    "policy_id": "SCREENING-POLICY-001",
    "version": "1.0",
    "title": "Customer Screening & Wallet Activation Policy",
    "section": "Requirements",
    "content": "1. Every customer must have a screening record...",
    "source_file": "01_customer_screening_policy.md",
}
"""

import re
from pathlib import Path
from typing import List, Dict


def parse_policy_file(filepath: Path) -> List[Dict]:
    """Parse a single policy markdown file into a list of section chunks."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()

    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"Malformed policy file (missing H1 title): {filepath}")

    title = lines[0].lstrip("#").strip()

    policy_id_match = re.search(r"Policy ID:\s*(\S+)", text)
    version_match = re.search(r"Version:\s*([^\n(]+)", text)

    if not policy_id_match:
        raise ValueError(f"Malformed policy file (missing Policy ID): {filepath}")

    policy_id = policy_id_match.group(1).strip()
    version = version_match.group(1).strip() if version_match else "unknown"

    # Split on '## ' section headers, keeping the header text
    section_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(section_pattern.finditer(text))

    if not matches:
        raise ValueError(f"Malformed policy file (no ## sections found): {filepath}")

    chunks = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if not content:
            continue  # skip empty sections rather than silently indexing nothing

        chunks.append({
            "policy_id": policy_id,
            "version": version,
            "title": title,
            "section": section_name,
            "content": content,
            "source_file": filepath.name,
        })

    return chunks


def chunk_all_policies(policy_dir: str = "data") -> List[Dict]:
    """Parse every 0X_*.md policy file in policy_dir into chunks."""
    policy_dir_path = Path(policy_dir)
    policy_files = sorted(policy_dir_path.glob("0*_*.md"))

    if not policy_files:
        raise FileNotFoundError(f"No policy .md files found in {policy_dir}")

    all_chunks = []
    for filepath in policy_files:
        all_chunks.extend(parse_policy_file(filepath))

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_policies()
    print(f"Parsed {len(chunks)} chunks from policy files:")
    for c in chunks:
        print(f"  [{c['policy_id']}] {c['section']} ({len(c['content'])} chars)")
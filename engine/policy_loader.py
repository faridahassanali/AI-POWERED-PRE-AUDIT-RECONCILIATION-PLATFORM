from pathlib import Path
import re


class PolicyLoadError(ValueError):
    """Raised when a policy file cannot be parsed or validated."""


_POLICY_METADATA_RE = re.compile(
    r"^Policy ID:\s*(?P<policy_id>\S+)\s+"
    r"Version:\s*(?P<version>\S+)"
    r"(?:\s+\([^)]*\))?\s*$",
    re.MULTILINE,
)

_SECTION_RE = re.compile(
    r"^##\s+(?P<section>.+?)\s*$",
    re.MULTILINE,
)


def parse_policy(content: str, source: str = "<string>") -> dict:
    """
    Parse one policy Markdown document into a normalized policy object.

    Expected structure:

        # Policy Title

        Policy ID: SOME-ID Version: 1.0 (Synthetic)

        ## Section Name

        Section content...
    """

    if not isinstance(content, str):
        raise PolicyLoadError(
            f"Policy content from {source} must be a string."
        )

    if not content.strip():
        raise PolicyLoadError(
            f"Policy file {source} is empty."
        )

     
    ##Title
     
    title_match = re.search(
        r"^#\s+(?P<title>.+?)\s*$",
        content,
        re.MULTILINE,
    )

    if not title_match:
        raise PolicyLoadError(
            f"Policy file {source} is missing its title."
        )

    title = title_match.group("title")

     
    ## Policy ID + version
  
    metadata_match = _POLICY_METADATA_RE.search(content)

    if not metadata_match:
        raise PolicyLoadError(
            f"Policy file {source} has invalid or missing "
            "policy metadata."
        )

    policy_id = metadata_match.group("policy_id")
    version = metadata_match.group("version")

     
    ##Sections
   
    section_matches = list(_SECTION_RE.finditer(content))

    if not section_matches:
        raise PolicyLoadError(
            f"Policy file {source} must contain at least one section."
        )

    sections = []

    for index, match in enumerate(section_matches):
        section_name = match.group("section").strip()

        content_start = match.end()

        # Skip only the newline belonging to the section heading.
        if content_start < len(content):
            if content.startswith("\r\n", content_start):
                content_start += 2
            elif content.startswith("\n", content_start):
                content_start += 1

        if index + 1 < len(section_matches):
            content_end = section_matches[index + 1].start()
        else:
            content_end = len(content)

        section_content = content[content_start:content_end]

        sections.append(
            {
                "section": section_name,
                "content": section_content,
            }
        )

    policy = {
        "policy_id": policy_id,
        "version": version,
        "title": title,
        "sections": sections,
    }

    validate_policy(policy, source)

    return policy


def validate_policy(policy: dict, source: str = "<policy>") -> None:
    """Validate the normalized policy structure."""

    required_fields = {
        "policy_id",
        "version",
        "title",
        "sections",
    }

    missing = required_fields - policy.keys()

    if missing:
        raise PolicyLoadError(
            f"Policy {source} is missing fields: "
            f"{sorted(missing)}"
        )

    for field in ("policy_id", "version", "title"):
        value = policy[field]

        if not isinstance(value, str) or not value.strip():
            raise PolicyLoadError(
                f"Policy {source} has an invalid {field}."
            )

    if not isinstance(policy["sections"], list) or not policy["sections"]:
        raise PolicyLoadError(
            f"Policy {source} must contain at least one section."
        )

    seen_sections = set()

    for section in policy["sections"]:
        if not isinstance(section, dict):
            raise PolicyLoadError(
                f"Policy {source} contains an invalid section."
            )

        if "section" not in section or "content" not in section:
            raise PolicyLoadError(
                f"Policy {source} contains a section with "
                "missing fields."
            )

        name = section["section"]

        if not isinstance(name, str) or not name.strip():
            raise PolicyLoadError(
                f"Policy {source} contains a section with "
                "an invalid name."
            )

        if name in seen_sections:
            raise PolicyLoadError(
                f"Policy {source} contains duplicate section "
                f"{name!r}."
            )

        seen_sections.add(name)

        if not isinstance(section["content"], str):
            raise PolicyLoadError(
                f"Policy {source} section {name!r} has invalid content."
            )


def load_policy(path: Path) -> dict:
    """Load and parse one Markdown policy file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Policy file does not exist: {path}"
        )

    if not path.is_file():
        raise PolicyLoadError(
            f"Policy path is not a file: {path}"
        )

    if path.suffix.lower() != ".md":
        raise PolicyLoadError(
            f"Policy file must be Markdown: {path}"
        )

    content = path.read_text(encoding="utf-8")

    return parse_policy(
        content,
        source=str(path),
    )


def load_policies(data_dir: Path) -> list[dict]:
    """
    Load all numbered policy Markdown files from data_dir.

    Returns policies in filename order.
    """

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    policy_files = sorted(
        data_dir.glob("[0-9][0-9]_*.md")
    )

    if not policy_files:
        raise PolicyLoadError(
            f"No policy Markdown files found in {data_dir}"
        )

    policies = []

    for path in policy_files:
        policies.append(load_policy(path))

    return policies
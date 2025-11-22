# src/pyfundlib/ml/versioning.py
from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


class VersionBump(Enum):
    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"


class ModelVersion:
    """
    Semantic versioning for ML models with rich metadata and changelog.
    """

    VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")

    def __init__(
        self,
        major: int = 1,
        minor: int = 0,
        patch: int = 0,
        prerelease: str | None = None,
        build: str | None = None,
    ):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.prerelease = prerelease
        self.build = build

    @classmethod
    def parse(cls, version_str: str) -> ModelVersion:
        match = cls.VERSION_PATTERN.match(version_str.strip())
        if not match:
            raise ValueError(f"Invalid version string: {version_str}")
        major, minor, patch = map(int, match.groups()[:3])
        return cls(major, minor, patch)

    def bump(self, level: (VersionBump | str]) -> ModelVersion:
        level = VersionBump(level.lower())
        if level == VersionBump.MAJOR:
            return ModelVersion(self.major + 1, 0, 0)
        elif level == VersionBump.MINOR:
            return ModelVersion(self.major, self.minor + 1, 0)
        else:  # patch
            return ModelVersion(self.major, self.minor, self.patch + 1)

    def __str__(self) -> str:
        base = f"v{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += f"-{self.prerelease}"
        if self.build:
            base += f"+{self.build}"
        return base

    def __repr__(self) -> str:
        return f"ModelVersion({self!s})"

    def __lt__(self, other: ModelVersion) -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: ModelVersion) -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)


class ModelVersionManager:
    """
    Full model versioning system with changelog, lineage, and promotion workflow.
    """

    def __init__(self, versions_dir: str = "models/versions"):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.changelog_path = self.versions_dir / "CHANGELOG.json"

        if not self.changelog_path.exists():
            self._init_changelog()

    def _init_changelog(self):
        initial = {"project": "pyfundlib", "versions": [], "latest": None}
        self._save_changelog(initial)

    def _load_changelog(self) -> dict[str, Any]:
        if not self.changelog_path.exists():
            self._init_changelog()
        return json.loads(self.changelog_path.read_text())

    def _save_changelog(self, data: dict[str, Any]):
        self.changelog_path.write_text(json.dumps(data, indent=2))

    def register_version(
        self,
        model_name: str,
        version: (str | ModelVersion],
        description: str,
        author: str = "unknown",
        metrics: dict[str | float] | None = None,
        tags: list[str] | None = None,
        parent_version: str | None = None,
        is_breaking: bool = False,
    ) -> ModelVersion:
        """Register a new model version with full changelog entry"""
        if isinstance(version, str):
            version = ModelVersion.parse(version)

        changelog = self._load_changelog()

        entry = {
            "model_name": model_name,
            "version": str(version),
            "released_at": datetime.utcnow().isoformat(),
            "author": author,
            "description": description,
            "metrics": metrics or {},
            "tags": tags or [],
            "parent_version": parent_version,
            "breaking_change": is_breaking,
        }

        # Avoid duplicates
        if any(
            e["version"] == str(version) and e["model_name"] == model_name
            for e in changelog["versions"]
        ):
            logger.warning(f"Version {version} for {model_name} already exists")
            return version

        changelog["versions"].append(entry)
        changelog["latest"] = entry
        self._save_changelog(changelog)

        logger.info(f"Registered {model_name} {version} | {description}")
        return version

    def get_latest_version(self, model_name: str) -> dict[str, Any] | None:
        changelog = self._load_changelog()
        versions = [v for v in changelog["versions"] if v["model_name"] == model_name]
        if not versions:
            return None
        return max(versions, key=lambda x: ModelVersion.parse(x["version"]))

    def suggest_next_version(
        self,
        model_name: str,
        bump_level: (VersionBump | str] = VersionBump.PATCH,
        breaking: bool = False,
    ) -> ModelVersion:
        latest = self.get_latest_version(model_name)
        if not latest:
            return ModelVersion(1, 0, 0)

        current = ModelVersion.parse(latest["version"])
        if breaking:
            return current.bump(VersionBump.MAJOR)
        return current.bump(bump_level)

    def generate_changelog(self, model_name: str | None = None) -> str:
        changelog = self._load_changelog()
        lines = ["# Model Changelog\n"]

        versions = changelog["versions"]
        if model_name:
            versions = [v for v in versions if v["model_name"] == model_name]

        for entry in sorted(versions, key=lambda x: ModelVersion.parse(x["version"]), reverse=True):
            ver = entry["version"]
            lines.append(f"\n## {ver} - {entry['released_at'][:10]}")
            lines.append(f"*Author: {entry['author']}*")
            if entry["breaking_change"]:
                lines.append("**BREAKING CHANGE**")
            lines.append(f"{entry['description']}")
            if entry["metrics"]:
                metrics = " | ".join(f"{k}: {v:.4f}" for k, v in entry["metrics"].items())
                lines.append(f"Metrics → {metrics}")
            if entry["tags"]:
                lines.append(f"Tags: {', '.join(entry['tags'])}")

        return "\n".join(lines)

    def list_versions(self, model_name: str | None = None) -> pd.DataFrame:
        changelog = self._load_changelog()
        df = pd.DataFrame(changelog["versions"])
        if model_name:
            df = df[df["model_name"] == model_name]
        df["version_obj"] = df["version"].apply(ModelVersion.parse)
        return df.sort_values("version_obj", ascending=False).drop("version_obj", axis=1)


# Global version manager
version_manager = ModelVersionManager()

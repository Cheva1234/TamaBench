"""Environment Specification Loader and SHA-256 Spec Hash Generator for TamaBench V1."""

import hashlib
import os
from typing import Any


class EnvironmentSpecLoader:
    _cached_spec: dict[str, Any] = {}
    _cached_hash: str = ""

    @classmethod
    def get_spec_filepath(cls) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "environment_v1.yaml")

    @classmethod
    def load_spec(cls) -> dict[str, Any]:
        if cls._cached_spec:
            return cls._cached_spec

        filepath = cls.get_spec_filepath()
        with open(filepath, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Parse simple key-value YAML without external heavy dependencies if possible
        import pydantic
        # Compute SHA-256 hash of raw YAML file content
        content_bytes = raw_content.encode("utf-8")
        spec_hash = f"sha256:{hashlib.sha256(content_bytes).hexdigest()}"

        cls._cached_hash = spec_hash
        cls._cached_spec = {"raw_content": raw_content, "hash": spec_hash}
        return cls._cached_spec

    @classmethod
    def get_spec_hash(cls) -> str:
        if not cls._cached_hash:
            cls.load_spec()
        return cls._cached_hash

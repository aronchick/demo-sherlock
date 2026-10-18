"""Local Ed25519 signatures; public verification is not independent attestation."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload(record: dict) -> dict:
    result = copy.deepcopy(record)
    for field in ("signature", "verification"):
        result.pop(field, None)
    for artifact in result.get("artifacts", []):
        artifact.pop("integrity", None)
    return result


def digest(value: dict) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_key(root: Path) -> Ed25519PrivateKey:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".signing-key"
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        key = Ed25519PrivateKey.generate()
        data = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                 serialization.NoEncryption())
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            return load_key(root)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return key
    with os.fdopen(descriptor, "rb") as stream:
        if os.fstat(stream.fileno()).st_mode & 0o077:
            raise ValueError("signing key permissions must be private")
        return Ed25519PrivateKey.from_private_bytes(stream.read())


def sign(record: dict, root: Path, revision: int, previous: str | None, signed_at: float) -> dict:
    key = load_key(root)
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    metadata = {"algorithm": "Ed25519", "public_key": base64.b64encode(public).decode(),
                "key_fingerprint": hashlib.sha256(public).hexdigest(),
                "payload_sha256": digest(payload(record)), "revision": revision,
                "previous_sha256": previous, "signed_at": signed_at,
                "canonicalization": "UTF-8 JSON; sorted keys; compact separators; no NaN",
                "scope": "local application signer"}
    metadata["signature"] = base64.b64encode(key.sign(canonical(metadata))).decode()
    return {**payload(record), "signature": metadata}


def verify(record: dict) -> str:
    if "signature" not in record:
        return "unsigned"
    try:
        metadata = dict(record["signature"])
        signature = base64.b64decode(metadata.pop("signature"), validate=True)
        public = base64.b64decode(metadata["public_key"], validate=True)
        if (metadata["algorithm"] != "Ed25519"
                or metadata["payload_sha256"] != digest(payload(record))
                or metadata["key_fingerprint"] != hashlib.sha256(public).hexdigest()):
            return "invalid"
        Ed25519PublicKey.from_public_bytes(public).verify(signature, canonical(metadata))
    except (ValueError, TypeError, KeyError, InvalidSignature):
        return "invalid"
    return "verified"

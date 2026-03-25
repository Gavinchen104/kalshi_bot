from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from time import time
from urllib.parse import urlparse


@dataclass
class KalshiAuth:
    api_key_id: str
    private_key_raw: str

    def build_headers(self, method: str, full_url_or_path: str) -> dict[str, str]:
        timestamp = str(int(time() * 1000))
        path = _normalize_signing_path(full_url_or_path)
        msg = f"{timestamp}{method.upper()}{path}"
        signature = _sign_pss_base64(self.private_key_raw, msg)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }


def _normalize_signing_path(full_url_or_path: str) -> str:
    parsed = urlparse(full_url_or_path)
    path = parsed.path if parsed.scheme else full_url_or_path
    return path.split("?")[0]


def _sign_pss_base64(private_key_raw: str, message: str) -> str:
    private_key = _load_private_key(private_key_raw)
    msg_bytes = message.encode("utf-8")
    signature = private_key.sign(
        msg_bytes,
        _pss_padding(),
        _sha256_hash(),
    )
    return base64.b64encode(signature).decode("utf-8")


def _load_private_key(private_key_raw: str):
    # Lazy imports keep non-auth test paths lightweight.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend

    key_str = private_key_raw.strip()
    if len(key_str) < 240:
        path = Path(key_str).expanduser()
        try:
            if path.exists() and path.is_file():
                key_bytes = path.read_bytes()
                return _deserialize_key_bytes(key_bytes, serialization, default_backend)
        except OSError:
            pass

    if "BEGIN" in key_str:
        return _deserialize_key_bytes(key_str.encode("utf-8"), serialization, default_backend)

    # Some users store a base64 DER private key in env vars.
    try:
        compact = "".join(key_str.split())
        padding = "=" * ((4 - len(compact) % 4) % 4)
        der_bytes = base64.b64decode(compact + padding, validate=False)
        return serialization.load_der_private_key(
            der_bytes,
            password=None,
            backend=default_backend(),
        )
    except Exception as exc:
        raise ValueError(
            "Unable to parse KALSHI private key. Provide PEM text, key file path, or base64 DER."
        ) from exc


def _deserialize_key_bytes(key_bytes, serialization, default_backend):
    try:
        return serialization.load_pem_private_key(
            key_bytes,
            password=None,
            backend=default_backend(),
        )
    except Exception:
        return serialization.load_der_private_key(
            key_bytes,
            password=None,
            backend=default_backend(),
        )


def _pss_padding():
    from cryptography.hazmat.primitives.asymmetric import padding

    return padding.PSS(
        mgf=padding.MGF1(_sha256_hash()),
        salt_length=padding.PSS.DIGEST_LENGTH,
    )


def _sha256_hash():
    from cryptography.hazmat.primitives import hashes

    return hashes.SHA256()


# Redactions

Copies of files from the training machine have had host-specific strings replaced so that no server address, username or home path is published. Only those strings were replaced. No logic, number, key, hash field or other content was changed; JSON files still parse and the Python file still compiles. Original hashes are the ones recorded in the delivery provenance record (for the two receipt files, the delivery manifest). Hash fields inside the JSON files were left as originally recorded, so they describe the unredacted originals.

| File | Original sha256 | Redacted sha256 | Replaced |
|---|---|---|---|
| `experiments/final_protocol_v3.json` | `304d11814d8f52aec21cda1bea5bcfc0b8f8ea309e61781293cd60c862bb38ed` | `a4e268252a014229064f7ae426d50f43f60a4d603f1a24508bed98fbb831c558` | the GPU host IP -> `<GPU_HOST>`; the host username -> `<USER>` |
| `experiments/x1_protocol_v2.json` | `6390994a878422a1079cadeb42308fc89c868b41902fc0a3a8d699004ce9e9b3` | `ac580476eb9e75de8bf03ce93f9594afad8395ccf002b3e59128a214b9b2c938` | the GPU host IP -> `<GPU_HOST>`; the host username -> `<USER>` |
| `scripts/release_p1.py` | `7c81db4a8d3e100da12c48cdcb3986ca0bbc3d48fa462ca8a3aba918c5022b88` | `b8cbc37e838a740c2b3c1e20c7d07f7435ae2f897a7d0fbee34bf5c4513c1630` | the host home directory prefix (including the username) -> `<HOME>` |
| `experiments/l1_protocol.json` | `b05689a738fde3afd3f90acd4eb5dec2319cc1b2ba553b4ae44911fcaf550c8e` | `e54d815e6c6207b38cf471d3ad6c3c00312ca065d64f5c98fbe4ebb4766f726b` | the wording about personal remote-login authentication -> `<private access>` |

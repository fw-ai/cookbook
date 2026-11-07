# DeepSeek V4.1 publisher encoder

`encoding.py` was moved from the existing supervised-renderer test reference
into runtime package source. Its publisher payload (after the seven-line
provenance header) is unchanged.

| Field | Value |
|---|---|
| Source | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Revision | `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` |
| Upstream path | `encoding/encoding.py` |
| Payload SHA256 | `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1` |
| License | MIT, copyright 2023 DeepSeek; see `LICENSE` |

Static review: publisher payload is 979 lines; imports only `typing`, `copy`,
`json`, and `re`. No dynamic execution, imports, subprocess or network access.
Module initialization defines constants, regexes and functions. The unused
`load_cases(input_file)` helper reads JSON from a caller-provided file; TITO
does not call it. The runtime parser uses only string/JSON/regex operations.

Do not edit the publisher payload. A revision update requires explicit source
review, matching digest/provenance updates and parent-versus-candidate
tokenizer/template/parser equivalence evidence.

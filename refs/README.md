# Reference bundles

Reference result bundles of the abTEM benchmark suite (`benchmarks/abtem_bench` in the abTEM repository), captured at release tags and published as release assets of this repository. Design: `../design/DESIGN.md`, section 10.

Naming: `refs-<tag>-<tier>-<device>-<preset>-<fingerprint8>.tar.zst`, attached to the release `refs-<tag>`. `INDEX.json` lists every asset with the key manifest fields (tag, sha, tier, device, preset, fingerprint, abtem version, capture date, machine). It is updated by `abtem-bench refs upload`.

Fetch: `abtem-bench refs fetch <tag> --tier quick --device cpu` downloads the asset matching the local machine fingerprint into `~/.cache/abtem-bench/refs/` and prints the manifest. When no fingerprint matches, the closest same-device bundle is used and compare downgrades the bit-exact expectation to the float64 tolerance.

Nothing is committed here except `INDEX.json`; bundles are release assets only, so every clone of this repository stays small.

Planned first release: `refs-v1.0.10` with the quick-tier CPU bundle from the dev box (M1) and a GPU bundle from a Perlmutter A100 (M3).

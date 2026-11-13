#!/usr/bin/env bash
set -euo pipefail

version="${1:?usage: build_marin_release.sh X.Y.Z OUTPUT_DIRECTORY}"
output="$(realpath -m "${2:?usage: build_marin_release.sh X.Y.Z OUTPUT_DIRECTORY}")"
root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
commit="$(git -C "$root" rev-parse HEAD)"

# Both upstream build backends derive their version from their own release tag.
for tag in "v$version" "hsc-v$version"; do
    test "$(git -C "$root" rev-parse "$tag^{commit}")" = "$commit"
done
git -C "$root" diff-index --quiet HEAD --
export SPECULATORS_BUILD_TYPE=release HS_CONNECTORS_BUILD_TYPE=release
uv build --wheel --out-dir "$output" "$root/hs_connectors"
uv build --wheel --out-dir "$output" "$root"
printf '%s\n' "$commit" > "$output/SOURCE_COMMIT"
(cd "$output" && sha256sum "hs_connectors-$version-"*.whl "speculators-$version-"*.whl > SHA256SUMS)

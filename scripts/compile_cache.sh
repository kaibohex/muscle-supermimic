#!/bin/bash
# Pack or unpack this machine's compile caches: torch's Inductor / AOTAutograd / Triton cache (/tmp/torchinductor_$USER),
# Warp's kernel cache and the CUDA JIT cache. A new machine of the same image that loads them before training skips
# the compilation a cold run spends inside the training clock (trial t17: the first policy update came after 10.6
# minutes instead of ~1). The caches key on code, shapes, versions and GPU, so save them from a machine that has
# run the configuration to be trained, with the same code and lockfile.
#
# Usage, on the training machine (HF_TOKEN in the environment; the archive goes to $HF_RESULTS_REPO/compile_cache/):
#   bash scripts/compile_cache.sh save <name>
#   bash scripts/compile_cache.sh load <name>
set -euo pipefail
cd ~/sky_workdir
REPO=${HF_RESULTS_REPO:-YOUR_HF_USER/supermimic-runs}
NAME=${2:?usage: bash scripts/compile_cache.sh save|load <name>}
ARCHIVE=/tmp/compile_cache_$NAME.tar.zst
DIRS=("/tmp/torchinductor_$(whoami)" "$HOME/.cache/warp" "$HOME/.nv/ComputeCache" "$HOME/.triton")
t0=$(date +%s)
case "${1:-}" in
  save)
    present=()
    for d in "${DIRS[@]}"; do [ -e "$d" ] && present+=("${d#/}"); done
    tar -C / -cf - "${present[@]}" | zstd -T0 -q -f -o "$ARCHIVE"
    echo "compile_cache: packed ${present[*]} into $(du -h "$ARCHIVE" | cut -f1)"
    .venv/bin/hf upload "$REPO" "$ARCHIVE" "compile_cache/$NAME.tar.zst" --repo-type dataset --quiet
    echo "compile_cache: uploaded to $REPO/compile_cache/$NAME.tar.zst in $(( $(date +%s) - t0 )) s"
    ;;
  load)
    .venv/bin/hf download "$REPO" "compile_cache/$NAME.tar.zst" --repo-type dataset --local-dir /tmp/compile_cache_dl --quiet >/dev/null
    zstd -d -q -c "/tmp/compile_cache_dl/compile_cache/$NAME.tar.zst" | tar -C / -xf -
    rm -f "/tmp/compile_cache_dl/compile_cache/$NAME.tar.zst"
    echo "compile_cache: loaded $NAME in $(( $(date +%s) - t0 )) s: $(du -sh "${DIRS[@]}" 2>/dev/null | awk '{print $2 " " $1}' | tr '\n' ' ')"
    ;;
  *) echo "usage: bash scripts/compile_cache.sh save|load <name>" >&2; exit 2 ;;
esac

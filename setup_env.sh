#!/usr/bin/env bash
# Source this file to configure the MapTune + gradmap environment.
# Usage:  source setup_env.sh
#
# Fill in the three TODO paths below, then run once per shell session.

# ------------------------------------------------------------------
# TODO: path to directory that contains the `abc` binary
# ------------------------------------------------------------------
ABC_BIN_DIR=""   # e.g. /home/user/abc  or  /opt/abc

# ------------------------------------------------------------------
# TODO: path to gradmap external libs
#   must contain: asap7.lib  and  rec6Lib_final_filtered3_recanon.aig
# ------------------------------------------------------------------
GRADMAP_LIBS_DIR=""   # e.g. /home/user/gradmap_libs

# ------------------------------------------------------------------
# TODO (optional): override the gradmap binary path
#   default is third_party/gradmap/build/gradmap_torch
# ------------------------------------------------------------------
# export GRADMAP_RUNNER=/custom/path/to/gradmap_torch

# ------------------------------------------------------------------
# Apply
# ------------------------------------------------------------------

if [[ -z "$ABC_BIN_DIR" || -z "$GRADMAP_LIBS_DIR" ]]; then
    echo "[setup_env] ERROR: fill in ABC_BIN_DIR and GRADMAP_LIBS_DIR above"
    return 1 2>/dev/null || exit 1
fi

export PATH="$ABC_BIN_DIR:$PATH"
export GRADMAP_LIBS="$GRADMAP_LIBS_DIR"
export ABC_PATH="$ABC_BIN_DIR"

# ------------------------------------------------------------------
# Verify
# ------------------------------------------------------------------
echo "[setup_env] Checking dependencies..."

ok=1

if command -v abc &>/dev/null; then
    echo "  [ok] abc found: $(which abc)"
else
    echo "  [MISSING] abc not on PATH — check ABC_BIN_DIR"
    ok=0
fi

for f in "$GRADMAP_LIBS/asap7.lib" "$GRADMAP_LIBS/rec6Lib_final_filtered3_recanon.aig"; do
    if [[ -f "$f" ]]; then
        echo "  [ok] $f"
    else
        echo "  [MISSING] $f"
        ok=0
    fi
done

GRADMAP_BINARY="${GRADMAP_RUNNER:-$(dirname "$0")/third_party/gradmap/build/gradmap_torch}"
if [[ -x "$GRADMAP_BINARY" ]]; then
    echo "  [ok] gradmap binary: $GRADMAP_BINARY"
else
    echo "  [not built] gradmap binary: $GRADMAP_BINARY"
    echo "              build with: cd third_party/gradmap && bash compile.sh"
fi

LIBCELL="$(dirname "$0")/third_party/gradmap/libs/asap7_libcell_info.txt"
if [[ -f "$LIBCELL" ]]; then
    echo "  [ok] asap7_libcell_info.txt"
else
    echo "  [missing] $LIBCELL"
    echo "            generate with:"
    echo "            python third_party/gradmap/libs/lut.py \\"
    echo "              -files \$GRADMAP_LIBS/asap7*.lib \\"
    echo "              -o third_party/gradmap/libs/asap7_libcell_info.txt"
fi

[[ $ok -eq 1 ]] && echo "[setup_env] All required deps found." || echo "[setup_env] Fix missing deps above."

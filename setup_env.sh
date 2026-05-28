#!/usr/bin/env bash
# Source this file to configure the MapTune + gradmap environment.
# Usage:  source setup_env.sh
#
# Fill in the three TODO paths below, then run once per shell session.

# ------------------------------------------------------------------
# TODO: path to directory that contains the `abc` binary
# ------------------------------------------------------------------
ABC_BIN_DIR="/Users/williamsu/tools/abc"

# ------------------------------------------------------------------
# Liberty .lib for ABC match generation and asap7_libcell_info.txt.
# Set GRADMAP_LIB to a single .lib file path (preferred), or set
# GRADMAP_LIBS to a directory containing asap7.lib (original convention).
# ------------------------------------------------------------------
GRADMAP_LIB_FILE="/Users/williamsu/Documents/ntu/lecture/32/project/MapTune/libs/7nm.lib"
GRADMAP_LIBS_DIR=""   # leave empty when using GRADMAP_LIB_FILE above

# ------------------------------------------------------------------
# TODO (optional): override the gradmap binary path
#   default is third_party/gradmap/build/gradmap_torch
# ------------------------------------------------------------------
# export GRADMAP_RUNNER=/custom/path/to/gradmap_torch

# ------------------------------------------------------------------
# Apply
# ------------------------------------------------------------------

if [[ -z "$ABC_BIN_DIR" ]]; then
    echo "[setup_env] ERROR: fill in ABC_BIN_DIR above"
    return 1 2>/dev/null || exit 1
fi
if [[ -z "$GRADMAP_LIB_FILE" && -z "$GRADMAP_LIBS_DIR" ]]; then
    echo "[setup_env] ERROR: set either GRADMAP_LIB_FILE or GRADMAP_LIBS_DIR above"
    return 1 2>/dev/null || exit 1
fi

export PATH="$ABC_BIN_DIR:$PATH"
export ABC_PATH="$ABC_BIN_DIR"
[[ -n "$GRADMAP_LIB_FILE" ]] && export GRADMAP_LIB="$GRADMAP_LIB_FILE"
[[ -n "$GRADMAP_LIBS_DIR" ]] && export GRADMAP_LIBS="$GRADMAP_LIBS_DIR"

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

if [[ -n "$GRADMAP_LIB" ]]; then
    if [[ -f "$GRADMAP_LIB" ]]; then
        echo "  [ok] GRADMAP_LIB=$GRADMAP_LIB"
    else
        echo "  [MISSING] GRADMAP_LIB=$GRADMAP_LIB"
        ok=0
    fi
elif [[ -n "$GRADMAP_LIBS" ]]; then
    if [[ -f "$GRADMAP_LIBS/asap7.lib" ]]; then
        echo "  [ok] $GRADMAP_LIBS/asap7.lib"
    else
        echo "  [MISSING] $GRADMAP_LIBS/asap7.lib"
        ok=0
    fi
fi

GRADMAP_BINARY="${GRADMAP_RUNNER:-}"
if [[ -z "$GRADMAP_BINARY" ]]; then
    for _candidate in "$(dirname "$0")/third_party/gradmap/gradmap_torch" \
                      "$(dirname "$0")/third_party/gradmap/build/gradmap_torch"; do
        [[ -x "$_candidate" ]] && GRADMAP_BINARY="$_candidate" && break
    done
    : "${GRADMAP_BINARY:=$(dirname "$0")/third_party/gradmap/gradmap_torch}"
fi
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

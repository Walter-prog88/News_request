#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
UV="$ROOT/.tools/uv"

if [ ! -x "$UV" ]; then
  mkdir -p "$ROOT/.tools"
  curl -LsSf https://astral.sh/uv/0.11.28/install.sh \
    | env UV_UNMANAGED_INSTALL="$ROOT/.tools" sh
fi

"$UV" python install 3.11
"$UV" sync --extra dev
echo "Ready. Run: $UV run index-timing --help"


#!/usr/bin/env bash
# Velox quick installer for macOS Apple Silicon.
#
# Builds Velox from source via cargo and copies the binary into
# ~/.local/bin (or /usr/local/bin if --system is passed).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Soflutionltd/velox/main/install.sh | sh
#   ./install.sh                      # local clone, install to ~/.local/bin
#   ./install.sh --system             # install to /usr/local/bin (sudo)
#
# Requirements:
#   - macOS 14+ on Apple Silicon (M1/M2/M3/M4)
#   - Xcode Command Line Tools (`xcode-select --install`)
#   - Rust 1.78+ (`https://rustup.rs`)
#   - protoc (`brew install protobuf`)

set -euo pipefail

REPO="https://github.com/Soflutionltd/velox"
INSTALL_DIR="${HOME}/.local/bin"
SYSTEM_INSTALL=0

for arg in "$@"; do
  case "$arg" in
    --system) SYSTEM_INSTALL=1 ;;
    --prefix=*) INSTALL_DIR="${arg#*=}" ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
  esac
done

if [[ "$SYSTEM_INSTALL" -eq 1 ]]; then
  INSTALL_DIR="/usr/local/bin"
fi

# Sanity checks.
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Velox is currently macOS-only (Apple Silicon)." >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Velox requires Apple Silicon (arm64). Detected: $(uname -m)" >&2
  exit 1
fi
command -v cargo >/dev/null 2>&1 || {
  echo "cargo not found. Install Rust: https://rustup.rs" >&2
  exit 1
}
command -v protoc >/dev/null 2>&1 || {
  echo "protoc not found. Install: brew install protobuf" >&2
  exit 1
}

# Resolve workdir: existing checkout next to the script, or a fresh
# clone in a temp directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/Cargo.toml" ]]; then
  WORKDIR="$SCRIPT_DIR"
else
  WORKDIR="$(mktemp -d -t velox-install-XXXXXX)"
  echo "Cloning $REPO into $WORKDIR …"
  git clone --depth 1 "$REPO" "$WORKDIR"
fi

echo "Building Velox (release, candle-metal) …"
( cd "$WORKDIR" && cargo build --release --features candle-metal )

mkdir -p "$INSTALL_DIR"
cp "$WORKDIR/target/release/velox" "$INSTALL_DIR/velox"
chmod +x "$INSTALL_DIR/velox"

echo
echo "Installed: $INSTALL_DIR/velox"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) echo "  Note: $INSTALL_DIR is not on your PATH. Add it:" >&2
     echo "    echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc" >&2 ;;
esac

echo
echo "Quick start:"
echo "  velox serve --model-dir ~/.velox/models"
echo

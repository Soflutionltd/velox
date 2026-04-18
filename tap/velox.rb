# Velox Homebrew Formula.
#
# Usage (after the tap is published):
#
#     brew tap soflutionltd/tap https://github.com/Soflutionltd/velox.git
#     brew install velox
#
# Or directly from the workspace checkout for testing:
#
#     brew install --build-from-source ./tap/velox.rb
class Velox < Formula
  desc "World's first Rust-native LLM inference server for Apple Silicon"
  homepage "https://github.com/Soflutionltd/velox"
  url "https://github.com/Soflutionltd/velox/archive/refs/heads/main.tar.gz"
  version "0.1.0"
  license "Apache-2.0"
  head "https://github.com/Soflutionltd/velox.git", branch: "main"

  depends_on "rust" => :build
  depends_on "protobuf" => :build
  depends_on macos: :sequoia
  depends_on arch: :arm64

  def install
    # Use Apple Silicon Metal acceleration by default. mistral.rs and
    # CUDA features stay off for the published formula — users who
    # need them rebuild from source with extra --features.
    system "cargo", "install", *std_cargo_args, "--features", "candle-metal"
  end

  def caveats
    <<~EOS
      Velox stores models at:
        ~/.velox/models

      Quick start:
        # Download a small model (4-bit, 600MB):
        hf download mlx-community/Qwen3-0.6B-4bit \\
          --local-dir ~/.velox/models/Qwen3-0.6B-4bit

        # Start the server:
        velox serve --model-dir ~/.velox/models

        # Hit it (OpenAI-compatible):
        curl http://localhost:8000/v1/chat/completions \\
          -H 'Content-Type: application/json' \\
          -d '{"model":"Qwen3-0.6B-4bit","messages":[{"role":"user","content":"Hi"}]}'

      Optional transports:
        --socket /tmp/velox.sock    # Unix domain socket (~30µs faster)
        --grpc-port 50051           # typed gRPC + HTTP/2 streaming
    EOS
  end

  test do
    assert_match "velox", shell_output("#{bin}/velox --help")
  end
end

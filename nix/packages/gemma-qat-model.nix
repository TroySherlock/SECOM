# Unsloth Gemma 4 12B QAT GGUF (~6.7 GB).
# HF resolve URL is public; re-run `nix run .#prefetch-gemma-hash` after upstream updates.
{
  lib,
  pkgs,
  fetchurl,
  modelUrl ? "https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF/resolve/main/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
  modelHash ? "sha256-000000000000000000000000000000000000000000000000000=",
  localPath ? null,
}:

if localPath != null then
  lib.cleanSource localPath
else
  fetchurl {
    url = modelUrl;
    name = "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
    hash = modelHash;
  }

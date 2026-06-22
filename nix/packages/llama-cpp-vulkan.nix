# Pinned llama.cpp with Vulkan (Gemma 4 / gemma4 architecture).
{ lib, pkgs, llamaCppRev ? "b9538", llamaCppHash ? "sha256-7cE5l+fnVsw72UyqAqKP2UmKg2seMCcqsZSKhHBSKLM=" }:

let
  llamaCppSrc = pkgs.fetchFromGitHub {
    owner = "ggml-org";
    repo = "llama.cpp";
    tag = llamaCppRev;
    hash = llamaCppHash;
  };
in
(pkgs.llama-cpp.override {
  vulkanSupport = true;
  blasSupport = true;
}).overrideAttrs (old: {
  pname = "llama-cpp-vulkan";
  version = llamaCppRev;
  src = llamaCppSrc;

  cmakeFlags =
    (old.cmakeFlags or [ ])
    ++ [
      (lib.cmakeBool "GGML_NATIVE" true)
      (lib.cmakeBool "LLAMA_BUILD_SERVER" true)
      (lib.cmakeBool "LLAMA_BUILD_UI" false)
    ];

  preConfigure = ''
    export NIX_ENFORCE_NO_NATIVE=0
    ${old.preConfigure or ""}
  '';
})

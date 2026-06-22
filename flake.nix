{
  description = "SECOM project with local Gemma 4 QAT inference (llama.cpp Vulkan)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      # Set after: nix run .#prefetch-gemma-hash
      gemmaModelHash = "sha256-000000000000000000000000000000000000000000000000000=";

      llama-cpp-vulkan = pkgs.callPackage ./nix/packages/llama-cpp-vulkan.nix { };
      gemma-qat-gguf = pkgs.callPackage ./nix/packages/gemma-qat-model.nix {
        modelHash = gemmaModelHash;
      };

      llamaServerArgs = [
        "--model"
        "${gemma-qat-gguf}"
        "--host"
        "127.0.0.1"
        "--port"
        "8080"
        "--ctx-size"
        "8192"
        "--n-gpu-layers"
        "99"
        "--device"
        "Vulkan0"
        "--parallel"
        "1"
        "--threads"
        "8"
        "--jinja"
        "--no-mmproj"
        "--flash-attn"
        "on"
      ];

      nixosModules = {
        llama-gemma = import ./nix/modules/llama-gemma.nix;
      };

    in
    {
      inherit nixosModules;

      nixosConfigurations.secom = nixpkgs.lib.nixosSystem {
        inherit system;
        specialArgs = {
          inherit llama-cpp-vulkan gemma-qat-gguf;
        };
        modules = [
          ./nix/configuration.nix
          {
            services.llama-gemma = {
              package = llama-cpp-vulkan;
              modelPath = gemma-qat-gguf;
            };
          }
        ];
      };

      packages.${system} = {
        inherit llama-cpp-vulkan gemma-qat-gguf;
        default = llama-cpp-vulkan;
      };

      apps.${system} = {
        gemma-serve = {
          type = "app";
          program = pkgs.writeShellScript "gemma-serve" ''
            export AMD_VULKAN_ICD=RADV
            export RADV_PERFTEST=aco
            export GGML_VK_VISIBLE_DEVICES=0
            exec ${llama-cpp-vulkan}/bin/llama-server ${builtins.concatStringsSep " " (map pkgs.lib.escapeShellArg llamaServerArgs)}
          '';
        };

        gemma-chat-test = {
          type = "app";
          program = pkgs.writeShellScript "gemma-chat-test" ''
            export PATH=${pkgs.lib.makeBinPath [ pkgs.curl pkgs.jq ]}:$PATH
            ${builtins.readFile ./scripts/gemma-smoke-test.sh}
          '';
        };

        prefetch-gemma-hash = {
          type = "app";
          program = "${pkgs.writeShellScript "prefetch-gemma-hash" ''
            set -euo pipefail
            url="https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF/resolve/main/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
            echo "Prefetching ~6.7 GB GGUF (may take several minutes)..."
            hash=$(${pkgs.nix}/bin/nix-prefetch-url "$url" --name gemma-4-12B-it-qat-UD-Q4_K_XL.gguf)
            echo
            echo "Add to flake.nix gemmaModelHash:"
            echo "  gemmaModelHash = \"$hash\";"
          ''}";
        };

        fetch-gemma-local = {
          type = "app";
          program = "${pkgs.writeShellScript "fetch-gemma-local" ''
            set -euo pipefail
            dest="''${1:-./models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf}"
            mkdir -p "$(dirname "$dest")"
            url="https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF/resolve/main/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
            token=""
            if [ -n "''${HF_TOKEN:-}" ]; then
              token="$HF_TOKEN"
            elif [ -f "$HOME/.config/huggingface/token" ]; then
              token="$(tr -d '[:space:]' < "$HOME/.config/huggingface/token")"
            fi
            auth=()
            if [ -n "$token" ]; then
              auth=(-H "Authorization: Bearer $token")
            fi
            echo "Downloading to $dest ..."
            ${pkgs.curl}/bin/curl -L "''${auth[@]}" -o "$dest" "$url"
            echo "Done. Build with: GEMMA_LOCAL=$dest nix build .#gemma-qat-gguf"
          ''}";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          curl
          jq
          python311
          uv
          ruff
          zlib
        ];

        shellHook = ''
          export UV_PYTHON_PREFERENCE=system
          export VIRTUAL_ENV_DISABLE_PROMPT=2
          export DBT_PROFILES_DIR=$(pwd)
          export LD_LIBRARY_PATH=${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
          export OPENAI_API_KEY=local

          if [ -f .env ]; then
            source .env
          fi

          if [ ! -d .venv ]; then
            uv venv
          fi
          source .venv/bin/activate
        '';
      };

      formatter.${system} = pkgs.nixfmt-rfc-style;
    };
}

{
  description = "SECOM project development shell";

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
      };
    in
    {
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

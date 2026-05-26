{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; }; } }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    python311Packages.xgboost
    uv
    ruff
    zlib
    stdenv.cc.cc.lib
  ];

  shellHook = ''
    export UV_PYTHON_PREFERENCE="system"
    export VIRTUAL_ENV_DISABLE_PROMPT=2
    export DBT_PROFILES_DIR=$(pwd)

    export LD_LIBRARY_PATH=${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

    if [ -f .env ]; then
      echo "🔑 Loading local environment variables..."
      source .env
    fi

    if [ ! -d ".venv" ]; then
      echo "⚡ Creating virtual environment with uv..."
      uv venv
    fi

    source .venv/bin/activate

    echo "⚡ Installing dbt-duckdb..."
    uv pip install dbt-duckdb pandas ipykernel duckdb scikit-learn feature-engine xgboost tqdm tqdm-joblib
    echo "✅ SECOM Environment Ready (DuckDB + dbt)"
  '';
}

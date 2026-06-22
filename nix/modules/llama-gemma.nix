{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.llama-gemma;
in
{
  options.services.llama-gemma = {
    enable = lib.mkEnableOption "Local Gemma 4 QAT llama-server (Vulkan)";

    package = lib.mkOption {
      type = lib.types.package;
      description = "llama.cpp build with Vulkan support.";
    };

    modelPath = lib.mkOption {
      type = lib.types.path;
      description = "Path to gemma-4-12B-it-qat-UD-Q4_K_XL.gguf.";
    };

    huggingfaceTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = ''
        Optional HF token file for manual model downloads (not used by the
        systemd service when modelPath points at a store path).
      '';
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
    };

    contextSize = lib.mkOption {
      type = lib.types.int;
      default = 8192;
    };

    gpuLayers = lib.mkOption {
      type = lib.types.int;
      default = 99;
      description = "GPU layer offload count (-1 / 99 = all layers on GPU).";
    };

    vulkanDevice = lib.mkOption {
      type = lib.types.str;
      default = "Vulkan0";
    };

    parallelSlots = lib.mkOption {
      type = lib.types.int;
      default = 1;
    };

    threads = lib.mkOption {
      type = lib.types.int;
      default = 8;
    };

    useJinja = lib.mkOption {
      type = lib.types.bool;
      default = true;
    };

    noMmproj = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Text-only SECOM narratives; skip multimodal projector.";
    };

    flashAttention = lib.mkOption {
      type = lib.types.bool;
      default = true;
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
    };
  };

  config = lib.mkIf cfg.enable {
    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];

    systemd.services.llama-gemma = {
      description = "Gemma 4 QAT llama-server (Vulkan, OpenAI API)";
      after = [
        "network.target"
        "systemd-udev-settle.service"
      ];
      wantedBy = [ "multi-user.target" ];

      environment = cfg.environment // {
        AMD_VULKAN_ICD = "RADV";
        RADV_PERFTEST = "aco";
        GGML_VK_VISIBLE_DEVICES = "0";
      };

      serviceConfig = {
        Type = "simple";
        User = "llama-gemma";
        Group = "llama-gemma";
        StateDirectory = "llama-gemma";
        CacheDirectory = "llama-gemma";
        Restart = "on-failure";
        RestartSec = "5s";
        ExecStartPre = "${pkgs.coreutils}/bin/sleep 2";
        ExecStart = lib.escapeShellArgs (
          [
            "${cfg.package}/bin/llama-server"
            "--model"
            (toString cfg.modelPath)
            "--host"
            cfg.host
            "--port"
            (toString cfg.port)
            "--ctx-size"
            (toString cfg.contextSize)
            "--n-gpu-layers"
            (toString cfg.gpuLayers)
            "--device"
            cfg.vulkanDevice
            "--parallel"
            (toString cfg.parallelSlots)
            "--threads"
            (toString cfg.threads)
          ]
          ++ lib.optional cfg.useJinja "--jinja"
          ++ lib.optional cfg.noMmproj "--no-mmproj"
          ++ lib.optional cfg.flashAttention "--flash-attn"
          ++ lib.optional cfg.flashAttention "on"
          ++ cfg.extraArgs
        );

        DeviceAllow = [ "char-drm rw" ];
        PrivateDevices = false;

        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/var/cache/llama-gemma" ];
      };
    };
  };
}

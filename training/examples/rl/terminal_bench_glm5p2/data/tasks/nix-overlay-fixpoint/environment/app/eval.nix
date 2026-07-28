# Main evaluation entrypoint
# Combines base config, module configs, and overlays into final deployment config
let
  lib = import ./lib.nix;

  # Base configuration with defaults
  base = {
    networking = {
      hostName = "localhost";
      domain = "example.com";
      interfaces = {};
      firewall = {
        enable = false;
        allowedTCPPorts = [];
        defaultPolicy = "ACCEPT";
        logRefused = false;
      };
      nameservers = [];
    };
    services = {};
    users = {};
  };

  # Configuration modules to merge
  modules = [
    (import ./modules/network.nix)
    (import ./modules/services.nix)
    (import ./modules/security.nix)
    (import ./modules/users.nix)
  ];

  # Overlays to apply after merging
  overlays = [
    (import ./overlays/customization.nix)
    (import ./overlays/hardening.nix)
  ];

  # Step 1: Evaluate all modules and merge their configs
  moduleConfig = lib.foldAttrs (builtins.map (m: m { inherit lib; }) modules);

  # Step 2: Merge base defaults with module config
  mergedBase = lib.recursiveUpdate moduleConfig base;

  # Step 3: Apply overlays using fixpoint composition
  finalConfig = lib.applyOverlays mergedBase overlays;

in finalConfig

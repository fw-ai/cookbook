#!/usr/bin/env python3
"""Create the buggy NixOS-style config evaluator app files at /app/."""
import os

def w(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

# ── eval.py (Bug 4: merge order inverted) ──────────────────────────────────
w("/app/eval.py", r'''#!/usr/bin/env python3
"""
NixOS-style deployment configuration evaluator.

Merges modular configuration fragments using recursive attribute set
merging, then applies deployment overlays using the self/super pattern
to produce the final configuration JSON.

Usage: python3 eval.py
"""
import json
import copy
from lib import recursive_update, apply_overlays
from modules.base import config as base_config
from modules.networking import config as networking_config
from modules.services import config as services_config
from modules.security import config as security_config
from modules.users import config as users_config
from overlays.customization import overlay as customization_overlay
from overlays.monitoring import overlay as monitoring_overlay


def evaluate():
    """Evaluate all modules and overlays to produce deployment config."""
    modules = [
        networking_config,
        services_config,
        security_config,
        users_config,
    ]

    # Start with base defaults, then layer each module on top.
    # Module values take precedence over the accumulated config.
    config = copy.deepcopy(base_config)
    for module in modules:
        config = recursive_update(module, config)

    # Apply deployment overlays
    overlays = [customization_overlay, monitoring_overlay]
    config = apply_overlays(config, overlays)

    return config


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, indent=2, sort_keys=True))
'''.lstrip())

# ── lib.py (Bug 1: missing list concat; Bug 2: shallow merge in overlays) ──
w("/app/lib.py", r'''"""
Core library for NixOS-style configuration evaluation.

Provides recursive attribute set merging and overlay application --
the fundamental operations behind the NixOS module system and the
nixpkgs overlay mechanism.
"""
import copy


def recursive_update(base, override):
    """Recursively merge *override* into *base*, producing a new dict.

    Merge semantics (mirrors ``lib.recursiveUpdate`` in nixpkgs):

    * Both values are dicts  -> merge recursively.
    * Both values are lists  -> concatenate (``base ++ override``).
    * Otherwise              -> *override* replaces *base*.

    Neither input dict is mutated.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_update(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def apply_overlays(config, overlays):
    """Apply a chain of overlays to a merged module configuration.

    Each overlay is a callable ``(self_ref, super_ref) -> modifications``:

    * **self_ref** -- the running result, updated after each overlay.
    * **super_ref** -- the original pre-overlay configuration (immutable).

    Overlay modifications are deep-merged into the running result so
    that nested keys added by earlier overlays or modules are preserved.
    """
    result = copy.deepcopy(config)
    for overlay_fn in overlays:
        modifications = overlay_fn(result, config)
        result = {**result, **modifications}
    return result
'''.lstrip())

# ── SPEC.md ─────────────────────────────────────────────────────────────────
w("/app/SPEC.md", r'''# Deployment Configuration -- Specification

The evaluator (`python3 eval.py`) must produce a JSON object whose
structure and values satisfy every property listed below.

## Merge semantics

Modules are merged left-to-right; later modules override earlier ones.

| Both values are | Action                          |
|-----------------|---------------------------------|
| dicts           | merge recursively               |
| lists           | concatenate (`base ++ override`)|
| otherwise       | override replaces base          |

Overlays modify the merged result.  Overlay modifications must be
**deep-merged** into the running result (i.e. existing nested keys
that the overlay does not mention must survive).

---

## networking

| Path                                      | Expected value                                          |
|-------------------------------------------|---------------------------------------------------------|
| `networking.hostName`                     | `"deploy-node-1"` (set by overlay)                      |
| `networking.domain`                       | `"example.com"`                                         |
| `networking.fqdn`                         | `"deploy-node-1.example.com"`                           |
| `networking.interfaces.eth0.ipv4`         | `"10.0.0.2"`                                            |
| `networking.interfaces.eth0.prefixLength` | `24`                                                    |
| `networking.interfaces.eth0.gateway`      | `"10.0.0.1"`                                            |
| `networking.firewall.enable`              | `true`                                                  |
| `networking.firewall.allowedTCPPorts`     | `[22, 80, 443, 5432, 8080]` (sorted, merged from networking + services) |
| `networking.firewall.defaultPolicy`       | `"DROP"`                                                |
| `networking.firewall.logRefused`          | `true`                                                  |
| `networking.nameservers`                  | `["1.1.1.1", "8.8.8.8"]`                               |

## services

All service definitions from modules must survive overlay application.
Security hardening settings belong under `services.ssh`, **not**
`services.sshd`.

| Path                                      | Expected value                |
|-------------------------------------------|-------------------------------|
| `services.ssh.enable`                     | `true`                        |
| `services.ssh.port`                       | `2222`                        |
| `services.ssh.permitRootLogin`            | `false`                       |
| `services.ssh.authorizedKeys`             | non-empty list                |
| `services.ssh.maxAuthTries`               | `3`                           |
| `services.ssh.passwordAuthentication`     | `false`                       |
| `services.postgresql.enable`              | `true`                        |
| `services.postgresql.port`               | `5432`                        |
| `services.postgresql.dataDir`            | `"/var/lib/postgresql/16"`    |
| `services.nginx.enable`                  | `true`                        |
| `services.nginx.virtualHosts`            | must contain both `app.example.com` (proxy -> `:8080`) and `api.example.com` (proxy -> `:3000`) |

## users

| Path                         | Expected value                    |
|------------------------------|-----------------------------------|
| `users.deploy.uid`           | `1000`                            |
| `users.deploy.shell`         | `"/bin/bash"`                     |
| `users.deploy.groups`        | includes `"wheel"` and `"docker"` |
| `users.postgres.uid`         | `999`                             |
| `users.postgres.isSystemUser`| `true`                            |
| `users.postgres.groups`      | includes `"postgres"`             |
'''.lstrip())

# ── modules/__init__.py ────────────────────────────────────────────────────
w("/app/modules/__init__.py", "")

# ── modules/base.py ────────────────────────────────────────────────────────
w("/app/modules/base.py", r'''"""Base configuration defaults.

Every key defined here acts as a fallback; modules override these
values through recursive merging.
"""

config = {
    "networking": {
        "hostName": "localhost",
        "domain": "localdomain",
        "fqdn": "localhost.localdomain",
        "firewall": {
            "enable": False,
            "allowedTCPPorts": [],
            "defaultPolicy": "ACCEPT",
            "logRefused": False,
        },
    },
    "services": {},
    "users": {},
}
'''.lstrip())

# ── modules/networking.py ──────────────────────────────────────────────────
w("/app/modules/networking.py", r'''"""Networking module -- interfaces, firewall rules, and DNS."""

config = {
    "networking": {
        "domain": "example.com",
        "interfaces": {
            "eth0": {
                "ipv4": "10.0.0.2",
                "prefixLength": 24,
                "gateway": "10.0.0.1",
            },
        },
        "firewall": {
            "enable": True,
            "allowedTCPPorts": [80, 443, 8080],
            "defaultPolicy": "DROP",
        },
        "nameservers": ["1.1.1.1", "8.8.8.8"],
    },
}
'''.lstrip())

# ── modules/services.py ────────────────────────────────────────────────────
w("/app/modules/services.py", r'''"""Service definitions -- SSH, PostgreSQL, Nginx."""

config = {
    "services": {
        "ssh": {
            "enable": True,
            "port": 2222,
        },
        "postgresql": {
            "enable": True,
            "port": 5432,
            "dataDir": "/var/lib/postgresql/16",
        },
        "nginx": {
            "enable": True,
            "virtualHosts": {
                "app.example.com": {
                    "proxyPass": "http://127.0.0.1:8080",
                },
            },
        },
    },
    "networking": {
        "firewall": {
            "allowedTCPPorts": [22, 5432],
        },
    },
}
'''.lstrip())

# ── modules/security.py (Bug 3: "sshd" instead of "ssh") ──────────────────
w("/app/modules/security.py", r'''"""Security hardening -- SSH lockdown and firewall logging."""

config = {
    "services": {
        "sshd": {
            "permitRootLogin": False,
            "authorizedKeys": ["ssh-ed25519 AAAA... deploy@admin"],
            "maxAuthTries": 3,
            "passwordAuthentication": False,
        },
    },
    "networking": {
        "firewall": {
            "logRefused": True,
        },
    },
}
'''.lstrip())

# ── modules/users.py ───────────────────────────────────────────────────────
w("/app/modules/users.py", r'''"""User account definitions."""

config = {
    "users": {
        "deploy": {
            "uid": 1000,
            "shell": "/bin/bash",
            "groups": ["wheel", "docker"],
        },
        "postgres": {
            "uid": 999,
            "isSystemUser": True,
            "groups": ["postgres"],
        },
    },
}
'''.lstrip())

# ── overlays/__init__.py ───────────────────────────────────────────────────
w("/app/overlays/__init__.py", "")

# ── overlays/customization.py (Bug 5: .get("hostName") not .get("domain"))─
w("/app/overlays/customization.py", r'''"""Deployment customization overlay -- sets hostname and FQDN."""


def overlay(self_ref, super_ref):
    """Set the deployment hostname and derive the FQDN.

    The FQDN is composed as ``{hostname}.{domain}`` where *domain*
    comes from the base (pre-overlay) networking configuration.
    """
    hostname = "deploy-node-1"
    domain = super_ref.get("networking", {}).get("hostName", "unknown")
    return {
        "networking": {
            "hostName": hostname,
            "fqdn": "{}.{}".format(hostname, domain),
        },
    }
'''.lstrip())

# ── overlays/monitoring.py ─────────────────────────────────────────────────
w("/app/overlays/monitoring.py", r'''"""Monitoring overlay -- adds the API reverse-proxy virtual host."""


def overlay(self_ref, super_ref):
    """Add an Nginx virtual host for the API monitoring endpoint."""
    return {
        "services": {
            "nginx": {
                "virtualHosts": {
                    "api.example.com": {
                        "proxyPass": "http://127.0.0.1:3000",
                    },
                },
            },
        },
    }
'''.lstrip())

# ── Reference .nix files ───────────────────────────────────────────────────
w("/app/eval.nix", r'''# Main evaluation entrypoint
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
'''.lstrip())

w("/app/lib.nix", r'''# Configuration library functions
# Implements functional composition patterns for config management
rec {
  # Fixed-point combinator
  fix = f: let x = f x; in x;

  # Compose two overlays (extensions)
  # An overlay has the signature: self: super: { ... }
  # where self is the final result (fixpoint) and super is the previous layer
  composeExtensions = f: g: self: super:
    let
      fApplied = f self super;
      gApplied = g (super // fApplied) self;
    in
    fApplied // gApplied;

  # Apply a list of overlays to a base attribute set
  applyOverlays = base: overlays:
    let
      composed = builtins.foldl' composeExtensions (_: _: {}) overlays;
    in
    fix (self: base // composed self base);

  # Recursively merge two attribute sets
  # For nested attrsets: merge recursively
  # For lists: concatenate
  # For other values: b takes precedence
  recursiveUpdate = a: b:
    a // builtins.mapAttrs (name: bVal:
      if builtins.hasAttr name a && builtins.isAttrs a.${name} && builtins.isAttrs bVal
      then recursiveUpdate a.${name} bVal
      else bVal
    ) b;

  # Merge a list of attribute sets using recursiveUpdate
  foldAttrs = list:
    builtins.foldl' recursiveUpdate {} list;

  # Concatenate nested lists
  concatLists = builtins.concatLists;

  # Remove duplicates from a list, preserving first occurrence order
  unique = list:
    builtins.foldl' (acc: x: if builtins.elem x acc then acc else acc ++ [x]) [] list;

  # Sort a list of comparable values
  sort = builtins.sort builtins.lessThan;

  # Filter attributes by predicate
  filterAttrs = pred: attrs:
    builtins.listToAttrs (
      builtins.filter (x: pred x.name x.value)
        (map (name: { inherit name; value = attrs.${name}; })
          (builtins.attrNames attrs))
    );

  # Conditionally include attributes
  optionalAttrs = cond: attrs: if cond then attrs else {};
}
'''.lstrip())

w("/app/modules/network.nix", r'''{ lib, ... }:
{
  networking = {
    interfaces = {
      eth0 = {
        ipv4 = "10.0.0.2";
        prefixLength = 24;
        gateway = "10.0.0.1";
      };
    };
    firewall = {
      enable = true;
      allowedTCPPorts = [ 22 80 443 ];
    };
    nameservers = [ "1.1.1.1" "8.8.8.8" ];
  };
}
'''.lstrip())

w("/app/modules/security.nix", r'''{ lib, ... }:
{
  networking = {
    firewall = {
      allowedTCPPorts = [ 5432 8080 ];
    };
  };
  services = {
    sshd = {
      authorizedKeys = [ "ssh-ed25519 AAAA...deploy-key" ];
    };
  };
}
'''.lstrip())

w("/app/modules/services.nix", r'''{ lib, ... }:
{
  services = {
    ssh = {
      enable = true;
      port = 2222;
      permitRootLogin = false;
    };
    postgresql = {
      enable = true;
      port = 5432;
      dataDir = "/var/lib/postgresql/16";
    };
    nginx = {
      enable = true;
      virtualHosts = {
        "app.example.com" = {
          root = "/var/www/app";
          proxyPass = "http://127.0.0.1:8080";
        };
      };
    };
  };
}
'''.lstrip())

w("/app/modules/users.nix", r'''{ lib, ... }:
{
  users = {
    deploy = {
      uid = 1000;
      groups = [ "wheel" "docker" ];
      shell = "/bin/bash";
    };
    postgres = {
      uid = 999;
      groups = [ "postgres" ];
      shell = "/bin/false";
      isSystemUser = true;
    };
  };
}
'''.lstrip())

w("/app/overlays/customization.nix", r'''# Customization overlay
# Sets deployment-specific values like hostname and additional virtual hosts
self: super:
{
  networking = super.networking // {
    hostName = "deploy-node-1";
    fqdn = "${self.networking.hostName}.${super.networking.hostName}";
  };
  services = super.services // {
    nginx = super.services.nginx // {
      virtualHosts = super.services.nginx.virtualHosts // {
        "api.example.com" = {
          root = "/var/www/api";
          proxyPass = "http://127.0.0.1:3000";
        };
      };
    };
  };
}
'''.lstrip())

w("/app/overlays/hardening.nix", r'''# Security hardening overlay
# Applies security best practices to the configuration
self: super:
{
  services = super.services // {
    ssh = super.services.ssh // {
      maxAuthTries = 3;
      passwordAuthentication = false;
    };
  };
  networking = super.networking // {
    firewall = super.networking.firewall // {
      defaultPolicy = "DROP";
      logRefused = true;
    };
  };
}
'''.lstrip())

print("All app files created successfully at /app/")

# Deployment Configuration -- Specification

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

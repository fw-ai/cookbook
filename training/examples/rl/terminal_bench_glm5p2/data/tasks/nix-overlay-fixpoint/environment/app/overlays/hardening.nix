# Security hardening overlay
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

# Customization overlay
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

{ lib, ... }:
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

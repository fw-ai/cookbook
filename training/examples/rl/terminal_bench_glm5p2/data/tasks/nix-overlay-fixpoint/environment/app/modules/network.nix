{ lib, ... }:
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

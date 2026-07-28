{ lib, ... }:
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

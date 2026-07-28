{ lib, ... }:
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

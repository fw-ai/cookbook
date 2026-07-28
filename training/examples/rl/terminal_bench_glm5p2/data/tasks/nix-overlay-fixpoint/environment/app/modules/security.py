"""Security hardening -- SSH lockdown and firewall logging."""

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

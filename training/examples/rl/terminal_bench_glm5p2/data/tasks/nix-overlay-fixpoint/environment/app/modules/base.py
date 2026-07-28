"""Base configuration defaults.

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

"""Networking module -- interfaces, firewall rules, and DNS."""

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

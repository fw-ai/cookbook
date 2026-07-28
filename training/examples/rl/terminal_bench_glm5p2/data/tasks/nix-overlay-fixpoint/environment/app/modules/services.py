"""Service definitions -- SSH, PostgreSQL, Nginx."""

config = {
    "services": {
        "ssh": {
            "enable": True,
            "port": 2222,
        },
        "postgresql": {
            "enable": True,
            "port": 5432,
            "dataDir": "/var/lib/postgresql/16",
        },
        "nginx": {
            "enable": True,
            "virtualHosts": {
                "app.example.com": {
                    "proxyPass": "http://127.0.0.1:8080",
                },
            },
        },
    },
    "networking": {
        "firewall": {
            "allowedTCPPorts": [22, 5432],
        },
    },
}

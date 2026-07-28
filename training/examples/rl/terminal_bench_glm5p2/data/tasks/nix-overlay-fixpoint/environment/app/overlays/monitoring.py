"""Monitoring overlay -- adds the API reverse-proxy virtual host."""


def overlay(self_ref, super_ref):
    """Add an Nginx virtual host for the API monitoring endpoint."""
    return {
        "services": {
            "nginx": {
                "virtualHosts": {
                    "api.example.com": {
                        "proxyPass": "http://127.0.0.1:3000",
                    },
                },
            },
        },
    }

"""Deployment customization overlay -- sets hostname and FQDN."""


def overlay(self_ref, super_ref):
    """Set the deployment hostname and derive the FQDN.

    The FQDN is composed as ``{hostname}.{domain}`` where *domain*
    comes from the base (pre-overlay) networking configuration.
    """
    hostname = "deploy-node-1"
    domain = super_ref.get("networking", {}).get("hostName", "unknown")
    return {
        "networking": {
            "hostName": hostname,
            "fqdn": "{}.{}".format(hostname, domain),
        },
    }

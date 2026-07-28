"""User account definitions."""

config = {
    "users": {
        "deploy": {
            "uid": 1000,
            "shell": "/bin/bash",
            "groups": ["wheel", "docker"],
        },
        "postgres": {
            "uid": 999,
            "isSystemUser": True,
            "groups": ["postgres"],
        },
    },
}

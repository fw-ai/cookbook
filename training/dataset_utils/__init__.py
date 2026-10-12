"""Lightweight dataset checks shared by preflight validation and rendering.

Keep this package independent of the renderer registry and training runtime so
validators can inspect datasets without importing model or SDK dependencies.
"""

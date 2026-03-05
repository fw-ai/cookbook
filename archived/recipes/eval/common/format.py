# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

import json
from datetime import datetime
from typing import Any, Optional

import jinja2
from bs4 import BeautifulSoup, NavigableString, Tag
from jinja2.sandbox import ImmutableSandboxedEnvironment


def compile_jinja_template(template: str) -> jinja2.Template:
    """
    Compiles a Jinja template.

    Args:
        template: The template string to compile.

    Returns:
        The compiled Jinja template.
    """

    jinja_env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )

    # Shared with inference.
    def raise_exception(message: str) -> None:
        raise jinja2.TemplateError(message)

    jinja_env.globals["raise_exception"] = raise_exception
    jinja_env.globals["datetime"] = datetime

    # Training-only. Do not use in published model templates.
    def safe_eval(expression: str) -> Any:
        allowed_names = {"__builtins__": None}  # Disable all built-in functions
        return eval(expression, allowed_names)

    def fix_encoding(text: str) -> str:
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def tojson(
        x: Any,
        ensure_ascii: bool = False,
        indent: Optional[int] = None,
        separators: Optional[tuple[str, str]] = None,
        sort_keys: bool = False,
    ) -> str:
        return json.dumps(
            x,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    jinja_env.filters["prettify_html"] = prettify_html
    jinja_env.filters["fix_encoding"] = fix_encoding
    jinja_env.filters["tojson"] = tojson
    jinja_env.filters["safe_eval"] = safe_eval
    return jinja_env.from_string(template)


def prettify_html(html_string: str) -> str:
    """
    Prettifies and formats the given HTML string with consistent indentation.

    We don't use BeautifulSoup's default prettification since it overuses
    newlines. We want to keep elements like table cell content in the same line
    with its surrounding tags.

    Args:
        html_string: A string containing valid HTML content.

    Returns:
        A string with formatted and indented HTML content.
    """
    soup = BeautifulSoup(html_string, "html.parser")

    def format_element(element, level=0):
        if isinstance(element, NavigableString):
            return element.strip() if element.strip() else ""

        if not isinstance(element, Tag):
            return ""

        indent = "  " * level
        attrs = "".join(f' {k}="{v}"' for k, v in element.attrs.items())

        if element.name in ("td", "th"):
            content = "".join(str(c).strip() for c in element.contents)
            return f"{indent}<{element.name}{attrs}>{content}</{element.name}>"

        result = [f"{indent}<{element.name}{attrs}>"]
        result.extend(
            format_element(child, level + 1)
            for child in element.children
            if format_element(child, level + 1)
        )
        result.append(f"{indent}</{element.name}>")

        return "\n".join(result)

    # If the soup has an html tag, ensure it's included
    if soup.html:
        return format_element(soup.html)

    return "\n".join(
        format_element(child) for child in soup.children if isinstance(child, Tag)
    )

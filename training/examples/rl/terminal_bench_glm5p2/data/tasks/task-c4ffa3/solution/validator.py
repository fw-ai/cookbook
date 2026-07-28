#!/usr/bin/env python3
"""
NIST ESV Protocol Payload Validation Engine — Reference Implementation

Implements the Validation Script Framework (VSF) for ESV protocol payloads.
Loads JSON validation tree and rule script specifications and evaluates
payloads against them.
"""
import json
import os
import re
from typing import Any, Optional


# ===================================================================
# Expression Tokenizer
# ===================================================================

class Token:
    __slots__ = ("type", "value")

    def __init__(self, type_: str, value: Any):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type!r}, {self.value!r})"


def tokenize(expr: str) -> list:
    tokens: list[Token] = []
    i = 0
    n = len(expr)
    while i < n:
        c = expr[i]

        # whitespace
        if c.isspace():
            i += 1
            continue

        # two-char operators
        if i + 1 < n:
            two = expr[i : i + 2]
            if two in ("==", "!=", ">=", "<=", "&&", "||"):
                tokens.append(Token("OP", two))
                i += 2
                continue

        # single-char tokens
        if c == "(":
            tokens.append(Token("LPAREN", c))
            i += 1
            continue
        if c == ")":
            tokens.append(Token("RPAREN", c))
            i += 1
            continue
        if c == ".":
            tokens.append(Token("DOT", c))
            i += 1
            continue
        if c == ",":
            tokens.append(Token("COMMA", c))
            i += 1
            continue
        if c == "!":
            tokens.append(Token("OP", "!"))
            i += 1
            continue
        if c in (">", "<"):
            tokens.append(Token("OP", c))
            i += 1
            continue

        # string literal
        if c == '"':
            j = i + 1
            parts: list[str] = []
            while j < n and expr[j] != '"':
                if expr[j] == "\\":
                    j += 1
                    if j < n:
                        parts.append(expr[j])
                else:
                    parts.append(expr[j])
                j += 1
            tokens.append(Token("STRING", "".join(parts)))
            i = j + 1
            continue

        # number
        if c.isdigit() or (
            c == "-" and i + 1 < n and (expr[i + 1].isdigit() or expr[i + 1] == ".")
        ):
            j = i
            if c == "-":
                j += 1
            while j < n and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            raw = expr[i:j]
            tokens.append(Token("NUMBER", float(raw) if "." in raw else int(raw)))
            i = j
            continue

        # identifier / keyword
        if c.isalpha() or c == "_":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            word = expr[i:j]
            if word == "null":
                tokens.append(Token("NULL", None))
            elif word == "true":
                tokens.append(Token("BOOL", True))
            elif word == "false":
                tokens.append(Token("BOOL", False))
            else:
                tokens.append(Token("IDENT", word))
            i = j
            continue

        raise ValueError(f"Unexpected character {c!r} at position {i} in: {expr}")

    return tokens


# ===================================================================
# Expression Parser — recursive descent producing tuple-AST
#
# AST node forms:
#   ('literal', value)
#   ('var', name)
#   ('prop', obj_node, prop_name)
#   ('call', obj_node, method_name, [arg_nodes])
#   ('func', func_name, [arg_nodes])
#   ('binary', op, left, right)
#   ('unary', op, operand)
# ===================================================================

class Parser:
    def __init__(self, tokens: list):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected_type: Optional[str] = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if expected_type and tok.type != expected_type:
            raise ValueError(
                f"Expected {expected_type}, got {tok.type} ({tok.value!r})"
            )
        self.pos += 1
        return tok

    def parse(self):
        node = self._or_expr()
        if self.pos < len(self.tokens):
            raise ValueError(f"Trailing tokens at position {self.pos}")
        return node

    # or_expr → and_expr ('||' and_expr)*
    def _or_expr(self):
        left = self._and_expr()
        while self.peek() and self.peek().type == "OP" and self.peek().value == "||":
            self.consume()
            right = self._and_expr()
            left = ("binary", "||", left, right)
        return left

    # and_expr → not_expr ('&&' not_expr)*
    def _and_expr(self):
        left = self._not_expr()
        while self.peek() and self.peek().type == "OP" and self.peek().value == "&&":
            self.consume()
            right = self._not_expr()
            left = ("binary", "&&", left, right)
        return left

    # not_expr → '!' not_expr | comparison
    def _not_expr(self):
        if self.peek() and self.peek().type == "OP" and self.peek().value == "!":
            self.consume()
            operand = self._not_expr()
            return ("unary", "!", operand)
        return self._comparison()

    # comparison → primary (comp_op primary)?
    def _comparison(self):
        left = self._primary()
        if (
            self.peek()
            and self.peek().type == "OP"
            and self.peek().value in ("==", "!=", ">=", "<=", ">", "<")
        ):
            op = self.consume().value
            right = self._primary()
            return ("binary", op, left, right)
        return left

    # primary → '(' or_expr ')' | literal | chain
    def _primary(self):
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression in primary")

        if tok.type == "LPAREN":
            self.consume()
            expr = self._or_expr()
            self.consume("RPAREN")
            return self._chain_tail(expr)

        if tok.type == "NULL":
            self.consume()
            return ("literal", None)
        if tok.type == "BOOL":
            self.consume()
            return ("literal", tok.value)
        if tok.type == "NUMBER":
            self.consume()
            return ("literal", tok.value)
        if tok.type == "STRING":
            self.consume()
            return ("literal", tok.value)

        if tok.type == "IDENT":
            return self._chain()

        raise ValueError(f"Unexpected token in primary: {tok}")

    # chain → IDENT (call_args)? chain_tail
    def _chain(self):
        name = self.consume("IDENT").value
        # bare function call: name(args)
        if self.peek() and self.peek().type == "LPAREN":
            self.consume()
            args = self._parse_args()
            self.consume("RPAREN")
            node = ("func", name, args)
        else:
            node = ("var", name)
        return self._chain_tail(node)

    # chain_tail → ('.' IDENT (call_args)?)*
    def _chain_tail(self, node):
        while self.peek() and self.peek().type == "DOT":
            self.consume()
            member = self.consume("IDENT").value
            if self.peek() and self.peek().type == "LPAREN":
                self.consume()
                args = self._parse_args()
                self.consume("RPAREN")
                node = ("call", node, member, args)
            else:
                node = ("prop", node, member)
            node = self._chain_tail(node)  # allow further chaining
        return node

    def _parse_args(self) -> list:
        args: list = []
        if self.peek() and self.peek().type != "RPAREN":
            args.append(self._or_expr())
            while self.peek() and self.peek().type == "COMMA":
                self.consume()
                args.append(self._or_expr())
        return args


def parse_expr(text: str):
    tokens = tokenize(text)
    return Parser(tokens).parse()


# ===================================================================
# Expression Evaluator
# ===================================================================

class EvalError(Exception):
    pass


class Evaluator:
    def __init__(self, current_property, parent_property, state: dict):
        self.current = current_property
        self.parent = parent_property
        self.state = state

    def evaluate(self, node):
        kind = node[0]

        if kind == "literal":
            return node[1]

        if kind == "var":
            return self._var(node[1])

        if kind == "prop":
            obj = self.evaluate(node[1])
            return self._prop(obj, node[2])

        if kind == "call":
            obj = self.evaluate(node[1])
            method = node[2]
            args = [self.evaluate(a) for a in node[3]]
            return self._method(obj, method, args)

        if kind == "func":
            name = node[1]
            args = [self.evaluate(a) for a in node[2]]
            return self._func(name, args)

        if kind == "binary":
            op = node[1]
            if op == "&&":
                return bool(self.evaluate(node[2])) and bool(self.evaluate(node[3]))
            if op == "||":
                return bool(self.evaluate(node[2])) or bool(self.evaluate(node[3]))
            left = self.evaluate(node[2])
            right = self.evaluate(node[3])
            return self._binop(op, left, right)

        if kind == "unary":
            if node[1] == "!":
                return not self.evaluate(node[2])
            raise EvalError(f"Unknown unary op: {node[1]}")

        raise EvalError(f"Unknown AST node: {kind}")

    # ---- helpers ----

    def _var(self, name: str):
        if name == "currentProperty":
            return self.current
        if name == "parentProperty":
            return self.parent
        if name in self.state:
            return self.state[name]
        # class-name stub (string, Regex, Int32, String)
        return name

    def _prop(self, obj, prop: str):
        if isinstance(obj, dict):
            return obj.get(prop)
        if isinstance(obj, str):
            if prop == "Length":
                return len(obj)
        if isinstance(obj, (list, tuple)):
            if prop == "Length":
                return len(obj)
        if obj is None:
            return None
        raise EvalError(f"Cannot access .{prop} on {type(obj).__name__}")

    def _method(self, obj, method: str, args: list):
        # static-style calls: string.IsNullOrWhiteSpace(x)
        if isinstance(obj, str) and obj in ("string", "String", "Regex", "Int32"):
            return self._static(obj, method, args)

        # instance methods
        if isinstance(obj, str):
            if method == "Strip":
                return obj.strip()
            if method == "StartsWith":
                return obj.startswith(args[0])
            if method == "ToUpper":
                return obj.upper()
            if method == "Substring":
                return obj[int(args[0]) :]

        if isinstance(obj, (list, tuple)):
            if method == "Count":
                return len(obj)
            if method == "Distinct":
                seen: list = []
                for item in obj:
                    if item not in seen:
                        seen.append(item)
                return seen
            if method == "PluckField":
                field = args[0]
                return [
                    (it.get(field) if isinstance(it, dict) else None) for it in obj
                ]
            if method == "Select":
                field = args[0]
                return [
                    (it.get(field) if isinstance(it, dict) else None) for it in obj
                ]

        if isinstance(obj, (int, float)):
            if method == "ToString":
                fmt = args[0] if args else ""
                return str(obj)

        raise EvalError(f"Cannot call .{method}() on {type(obj).__name__}: {obj!r}")

    def _static(self, cls: str, method: str, args: list):
        if cls in ("string", "String"):
            if method == "IsNullOrWhiteSpace":
                val = args[0]
                return val is None or (isinstance(val, str) and val.strip() == "")
            if method == "Format":
                fmt = args[0]
                result = fmt
                for i, arg in enumerate(args[1:]):
                    result = result.replace(f"{{{i}}}", str(arg))
                return result

        if cls == "Regex":
            if method == "IsMatch":
                return bool(re.match(args[1], args[0]))

        if cls == "Int32":
            if method == "Parse":
                return int(args[0])

        raise EvalError(f"Unknown static call: {cls}.{method}")

    def _func(self, name: str, args: list):
        if name == "ExtractField":
            lst, field = args[0], args[1]
            if not isinstance(lst, list):
                raise EvalError("ExtractField expects a list")
            return [(it.get(field) if isinstance(it, dict) else None) for it in lst]
        raise EvalError(f"Unknown function: {name}")

    def _binop(self, op: str, left, right):
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        # numeric comparisons — coerce for mixed int/float
        try:
            l, r = self._numeric(left), self._numeric(right)
        except (TypeError, ValueError):
            raise EvalError(
                f"Cannot compare {type(left).__name__} {op} {type(right).__name__}"
            )
        if op == ">=":
            return l >= r
        if op == "<=":
            return l <= r
        if op == ">":
            return l > r
        if op == "<":
            return l < r
        raise EvalError(f"Unknown op: {op}")

    @staticmethod
    def _numeric(v):
        if isinstance(v, (int, float)):
            return v
        raise TypeError(f"Not numeric: {v!r}")


def eval_expr(text: str, current, parent, state: dict):
    ast = parse_expr(text)
    return Evaluator(current, parent, state).evaluate(ast)


# ===================================================================
# Rule-Script Interpreter
# ===================================================================

class ScriptInterpreter:
    def __init__(self, rules_dir: str, errors: list):
        self.rules_dir = rules_dir
        self.errors = errors
        self.state: dict[str, Any] = {}
        self._cache: dict[str, dict] = {}

    def _load(self, script_file: str) -> dict:
        if script_file not in self._cache:
            path = os.path.join(self.rules_dir, script_file)
            with open(path) as f:
                self._cache[script_file] = json.load(f)
        return self._cache[script_file]

    # ---- public entry ----

    def run_script(self, script_file: str, current, parent, path: str) -> bool:
        rule = self._load(script_file)
        return self._exec_lines(rule.get("vsfScript", []), current, parent, path)

    # ---- core line executor ----

    def _exec_lines(
        self,
        lines: list,
        current,
        parent,
        path: str,
        *,
        test_mode: bool = False,
    ) -> bool:
        for line in lines:
            lt = line.get("lineType", "")

            if lt == "Rule":
                text = line.get("parameters", {}).get("ruleText", "")
                try:
                    result = eval_expr(text, current, parent, self.state)
                except Exception as exc:
                    if not test_mode:
                        self.errors.append(
                            {"path": path, "message": f"Eval error ({text}): {exc}"}
                        )
                    return False
                if not result:
                    if not test_mode:
                        self.errors.append(
                            {"path": path, "message": f"Rule failed: {text}"}
                        )
                    return False

            elif lt == "ImportScript":
                sf = line.get("parameters", {}).get("scriptFile", "")
                if test_mode:
                    ok = self._exec_imported(sf, current, parent, path, test_mode=True)
                else:
                    ok = self.run_script(sf, current, parent, path)
                if not ok:
                    return False

            elif lt == "State":
                key = line.get("parameters", {}).get("key", "")
                val_expr = line.get("parameters", {}).get("value", "")
                try:
                    self.state[key] = eval_expr(val_expr, current, parent, self.state)
                except Exception:
                    pass

            elif lt == "Assert":
                text = line.get("parameters", {}).get("ruleText", "")
                try:
                    result = eval_expr(text, current, parent, self.state)
                except Exception:
                    return True  # skip remaining, not an error
                if not result:
                    return True  # skip remaining, not an error

            elif lt == "Branch":
                ok = self._exec_branch(line, current, parent, path)
                if not ok:
                    return False

            elif lt == "Information":
                pass  # no-op

        return True

    def _exec_imported(self, script_file, current, parent, path, *, test_mode):
        rule = self._load(script_file)
        return self._exec_lines(
            rule.get("vsfScript", []), current, parent, path, test_mode=test_mode
        )

    def _exec_branch(self, line, current, parent, path) -> bool:
        for cond in line.get("conditions", []):
            if "if" in cond:
                if_lines = cond["if"].get("scriptLines", [])
                met = self._exec_lines(
                    if_lines, current, parent, path, test_mode=True
                )
                if met:
                    then_lines = cond.get("then", {}).get("scriptLines", [])
                    return self._exec_lines(then_lines, current, parent, path)
            elif "elseif" in cond:
                ei_lines = cond["elseif"].get("scriptLines", [])
                met = self._exec_lines(
                    ei_lines, current, parent, path, test_mode=True
                )
                if met:
                    then_lines = cond.get("then", {}).get("scriptLines", [])
                    return self._exec_lines(then_lines, current, parent, path)
            elif "else" in cond:
                else_lines = cond["else"].get("scriptLines", [])
                return self._exec_lines(else_lines, current, parent, path)
        return True  # no condition matched → no error


# ===================================================================
# Validation-Tree Traverser
# ===================================================================

class TreeTraverser:
    def __init__(self, interp: ScriptInterpreter):
        self.interp = interp

    def traverse(self, payload: dict, tree: list):
        root = tree[0]["rootNode"]
        # root-level scripts
        self._run_node_scripts(root, payload, payload, "")
        # child nodes
        self._visit_nodes(root.get("nodes", []), payload, "")

    # ---- helpers ----

    def _run_node_scripts(self, node, current, parent, path):
        for entry in node.get("nodeData", {}).get("vsfScriptFiles", []):
            sf = entry.get("scriptFile", "")
            boe = entry.get("breakOnError", True)
            ok = self.interp.run_script(sf, current, parent, path)
            if not ok and boe:
                break

    def _visit_nodes(self, nodes: list, parent_obj, path_prefix: str):
        for node in nodes:
            nt = node.get("nodeType", "")
            pid = node.get("property", {}).get("internalIdentifier", "")
            npath = f"{path_prefix}.{pid}" if path_prefix else pid

            if nt == "leaf":
                val = parent_obj.get(pid) if isinstance(parent_obj, dict) else None
                self._run_node_scripts(node, val, parent_obj, npath)

            elif nt == "parent":
                val = parent_obj.get(pid) if isinstance(parent_obj, dict) else None
                self._run_node_scripts(node, val, parent_obj, npath)
                if val is not None and isinstance(val, dict):
                    self._visit_nodes(node.get("nodes", []), val, npath)

            elif nt == "list":
                lst = parent_obj.get(pid) if isinstance(parent_obj, dict) else None
                # list-level scripts (e.g. listMinCount, eaIdIsDistinct)
                self._run_node_scripts(node, lst, parent_obj, npath)
                # per-item traversal
                if lst is not None and isinstance(lst, list):
                    li = node.get("listItem", {})
                    for i, item in enumerate(lst):
                        ipath = f"{npath}[{i}]"
                        # runBeforeListItem
                        before = (
                            li.get("branchNodeData", {})
                            .get("runBeforeListItem", {})
                            .get("vsfScriptFiles", [])
                        )
                        for entry in before:
                            self.interp.run_script(
                                entry.get("scriptFile", ""), item, parent_obj, ipath
                            )
                        # child nodes
                        self._visit_nodes(li.get("nodes", []), item, ipath)
                        # runAfterListItem
                        after = (
                            li.get("branchNodeData", {})
                            .get("runAfterListItem", {})
                            .get("vsfScriptFiles", [])
                        )
                        for entry in after:
                            self.interp.run_script(
                                entry.get("scriptFile", ""), item, parent_obj, ipath
                            )


# ===================================================================
# Public API
# ===================================================================

_TREE_MAP = {
    "registerEntropySource": "registerEntropySource.json",
    "certifyFull": "certifyFull.json",
}


def validate(
    payload: dict, request_type: str, spec_dir: str = "/app/spec"
) -> dict:
    if request_type not in _TREE_MAP:
        return {
            "valid": False,
            "errors": [{"path": "", "message": f"Unknown request type: {request_type}"}],
        }

    tree_path = os.path.join(spec_dir, "trees", _TREE_MAP[request_type])
    rules_dir = os.path.join(spec_dir, "rules")

    with open(tree_path) as f:
        tree = json.load(f)

    errors: list[dict] = []
    interp = ScriptInterpreter(rules_dir, errors)
    trav = TreeTraverser(interp)
    trav.traverse(payload, tree)

    return {"valid": len(errors) == 0, "errors": errors}

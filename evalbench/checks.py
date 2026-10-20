"""Rule-based checks for scoring model outputs.

Each check is a plain dict ("check spec") with a ``type`` key plus
type-specific parameters. ``run_check(spec, output)`` returns a
``CheckResult`` describing pass/fail, the weight, and human-readable detail.

Supported check types:
    contains, not_contains, contains_any, contains_all,
    regex, not_regex, starts_with, ends_with,
    word_count, char_count, json_object, key_exists,
    required_headings, max_exclamations, placeholder_free
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single check against a model output."""

    check_type: str
    passed: bool
    weight: float
    description: str
    details: str = ""


class CheckError(ValueError):
    """Raised when a check spec is malformed."""


_PLACEHOLDER_TOKENS = ("TODO", "TBD", "FIXME", "[INSERT", "XXX", "<insert", "PLACEHOLDER")


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _has(needle: str, haystack: str, case_sensitive: bool) -> bool:
    return needle in haystack if case_sensitive else needle.lower() in haystack.lower()


def _describe(spec: dict[str, Any]) -> str:
    """Human-readable description of what a check verifies."""
    ctype = spec.get("type", "?")
    if ctype in ("contains", "not_contains"):
        verb = "Contains" if ctype == "contains" else "Does not contain"
        return f"{verb} {spec.get('value', '')!r}"
    if ctype in ("contains_any", "contains_all"):
        return f"{ctype.replace('_', ' ')} {spec.get('values')}"
    if ctype in ("regex", "not_regex"):
        verb = "Matches" if ctype == "regex" else "Does not match"
        return f"{verb} regex /{spec.get('pattern', '')}/"
    if ctype in ("starts_with", "ends_with"):
        return f"{ctype.replace('_', ' ').capitalize()} {spec.get('value', '')!r}"
    if ctype in ("word_count", "char_count"):
        return f"{ctype.replace('_', ' ').capitalize()} within [{spec.get('min')}, {spec.get('max')}]"
    if ctype == "json_object":
        return "Output is a valid JSON object"
    if ctype == "key_exists":
        return f"JSON output contains key {spec.get('key', '')!r}"
    if ctype == "required_headings":
        return f"Contains markdown headings {spec.get('headings')}"
    if ctype == "max_exclamations":
        return f"At most {spec.get('max', 0)} exclamation marks"
    if ctype == "placeholder_free":
        return "Contains no placeholder tokens (TODO/TBD/[INSERT...])"
    return f"Check type {ctype!r}"


_CHECK_IMPLS: dict[str, Callable[[dict[str, Any], str], tuple[bool, str]]] = {}


def _register(name: str):
    def deco(fn):
        _CHECK_IMPLS[name] = fn
        return fn

    return deco


def _get_values(spec: dict[str, Any], ctype: str) -> list[str]:
    values = [str(v) for v in spec.get("values", [])]
    if not values:
        raise CheckError(f"{ctype} requires a non-empty 'values' list")
    return values


def _in_range(n: int, spec: dict[str, Any], label: str) -> tuple[bool, str]:
    lo, hi = spec.get("min"), spec.get("max")
    ok = (lo is None or n >= lo) and (hi is None or n <= hi)
    return ok, f"{label} = {n} (allowed {lo}..{hi})"


def _try_parse_json(output: str) -> tuple[Any, str | None]:
    try:
        return json.loads(output), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"


def _regex_flags(spec: dict[str, Any]) -> int:
    return 0 if spec.get("case_sensitive", False) else re.IGNORECASE


@_register("contains")
def _contains(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    value = str(spec["value"])
    ok = _has(value, output, bool(spec.get("case_sensitive", False)))
    return ok, f"{'found' if ok else 'not found'}: {value!r}"


@_register("not_contains")
def _not_contains(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    value = str(spec["value"])
    found = _has(value, output, bool(spec.get("case_sensitive", False)))
    return not found, f"{'found (unexpected)' if found else 'absent (good)'}: {value!r}"


@_register("contains_any")
def _contains_any(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    values = _get_values(spec, "contains_any")
    cs = bool(spec.get("case_sensitive", False))
    hits = [v for v in values if _has(v, output, cs)]
    return bool(hits), f"matched: {hits}"


@_register("contains_all")
def _contains_all(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    values = _get_values(spec, "contains_all")
    cs = bool(spec.get("case_sensitive", False))
    missing = [v for v in values if not _has(v, output, cs)]
    return not missing, f"missing: {missing}"


@_register("regex")
def _regex(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    match = re.search(spec["pattern"], output, _regex_flags(spec))
    return bool(match), f"matched: {match.group(0)!r}" if match else "no match"


@_register("not_regex")
def _not_regex(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    match = re.search(spec["pattern"], output, _regex_flags(spec))
    return not match, f"matched (unexpected): {match.group(0)!r}" if match else "no match"


@_register("starts_with")
def _starts_with(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    value = str(spec["value"])
    cs = bool(spec.get("case_sensitive", False))
    ok = output.startswith(value) if cs else output.lower().startswith(value.lower())
    return ok, f"prefix {'ok' if ok else 'mismatch'}: {value!r}"


@_register("ends_with")
def _ends_with(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    value = str(spec["value"])
    cs = bool(spec.get("case_sensitive", False))
    tail = output.rstrip()
    ok = tail.endswith(value) if cs else tail.lower().endswith(value.lower())
    return ok, f"suffix {'ok' if ok else 'mismatch'}: {value!r}"


@_register("word_count")
def _word_count(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    return _in_range(len(_words(output)), spec, "word count")


@_register("char_count")
def _char_count(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    return _in_range(len(output), spec, "char count")


@_register("json_object")
def _json_object(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    parsed, error = _try_parse_json(output)
    if error:
        return False, error
    if not isinstance(parsed, dict):
        return False, f"top-level JSON value is {type(parsed).__name__}, expected object"
    return True, "valid JSON object"


@_register("key_exists")
def _key_exists(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    parsed, error = _try_parse_json(output)
    if error:
        return False, error
    key = spec["key"]
    ok = isinstance(parsed, dict) and key in parsed
    return ok, f"key {key!r} {'present' if ok else 'missing'}"


@_register("required_headings")
def _required_headings(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    wanted = [h.lower() for h in spec.get("headings", [])]
    found = {ln.lstrip("#").strip().lower() for ln in output.splitlines() if ln.startswith("#")}
    missing = [h for h in wanted if h not in found]
    return not missing, f"missing headings: {missing}"


@_register("max_exclamations")
def _max_exclamations(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    limit = int(spec.get("max", 0))
    n = output.count("!")
    return n <= limit, f"{n} exclamation mark(s) (max {limit})"


@_register("placeholder_free")
def _placeholder_free(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    found = [tok for tok in _PLACEHOLDER_TOKENS if tok in output]
    return not found, f"placeholder tokens found: {found}" if found else "no placeholders"


def run_check(spec: dict[str, Any], output: str) -> CheckResult:
    """Evaluate one check spec against a model output.

    Raises:
        CheckError: if the spec is malformed (unknown type, missing fields).
    """
    ctype = spec.get("type")
    if not ctype:
        raise CheckError("Check spec is missing a 'type' key")
    impl = _CHECK_IMPLS.get(ctype)
    if impl is None:
        raise CheckError(
            f"Unknown check type {ctype!r}. Supported: {', '.join(sorted(_CHECK_IMPLS))}"
        )
    weight = float(spec.get("weight", 1.0))
    try:
        passed, details = impl(spec, output)
    except KeyError as exc:
        raise CheckError(f"Check type {ctype!r} is missing required field {exc}") from exc
    return CheckResult(
        check_type=ctype,
        passed=passed,
        weight=weight,
        description=_describe(spec),
        details=details,
    )

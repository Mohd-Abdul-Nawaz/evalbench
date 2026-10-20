"""Tests for evalbench.checks."""

import json

import pytest

from evalbench.checks import CheckError, run_check


def test_contains_pass():
    result = run_check({"type": "contains", "value": "Paris"}, "The capital is Paris.")
    assert result.passed
    assert result.check_type == "contains"


def test_contains_case_insensitive_by_default():
    result = run_check({"type": "contains", "value": "PARIS"}, "paris is nice")
    assert result.passed


def test_contains_fail():
    result = run_check({"type": "contains", "value": "Berlin"}, "The capital is Paris.")
    assert not result.passed


def test_not_contains():
    result = run_check({"type": "not_contains", "value": "TODO"}, "All done here.")
    assert result.passed


def test_contains_any_and_all():
    assert run_check({"type": "contains_any", "values": ["a", "z"]}, "zap").passed
    assert not run_check({"type": "contains_any", "values": ["x", "y"]}, "zap").passed
    assert run_check({"type": "contains_all", "values": ["z", "a"]}, "zap").passed
    assert not run_check({"type": "contains_all", "values": ["z", "q"]}, "zap").passed


def test_regex_and_not_regex():
    assert run_check({"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}"}, "on 2026-10-20").passed
    assert not run_check({"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}"}, "no date").passed
    assert run_check({"type": "not_regex", "pattern": "http"}, "no links here").passed


def test_word_count_bounds():
    text = "one two three"
    assert run_check({"type": "word_count", "min": 3, "max": 3}, text).passed
    assert not run_check({"type": "word_count", "min": 4}, text).passed
    assert not run_check({"type": "word_count", "max": 2}, text).passed


def test_char_count():
    assert run_check({"type": "char_count", "min": 5, "max": 5}, "hello").passed
    assert not run_check({"type": "char_count", "max": 4}, "hello").passed


def test_json_object_and_key_exists():
    payload = json.dumps({"status": "ok"})
    assert run_check({"type": "json_object"}, payload).passed
    assert not run_check({"type": "json_object"}, "not json").passed
    assert not run_check({"type": "json_object"}, "[1, 2]").passed
    assert run_check({"type": "key_exists", "key": "status"}, payload).passed
    assert not run_check({"type": "key_exists", "key": "missing"}, payload).passed


def test_required_headings():
    md = "## Overview\nstuff\n## Next steps\nmore"
    assert run_check({"type": "required_headings", "headings": ["Overview", "next STEPS"]}, md).passed
    assert not run_check({"type": "required_headings", "headings": ["Risks"]}, md).passed


def test_max_exclamations_and_placeholders():
    assert run_check({"type": "max_exclamations", "max": 1}, "Hi! Bye").passed
    assert not run_check({"type": "max_exclamations", "max": 0}, "Hi! Bye!").passed
    assert run_check({"type": "placeholder_free"}, "All concrete text.").passed
    assert not run_check({"type": "placeholder_free"}, "See TODO item").passed


def test_weight_is_carried_through():
    result = run_check({"type": "contains", "value": "x", "weight": 2.5}, "x")
    assert result.weight == 2.5


def test_unknown_type_raises():
    with pytest.raises(CheckError):
        run_check({"type": "no_such_check"}, "output")


def test_missing_type_raises():
    with pytest.raises(CheckError):
        run_check({"value": "x"}, "output")


def test_missing_required_field_raises():
    with pytest.raises(CheckError):
        run_check({"type": "contains"}, "output")
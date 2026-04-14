"""Tests for lib.similarity."""

from lib.similarity import similar, _sequence_similarity, _jaccard_similarity


def test_similar_identical():
    assert similar("hello world", "hello world")


def test_similar_minor_difference():
    assert similar("hello world", "hello worlds")


def test_similar_unrelated():
    assert not similar("hello world", "xyzzy foobar quux")


def test_similar_empty_strings():
    assert similar("", "")


def test_similar_one_empty():
    assert not similar("hello", "")


def test_sequence_circuit_breaker():
    long = "a" * 201
    assert _sequence_similarity(long, long) == 0.0


def test_jaccard_circuit_breaker():
    long = "a " * 101
    assert _jaccard_similarity(long, long) == 0.0


def test_sequence_similarity_identical():
    assert _sequence_similarity("test", "test") == 1.0


def test_jaccard_similarity_identical():
    assert _jaccard_similarity("hello world", "hello world") == 1.0


def test_jaccard_similarity_disjoint():
    assert _jaccard_similarity("aaa bbb", "ccc ddd") == 0.0

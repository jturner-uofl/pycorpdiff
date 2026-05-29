"""Tests for ``read_txt(one_doc_per='line')``."""

from __future__ import annotations

from pathlib import Path

import pytest

import pycorpdiff as pcd


def test_read_txt_line_mode_one_doc_per_line(tmp_path: Path) -> None:
    path = tmp_path / "lines.txt"
    path.write_text("first line\nsecond line\nthird line\n", encoding="utf-8")
    corpus = pcd.read_txt(path, one_doc_per="line")
    assert len(corpus) == 3
    assert list(corpus.docs["text"]) == ["first line", "second line", "third line"]


def test_read_txt_line_mode_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "lines.txt"
    path.write_text("alpha\n\n\nbeta\n   \ngamma\n", encoding="utf-8")
    corpus = pcd.read_txt(path, one_doc_per="line")
    assert list(corpus.docs["text"]) == ["alpha", "beta", "gamma"]


def test_read_txt_line_mode_records_line_numbers(tmp_path: Path) -> None:
    # Line numbers are 1-based and reflect the *original* file's lines —
    # so a blank line that's skipped still bumps the line counter.
    path = tmp_path / "lines.txt"
    path.write_text("alpha\n\nbeta\n", encoding="utf-8")
    corpus = pcd.read_txt(path, one_doc_per="line")
    assert corpus.docs["line"].tolist() == [1, 3]


def test_read_txt_line_mode_records_source(tmp_path: Path) -> None:
    path = tmp_path / "lines.txt"
    path.write_text("hello world\n", encoding="utf-8")
    corpus = pcd.read_txt(path, one_doc_per="line")
    assert corpus.docs["source"].iloc[0] == str(path)


def test_read_txt_file_mode_still_works(tmp_path: Path) -> None:
    path = tmp_path / "whole.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    corpus = pcd.read_txt(path, one_doc_per="file")
    assert len(corpus) == 1
    assert corpus.docs["text"].iloc[0] == "alpha\nbeta\ngamma\n"


def test_read_txt_rejects_unknown_one_doc_per(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="one_doc_per must be"):
        pcd.read_txt(path, one_doc_per="paragraph")


def test_read_txt_line_mode_empty_file_yields_empty_corpus(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    corpus = pcd.read_txt(path, one_doc_per="line")
    assert len(corpus) == 0

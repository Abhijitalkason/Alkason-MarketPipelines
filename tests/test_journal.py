"""Journal: hash-chain verification, tamper detection, write-once semantics."""

from __future__ import annotations

import json

import pytest

from src.intraday.journal import (JournalError, _path, journal_verify, journal_write,
                                  read_journal, seal_day)


def test_chain_verifies(repo, frozen_day):
    journal_write("screen", {"top": ["AAA"]}, day=frozen_day)
    journal_write("gate_eval", {"symbol": "AAA", "fired": True}, day=frozen_day)
    seal_day(frozen_day)
    assert journal_verify(frozen_day)


def test_tamper_detected(repo, frozen_day):
    journal_write("screen", {"top": ["AAA"]}, day=frozen_day)
    journal_write("fill", {"symbol": "AAA"}, day=frozen_day)
    p = _path(frozen_day)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["payload"]["top"] = ["TAMPERED"]
    lines[0] = json.dumps(rec, default=str, sort_keys=True)
    p.write_text("\n".join(lines) + "\n")
    assert not journal_verify(frozen_day)


def test_unknown_event_raises(repo, frozen_day):
    with pytest.raises(JournalError):
        journal_write("not_a_real_event", {}, day=frozen_day)


def test_read_filter(repo, frozen_day):
    journal_write("screen", {"top": []}, day=frozen_day)
    journal_write("fill", {"symbol": "AAA"}, day=frozen_day)
    assert len(read_journal(frozen_day, "fill")) == 1
    assert len(read_journal(frozen_day)) == 2

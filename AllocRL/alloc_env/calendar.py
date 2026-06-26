"""
근무일(Working Day) 관리 모듈.

C# Calendar 클래스의 1:1 Python 재구현.
- 주말(토·일) 판별
- 공휴일 등록/관리
- 두 날짜 사이 실 근무일 수 계산
- 출고일 역산 (입고일 + 근무일수 → 출고일)
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Iterable

# ── 휴일 관리 ─────────────────────────────────────────────────────────

_holidays: set[date] = set()


def set_holidays(holidays: Iterable[date]) -> None:
    """공휴일 목록을 교체합니다."""
    _holidays.clear()
    _holidays.update(holidays)


def add_holiday(d: date) -> None:
    """공휴일을 하나 추가합니다."""
    _holidays.add(d)


def get_holidays() -> frozenset[date]:
    """현재 등록된 공휴일 목록을 반환합니다."""
    return frozenset(_holidays)


# ── 근무일 판별 ───────────────────────────────────────────────────────

def is_working_day(d: date) -> bool:
    """주말·공휴일이 아닌 평일이면 True."""
    if d.weekday() >= 5:  # 5=토, 6=일
        return False
    return d not in _holidays


# ── 기능 1: 두 날짜 사이의 실 근무일 수 ───────────────────────────────

def get_working_days_between(start: date, end: date) -> int:
    """
    start~end(양 끝 포함) 사이의 실 근무일 수를 반환합니다.
    start > end이면 0.
    """
    if start > end:
        return 0
    count = 0
    d = start
    while d <= end:
        if is_working_day(d):
            count += 1
        d += timedelta(days=1)
    return count


# ── 기능 2: 날짜를 가장 가까운 근무일로 조정 ──────────────────────────

def adjust_to_working_day(d: date, forward: bool = True) -> date:
    """비근무일이면 가장 가까운 근무일로 이동합니다."""
    step = timedelta(days=1) if forward else timedelta(days=-1)
    while not is_working_day(d):
        d += step
    return d


# ── 기능 3: 입고일 + 실 근무일 수 → 출고일 ───────────────────────────

def calculate_end_date(start: date, working_days: int) -> date:
    """
    start를 포함하여 working_days 만큼의 실 근무일이 지난 날짜를 반환합니다.
    예) start=월요일, working_days=3 → 수요일.
    """
    if working_days <= 0:
        return adjust_to_working_day(start)

    current = adjust_to_working_day(start)
    counted = 1  # start(조정된 값)를 첫 번째 근무일로 카운트
    while counted < working_days:
        current += timedelta(days=1)
        if is_working_day(current):
            counted += 1
    return current


def next_working_day(d: date) -> date:
    """다음 근무일을 반환합니다."""
    return adjust_to_working_day(d + timedelta(days=1), forward=True)

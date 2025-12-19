from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class SlotExtraction:
    slots: dict[str, str]
    rationale: list[str]


_AMOUNT_RE = re.compile(
    r"\b(?:(?P<currency>CHF|USD|EUR|GBP|\$|€|£)\s*)?"
    r"(?P<amount>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))\b"
)

_DATE_ISO_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
_DATE_SLASH_RE = re.compile(r"\b([0-3]?\d)[/.-]([01]?\d)[/.-](20\d{2})\b")
_DATE_MONTHNAME_1_RE = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
    r"(?P<day>[0-3]?\d)(?:,?\s+(?P<year>20\d{2}))?\b",
    flags=re.IGNORECASE,
)
_DATE_MONTHNAME_2_RE = re.compile(
    r"\b(?P<day>[0-3]?\d)\s+"
    r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"(?:,?\s+(?P<year>20\d{2}))?\b",
    flags=re.IGNORECASE,
)

_REF_RE = re.compile(
    r"\b(?:order|reference|ref|ticket|case)\s*(?:id|number|#)?\s*[:#]?\s*([A-Z0-9][A-Z0-9-]{4,})\b",
    flags=re.IGNORECASE,
)

_LAST4_RE = re.compile(
    r"\b(?:ending\s+in|last\s+(?:four|4)\s+(?:digits\s+)?)"
    r"(?:are\s+|is\s+)?(\d{4})\b",
    flags=re.IGNORECASE,
)

_MERCHANT_RE = re.compile(
    r"\b(?:merchant\s+|at\s+|from\s+)"
    r"(?P<merchant>[A-Za-z0-9][A-Za-z0-9&' -]{2,40}?)"
    r"(?:\s+\b(?:on|for|in)\b|[.,;!?]|$)",
    flags=re.IGNORECASE,
)


def _normalize_amount(amount: str) -> str:
    amount = amount.strip()
    if "," in amount and "." in amount:
        amount = amount.replace(",", "")
    amount = amount.replace(",", ".")
    return amount


def _normalize_date_ymd(year: int, month: int, day: int) -> Optional[str]:
    try:
        dt = datetime(year=year, month=month, day=day)
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d")


def _extract_amount(text: str) -> tuple[Optional[str], list[str]]:
    m = _AMOUNT_RE.search(text)
    if not m:
        return None, []
    amount = _normalize_amount(m.group("amount"))
    currency = (m.group("currency") or "").strip()
    if currency:
        return amount, [f"slot: amount={amount}", f"slot: currency={currency}"]
    return amount, [f"slot: amount={amount}"]


def _extract_date(text: str) -> tuple[Optional[str], list[str]]:
    m = _DATE_ISO_RE.search(text)
    if m:
        return m.group(0), [f"slot: date={m.group(0)}"]

    m = _DATE_SLASH_RE.search(text)
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        year = int(m.group(3))
        norm = _normalize_date_ymd(year, month, day)
        if norm:
            return norm, [f"slot: date={norm}"]

    for rx in (_DATE_MONTHNAME_1_RE, _DATE_MONTHNAME_2_RE):
        m = rx.search(text)
        if not m:
            continue
        month = _MONTHS.get((m.group("month") or "").lower())
        day = int(m.group("day"))
        year = int(m.group("year") or datetime.now().year)
        if not month:
            continue
        norm = _normalize_date_ymd(year, month, day)
        if norm:
            return norm, [f"slot: date={norm}"]

    return None, []


def _extract_reference_id(text: str) -> tuple[Optional[str], list[str]]:
    m = _REF_RE.search(text)
    if not m:
        return None, []
    ref_id = m.group(1).strip()
    return ref_id, [f"slot: reference_id={ref_id}"]


def _extract_card_last4(text: str) -> tuple[Optional[str], list[str]]:
    m = _LAST4_RE.search(text)
    if not m:
        return None, []
    last4 = m.group(1)
    return last4, [f"slot: card_last4={last4}"]


def _extract_merchant(text: str) -> tuple[Optional[str], list[str]]:
    m = _MERCHANT_RE.search(text)
    if not m:
        return None, []
    merchant = (m.group("merchant") or "").strip(" ,.")
    if len(merchant) < 3:
        return None, []
    return merchant, [f"slot: merchant={merchant}"]


REQUIRED_SLOTS_BY_INTENT: dict[str, tuple[str, ...]] = {
    "chargeback": ("amount", "date"),
    "refund_not_showing_up": ("amount", "date"),
    "lost_or_stolen_card": ("card_last4",),
}


def missing_required_slots(intent: str, slots: dict[str, str]) -> list[str]:
    required = REQUIRED_SLOTS_BY_INTENT.get((intent or "").strip().lower(), ())
    return [name for name in required if not (slots or {}).get(name)]


def suggest_followups(intent: str, slots: dict[str, str]) -> list[str]:
    missing = missing_required_slots(intent, slots)
    if not missing:
        return []
    questions = []
    if "amount" in missing:
        questions.append("Ask for the transaction amount.")
    if "date" in missing:
        questions.append("Ask for the transaction date.")
    if "card_last4" in missing:
        questions.append("Ask for the last 4 digits of the card (if known).")
    return questions


def extract_slots(text: str, *, intent: Optional[str] = None) -> SlotExtraction:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return SlotExtraction(slots={}, rationale=[])

    slots: dict[str, str] = {}
    rationale: list[str] = []

    amount, why = _extract_amount(text)
    if amount:
        slots["amount"] = amount
        rationale.extend(why)

    date, why = _extract_date(text)
    if date:
        slots["date"] = date
        rationale.extend(why)

    ref_id, why = _extract_reference_id(text)
    if ref_id:
        slots["reference_id"] = ref_id
        rationale.extend(why)

    last4, why = _extract_card_last4(text)
    if last4:
        slots["card_last4"] = last4
        rationale.extend(why)

    merchant, why = _extract_merchant(text)
    if merchant:
        slots["merchant"] = merchant
        rationale.extend(why)

    missing = missing_required_slots(intent or "", slots)
    for m in missing:
        rationale.append(f"missing_slot: {m}")

    return SlotExtraction(slots=slots, rationale=rationale)

import re
import statistics
import datetime
from datetime import date, datetime
from functools import lru_cache
from typing import List, Dict

from recur_scan.transactions import Transaction
from collections import defaultdict

def detect_sequence_patterns(
    transaction: Transaction, 
    all_transactions: List[Transaction],
    min_occurrences: int = 3
) -> Dict[str, float]:
    """
    Detects recurring sequences with confidence scores.
    """
    # Skip transactions with zero amount to avoid division by zero
    if transaction.amount == 0:
        return {"sequence_confidence": 0.0, "sequence_pattern": "none", "sequence_length": 0}

    vendor_txs = [
        t for t in all_transactions 
        if t.name.lower() == transaction.name.lower()
        and abs(t.amount - transaction.amount) / transaction.amount < 0.05
    ]
    
    if len(vendor_txs) < min_occurrences:
        return {"sequence_confidence": 0.0, "sequence_pattern": "none", "sequence_length": 0}
    
    vendor_txs_sorted = sorted(vendor_txs, key=lambda x: _parse_date(x.date))

    intervals = [
        (_parse_date(vendor_txs_sorted[i].date) - _parse_date(vendor_txs_sorted[i-1].date)).days
        for i in range(1, len(vendor_txs_sorted))
    ]
    avg_interval = statistics.mean(intervals)
    stdev_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
    
    patterns = {"weekly": 7, "monthly": 30, "yearly": 365}
    best_pattern, best_confidence = "none", 0.0
    
    for name, expected_interval in patterns.items():
        deviation = abs(avg_interval - expected_interval)
        tolerance = max(2, expected_interval * 0.1)
        if deviation <= tolerance:
            confidence = 1 - (stdev_interval / (expected_interval + 1e-6))
            if confidence > best_confidence:
                best_pattern, best_confidence = name, max(0, min(1, confidence))
    
    return {
        "sequence_confidence": best_confidence,
        "sequence_pattern": best_pattern,
        "sequence_length": len(vendor_txs)
    }


def get_is_always_recurring(transaction: Transaction) -> bool:
    """Check if the transaction is always recurring because of the vendor name - check lowercase match"""
    always_recurring_vendors = {
        "google storage",
        "netflix",
        "hulu",
        "spotify",
    }
    return transaction.name.lower() in always_recurring_vendors


def get_is_insurance(transaction: Transaction) -> bool:
    """Check if the transaction is an insurance payment."""
    # use a regular expression with boundaries to match case-insensitive insurance
    # and insurance-related terms
    match = re.search(r"\b(insurance|insur|insuranc)\b", transaction.name, re.IGNORECASE)
    return bool(match)


def get_is_utility(transaction: Transaction) -> bool:
    """Check if the transaction is a utility payment."""
    # use a regular expression with boundaries to match case-insensitive utility
    # and utility-related terms
    match = re.search(r"\b(utility|utilit|energy)\b", transaction.name, re.IGNORECASE)
    return bool(match)


def get_is_phone(transaction: Transaction) -> bool:
    """Check if the transaction is a phone payment."""
    # use a regular expression with boundaries to match case-insensitive phone
    # and phone-related terms
    match = re.search(r"\b(at&t|t-mobile|verizon)\b", transaction.name, re.IGNORECASE)
    return bool(match)


@lru_cache(maxsize=1024)
def _parse_date(date_str: str) -> date:
    """Parse a date string into a datetime.date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def get_n_transactions_days_apart(
    transaction: Transaction,
    all_transactions: list[Transaction],
    n_days_apart: int,
    n_days_off: int,
) -> int:
    """
    Get the number of transactions in all_transactions that are within n_days_off of
    being n_days_apart from transaction
    """
    n_txs = 0
    transaction_date = _parse_date(transaction.date)

    # Pre-calculate bounds for faster checking
    lower_remainder = n_days_apart - n_days_off
    upper_remainder = n_days_off

    for t in all_transactions:
        t_date = _parse_date(t.date)
        days_diff = abs((t_date - transaction_date).days)

        # Skip if the difference is less than minimum required
        if days_diff < n_days_apart - n_days_off:
            continue

        # Check if the difference is close to any multiple of n_days_apart
        remainder = days_diff % n_days_apart

        if remainder <= upper_remainder or remainder >= lower_remainder:
            n_txs += 1

    return n_txs


def get_pct_transactions_days_apart(
    transaction: Transaction, all_transactions: list[Transaction], n_days_apart: int, n_days_off: int
) -> float:
    """
    Get the percentage of transactions in all_transactions that are within
    n_days_off of being n_days_apart from transaction
    """
    return get_n_transactions_days_apart(transaction, all_transactions, n_days_apart, n_days_off) / len(
        all_transactions
    )


def _get_day(date: str) -> int:
    """Get the day of the month from a transaction date."""
    return int(date.split("-")[2])


def get_n_transactions_same_day(transaction: Transaction, all_transactions: list[Transaction], n_days_off: int) -> int:
    """Get the number of transactions in all_transactions that are on the same day of the month as transaction"""
    return len([t for t in all_transactions if abs(_get_day(t.date) - _get_day(transaction.date)) <= n_days_off])


def get_pct_transactions_same_day(
    transaction: Transaction, all_transactions: list[Transaction], n_days_off: int
) -> float:
    """Get the percentage of transactions in all_transactions that are on the same day of the month as transaction"""
    return get_n_transactions_same_day(transaction, all_transactions, n_days_off) / len(all_transactions)


def get_ends_in_99(transaction: Transaction) -> bool:
    """Check if the transaction amount ends in 99"""
    return (transaction.amount * 100) % 100 == 99

def get_is_recurring(transaction: Transaction, all_transactions: list[Transaction]) -> bool:
    """
    Check if the transaction is part of a recurring pattern based on the time intervals
    between transactions with the same amount and vendor name.
    """
    transaction_date = _parse_date(transaction.date)
    similar_transactions = [
        t for t in all_transactions
        if t.amount == transaction.amount and t.name == transaction.name and t.date != transaction.date
    ]

    if not similar_transactions:
        return False

    # Calculate the time intervals between the transaction and similar transactions
    intervals = sorted(
        abs((transaction_date - _parse_date(t.date)).days) for t in similar_transactions
    )

    # Check if the intervals form a recurring pattern (e.g., weekly, bi-weekly, monthly)
    for interval in intervals:
        if interval % 7 == 0 or interval % 14 == 0 or interval % 30 == 0:
            return True

    return False
def get_recurring_transaction_confidence(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """
    Calculate a recurring transaction confidence score by combining:
    - Amount stability
    - Interval regularity
    - Transaction frequency
    - Metadata similarity
    """
    # 1. Amount Stability
    similar_transactions = [
        t.amount for t in all_transactions
        if t.name == transaction.name and t.date != transaction.date
    ]
    if len(similar_transactions) < 2:
        amount_stability = 1.0  # High variability if fewer than 2 transactions
    else:
        mean = sum(similar_transactions) / len(similar_transactions)
        stdev = statistics.stdev(similar_transactions)
        amount_stability = stdev / mean if mean != 0 else 1.0

    # 2. Interval Regularity
    similar_dates = [
        _parse_date(t.date) for t in all_transactions
        if t.name == transaction.name and t.date != transaction.date
    ]
    if len(similar_dates) < 2:
        interval_regularities = float('inf')  # No intervals if fewer than 2 transactions
    else:
        intervals = [(similar_dates[i] - similar_dates[i - 1]).days for i in range(1, len(similar_dates))]
        if len(intervals) < 2:
            interval_regularities = float('inf')  # Default value for insufficient data
        else:
            interval_regularities = statistics.stdev(intervals)

    # 3. Transaction Frequency
    transaction_date = _parse_date(transaction.date)
    transaction_frequency = len([
        t for t in all_transactions
        if t.name == transaction.name and abs((_parse_date(t.date) - transaction_date).days) <= 30
    ])

    # 4. Metadata Similarity
    def jaccard_similarity(set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0.0

    metadata_similarities = [
        jaccard_similarity(set(transaction.name.split()), set(t.name.split()))
        for t in all_transactions
        if t.name == transaction.name and t.date != transaction.date
    ]
    metadata_similarity = sum(metadata_similarities) / len(metadata_similarities) if metadata_similarities else 0.0

    # 5. Combine into a Confidence Score
    score = (
        (1 / (1 + amount_stability)) * 0.3 +  # Weight: 30%
        (1 / (1 + interval_regularities)) * 0.3 +  # Weight: 30%
        (transaction_frequency / max(transaction_frequency, 1)) * 0.2 +  # Weight: 20%
        metadata_similarity * 0.2  # Weight: 20%
    )
    return score

def get_n_transactions_same_amount(transaction: Transaction, all_transactions: list[Transaction]) -> int:
    """Get the number of transactions in all_transactions with the same amount as transaction"""
    return len([t for t in all_transactions if t.amount == transaction.amount])


def get_percent_transactions_same_amount(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """Get the percentage of transactions in all_transactions with the same amount as transaction"""
    if not all_transactions:
        return 0.0
    n_same_amount = len([t for t in all_transactions if t.amount == transaction.amount])
    return n_same_amount / len(all_transactions)


def get_features(transaction: Transaction, all_transactions: list[Transaction]) -> dict[str, float | int]:
    """
    Extract features for a given transaction.
    """
    # Detect sequence patterns
    sequence_features = detect_sequence_patterns(transaction, all_transactions)

    return {
        # Existing features
        "n_transactions_same_amount": get_n_transactions_same_amount(transaction, all_transactions),
        "percent_transactions_same_amount": get_percent_transactions_same_amount(transaction, all_transactions),
        "ends_in_99": get_ends_in_99(transaction),
        "amount": transaction.amount,
        "same_day_exact": get_n_transactions_same_day(transaction, all_transactions, 0),
        "pct_transactions_same_day": get_pct_transactions_same_day(transaction, all_transactions, 0),
        "same_day_off_by_1": get_n_transactions_same_day(transaction, all_transactions, 1),
        "same_day_off_by_2": get_n_transactions_same_day(transaction, all_transactions, 2),
        "14_days_apart_exact": get_n_transactions_days_apart(transaction, all_transactions, 14, 0),
        "pct_14_days_apart_exact": get_pct_transactions_days_apart(transaction, all_transactions, 14, 0),
        "14_days_apart_off_by_1": get_n_transactions_days_apart(transaction, all_transactions, 14, 1),
        "pct_14_days_apart_off_by_1": get_pct_transactions_days_apart(transaction, all_transactions, 14, 1),
        "7_days_apart_exact": get_n_transactions_days_apart(transaction, all_transactions, 7, 0),
        "pct_7_days_apart_exact": get_pct_transactions_days_apart(transaction, all_transactions, 7, 0),
        "7_days_apart_off_by_1": get_n_transactions_days_apart(transaction, all_transactions, 7, 1),
        "pct_7_days_apart_off_by_1": get_pct_transactions_days_apart(transaction, all_transactions, 7, 1),
        "is_insurance": get_is_insurance(transaction),
        "is_utility": get_is_utility(transaction),
        "is_phone": get_is_phone(transaction),
        "is_always_recurring": get_is_always_recurring(transaction),
        "is_recurring": get_is_recurring(transaction, all_transactions),
        "recurring_transaction_confidence": get_recurring_transaction_confidence(transaction, all_transactions),

        # New features from sequence detection
        "sequence_confidence": sequence_features["sequence_confidence"],
        "is_sequence_weekly": 1.0 if sequence_features["sequence_pattern"] == "weekly" else 0.0,
        "is_sequence_monthly": 1.0 if sequence_features["sequence_pattern"] == "monthly" else 0.0,
        "sequence_length": sequence_features["sequence_length"],
    }
        

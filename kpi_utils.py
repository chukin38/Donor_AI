"""Utility helpers for KPI baseline estimation and adjustment."""

from typing import Dict, Any
import pandas as pd


def baseline_kpis(event: Dict[str, Any], donors: pd.DataFrame) -> Dict[str, float]:
    """Estimate baseline KPIs from historical event data and selected donors.

    Parameters
    ----------
    event : dict
        Event dictionary with optional ``prev_years`` and ``goal_amount`` keys.
    donors : pandas.DataFrame
        DataFrame of selected donors. Must contain a column with average gift
        amounts, e.g. ``avg_gift_hkd`` or ``avg_gift_usd``.
    Returns
    -------
    dict
        Dictionary with keys ``conv_rate``, ``avg_gift``, and ``revenue`` plus
        averages from ``prev_years`` if available.
    """
    prev = event.get("prev_years") or []
    avg_attendees = (
        sum(p.get("attendees", 0) for p in prev) / len(prev) if prev else 0
    )
    avg_total = (
        sum(p.get("total_raised", 0) for p in prev) / len(prev) if prev else 0
    )

    donor_count = len(donors)
    if donor_count == 0:
        return {
            "avg_prev_attendees": avg_attendees,
            "avg_prev_total": avg_total,
            "conv_rate": 0.0,
            "avg_gift": 0.0,
            "revenue": 0.0,
        }

    gift_col = "avg_gift_hkd" if "avg_gift_hkd" in donors.columns else "avg_gift_usd"
    donor_avg_gift = float(donors[gift_col].mean())

    conv_rate = min(avg_attendees / donor_count, 1.0) if avg_attendees else 0.1
    avg_gift = avg_total / avg_attendees if avg_attendees else donor_avg_gift
    revenue = conv_rate * avg_gift * donor_count

    return {
        "avg_prev_attendees": avg_attendees,
        "avg_prev_total": avg_total,
        "conv_rate": conv_rate,
        "avg_gift": avg_gift,
        "revenue": revenue,
    }


def adjust_with_baseline(
    event: Dict[str, Any],
    donors: pd.DataFrame,
    llm_kpis: Dict[str, float],
) -> Dict[str, float]:
    """Adjust LLM-generated KPIs so estimated revenue meets the event goal."""
    baseline = baseline_kpis(event, donors)
    goal = event.get("goal_amount", baseline["revenue"])
    donor_count = len(donors)
    if donor_count == 0:
        return llm_kpis

    revenue = llm_kpis.get("conv_rate", 0) * llm_kpis.get("avg_gift_hkd", 0) * donor_count
    if revenue < goal:
        llm_kpis["conv_rate"] = max(llm_kpis.get("conv_rate", 0), baseline["conv_rate"])
        llm_kpis["avg_gift_hkd"] = max(
            llm_kpis.get("avg_gift_hkd", 0), baseline["avg_gift"]
        )
        revenue = llm_kpis["conv_rate"] * llm_kpis["avg_gift_hkd"] * donor_count
        if revenue < goal:
            llm_kpis["avg_gift_hkd"] = goal / (llm_kpis["conv_rate"] * donor_count)

    llm_kpis["baseline_revenue"] = baseline["revenue"]
    llm_kpis["estimated_revenue"] = llm_kpis["conv_rate"] * llm_kpis["avg_gift_hkd"] * donor_count
    return llm_kpis


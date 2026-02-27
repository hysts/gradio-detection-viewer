"""Tests for advanced label filter features: double-click solo/unsolo."""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect


def test_double_click_solos_label(detection_app: Page):
    """Double-clicking a label filter should solo it (only that label visible)."""
    btns = detection_app.locator(".label-filter-btn")
    first_btn = btns.first

    # Double-click to solo the first label
    first_btn.dblclick()

    # First button should be active, all others inactive
    expect(first_btn).to_have_class(re.compile("active"))
    for i in range(1, btns.count()):
        expect(btns.nth(i)).not_to_have_class(re.compile("active"))


def test_double_click_solo_shows_hidden_group(detection_app: Page):
    """Soloing a label should show the Hidden group separator for other labels."""
    btns = detection_app.locator(".label-filter-btn")
    btns.first.dblclick()

    separator = detection_app.locator(".annotation-group-separator")
    expect(separator).to_be_visible()
    expect(separator).to_contain_text("Hidden")


def test_double_click_again_unsolos(detection_app: Page):
    """Double-clicking the solo label again should show all labels."""
    btns = detection_app.locator(".label-filter-btn")
    first_btn = btns.first

    # Solo
    first_btn.dblclick()
    expect(first_btn).to_have_class(re.compile("active"))

    # Unsolo
    first_btn.dblclick()

    # All should be active again
    for i in range(btns.count()):
        expect(btns.nth(i)).to_have_class(re.compile("active"))


def test_double_click_solo_filters_rows(detection_app: Page):
    """Soloing a label should filter out non-matching annotation rows."""
    btns = detection_app.locator(".label-filter-btn")
    btns.first.dblclick()

    # There should be some filtered-out rows
    filtered = detection_app.locator(".annotation-row.filtered-out")
    expect(filtered.first).to_be_visible()

    # At least one row should not be filtered
    visible = detection_app.locator(".annotation-row:not(.filtered-out)")
    expect(visible.first).to_be_visible()


def test_double_click_different_label_switches_solo(detection_app: Page):
    """Double-clicking a different label should switch the solo to that label."""
    btns = detection_app.locator(".label-filter-btn")

    # Solo first label
    btns.first.dblclick()
    expect(btns.first).to_have_class(re.compile("active"))
    expect(btns.nth(1)).not_to_have_class(re.compile("active"))

    # Solo second label
    btns.nth(1).dblclick()
    expect(btns.nth(1)).to_have_class(re.compile("active"))
    expect(btns.first).not_to_have_class(re.compile("active"))

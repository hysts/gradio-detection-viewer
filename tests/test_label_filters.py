"""Tests for label filter buttons."""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect


def test_label_filter_buttons_present(detection_app: Page):
    """Label filter buttons should exist for each unique label."""
    btns = detection_app.locator(".label-filter-btn")
    expect(btns).to_have_count(4)  # person, dog, chair, bench


def test_all_labels_active_by_default(detection_app: Page):
    """All label filter buttons should be active initially."""
    btns = detection_app.locator(".label-filter-btn")
    for i in range(btns.count()):
        expect(btns.nth(i)).to_have_class(re.compile("active"))


def test_click_label_deactivates(detection_app: Page):
    """Clicking an active label filter should deactivate it."""
    btn = detection_app.locator(".label-filter-btn").first
    btn.click()
    expect(btn).not_to_have_class(re.compile("active"))


def test_deactivated_label_dims_rows(detection_app: Page):
    """Deactivating a label should dim the corresponding annotation rows."""
    # Get the label text from the first button
    btn = detection_app.locator(".label-filter-btn").first
    btn.click()
    detection_app.wait_for_timeout(200)

    filtered = detection_app.locator(".annotation-row.filtered-out")
    assert filtered.count() > 0


def test_hidden_group_separator(detection_app: Page):
    """Deactivating a label should show a 'Hidden' group separator."""
    btn = detection_app.locator(".label-filter-btn").first
    btn.click()
    detection_app.wait_for_timeout(200)

    separator = detection_app.locator(".annotation-group-separator")
    expect(separator).to_be_visible()
    expect(separator).to_contain_text("Hidden")


def test_reactivate_label_removes_filter(detection_app: Page):
    """Clicking a deactivated label should reactivate it and remove filtering."""
    btn = detection_app.locator(".label-filter-btn").first
    btn.click()
    detection_app.wait_for_timeout(200)
    assert detection_app.locator(".annotation-row.filtered-out").count() > 0

    btn.click()
    detection_app.wait_for_timeout(200)
    assert detection_app.locator(".annotation-row.filtered-out").count() == 0

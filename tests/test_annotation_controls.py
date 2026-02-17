"""Tests for annotation visibility, selection, and expand controls."""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect


def test_all_checkboxes_checked_by_default(detection_app: Page):
    """All annotation checkboxes should be checked initially."""
    checkboxes = detection_app.locator(".ann-checkbox")
    for i in range(checkboxes.count()):
        expect(checkboxes.nth(i)).to_be_checked()


def test_select_all_checked_by_default(detection_app: Page):
    """Select-all checkbox should be checked initially."""
    select_all = detection_app.locator(".select-all-checkbox")
    expect(select_all).to_be_checked()


def test_uncheck_annotation(detection_app: Page):
    """Unchecking an annotation checkbox should uncheck it."""
    checkbox = detection_app.locator(".ann-checkbox").first
    checkbox.uncheck()
    expect(checkbox).not_to_be_checked()


def test_uncheck_one_makes_select_all_indeterminate(detection_app: Page):
    """Unchecking one annotation should make select-all indeterminate."""
    detection_app.locator(".ann-checkbox").first.uncheck()
    select_all = detection_app.locator(".select-all-checkbox")
    # Select-all should still be checked (indeterminate state keeps checked)
    expect(select_all).to_be_checked()


def test_uncheck_all_via_select_all(detection_app: Page):
    """Unchecking select-all should uncheck all annotations."""
    detection_app.locator(".select-all-checkbox").uncheck()
    checkboxes = detection_app.locator(".ann-checkbox")
    for i in range(checkboxes.count()):
        expect(checkboxes.nth(i)).not_to_be_checked()


def test_recheck_all_via_select_all(detection_app: Page):
    """After unchecking all, rechecking select-all should check all annotations."""
    select_all = detection_app.locator(".select-all-checkbox")
    select_all.uncheck()
    select_all.check()
    checkboxes = detection_app.locator(".ann-checkbox")
    for i in range(checkboxes.count()):
        expect(checkboxes.nth(i)).to_be_checked()


def test_click_row_selects(detection_app: Page):
    """Clicking an annotation row should add the 'selected' class."""
    row = detection_app.locator(".annotation-row").first
    row.click()
    expect(row).to_have_class(re.compile("selected"))


def test_click_row_again_deselects(detection_app: Page):
    """Clicking the same row again should deselect it."""
    row = detection_app.locator(".annotation-row").first
    row.click()
    expect(row).to_have_class(re.compile("selected"))
    row.click()
    expect(row).not_to_have_class(re.compile("selected"))


def test_expand_button_shows_detail(detection_app: Page):
    """Clicking expand button should show the detail panel."""
    expand_btn = detection_app.locator(".ann-expand").first
    expand_btn.click()
    detail = detection_app.locator(".annotation-detail").first
    expect(detail).to_have_class(re.compile("visible"))


def test_expand_button_toggles(detection_app: Page):
    """Clicking expand button twice should hide the detail panel."""
    expand_btn = detection_app.locator(".ann-expand").first
    expand_btn.click()
    detail = detection_app.locator(".annotation-detail").first
    expect(detail).to_have_class(re.compile("visible"))
    expand_btn.click()
    expect(detail).not_to_have_class(re.compile("visible"))


def test_only_one_detail_expanded(detection_app: Page):
    """Expanding one annotation should collapse any previously expanded one."""
    btns = detection_app.locator(".ann-expand")
    details = detection_app.locator(".annotation-detail")

    btns.nth(0).click()
    expect(details.nth(0)).to_have_class(re.compile("visible"))

    btns.nth(1).click()
    expect(details.nth(0)).not_to_have_class(re.compile("visible"))
    expect(details.nth(1)).to_have_class(re.compile("visible"))

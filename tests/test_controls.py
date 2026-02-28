"""Tests for toolbar buttons, keyboard shortcuts, and draw options."""

from __future__ import annotations

import re

from _helpers import focus_viewer
from playwright.sync_api import Page, expect

# ── Image toggle ──


def test_image_toggle_button(detection_app: Page):
    """Image button should toggle active class."""
    btn = detection_app.locator(".toggle-image-btn")
    expect(btn).to_have_class(re.compile("active"))

    btn.click()
    expect(btn).not_to_have_class(re.compile("active"))

    btn.click()
    expect(btn).to_have_class(re.compile("active"))


# ── Maximize ──


def test_maximize_button(detection_app: Page):
    """Maximize button should add 'maximized' class to the wrapper (element)."""
    wrapper = detection_app.locator(".pose-viewer-container").first.locator("xpath=..")
    btn = detection_app.locator(".maximize-btn")

    btn.click()
    expect(wrapper).to_have_class(re.compile("maximized"))

    btn.click()
    expect(wrapper).not_to_have_class(re.compile("maximized"))


# ── Help dialog ──


def test_help_button(detection_app: Page):
    """Help button should open the help overlay."""
    overlay = detection_app.locator(".help-overlay")
    expect(overlay).not_to_have_class(re.compile("visible"))

    detection_app.locator(".help-btn").click()
    expect(overlay).to_have_class(re.compile("visible"))


def test_help_close_button(detection_app: Page):
    """Close button in help dialog should close it."""
    detection_app.locator(".help-btn").click()
    overlay = detection_app.locator(".help-overlay")
    expect(overlay).to_have_class(re.compile("visible"))

    detection_app.locator(".help-close-btn").click()
    expect(overlay).not_to_have_class(re.compile("visible"))


# ── Keyboard shortcuts ──


def test_key_i_toggles_image(detection_app: Page):
    """Pressing 'i' should toggle image visibility."""
    focus_viewer(detection_app)
    btn = detection_app.locator(".toggle-image-btn")
    expect(btn).to_have_class(re.compile("active"))

    detection_app.keyboard.press("i")
    expect(btn).not_to_have_class(re.compile("active"))

    detection_app.keyboard.press("i")
    expect(btn).to_have_class(re.compile("active"))


def test_key_f_toggles_maximize(detection_app: Page):
    """Pressing 'f' should toggle maximize mode."""
    focus_viewer(detection_app)
    wrapper = detection_app.locator(".pose-viewer-container").first.locator("xpath=..")

    detection_app.keyboard.press("f")
    expect(wrapper).to_have_class(re.compile("maximized"))

    detection_app.keyboard.press("f")
    expect(wrapper).not_to_have_class(re.compile("maximized"))


def test_key_escape_exits_maximize(detection_app: Page):
    """Pressing Escape while maximized should exit maximize mode."""
    focus_viewer(detection_app)
    wrapper = detection_app.locator(".pose-viewer-container").first.locator("xpath=..")

    detection_app.keyboard.press("f")
    expect(wrapper).to_have_class(re.compile("maximized"))

    detection_app.keyboard.press("Escape")
    expect(wrapper).not_to_have_class(re.compile("maximized"))


def test_key_question_mark_opens_help(detection_app: Page):
    """Pressing '?' should open the help dialog."""
    focus_viewer(detection_app)
    overlay = detection_app.locator(".help-overlay")

    detection_app.keyboard.press("?")
    expect(overlay).to_have_class(re.compile("visible"))


def test_key_escape_closes_help(detection_app: Page):
    """Pressing Escape should close the help dialog."""
    focus_viewer(detection_app)
    overlay = detection_app.locator(".help-overlay")

    detection_app.keyboard.press("?")
    expect(overlay).to_have_class(re.compile("visible"))

    detection_app.keyboard.press("Escape")
    expect(overlay).not_to_have_class(re.compile("visible"))


def test_key_a_toggles_all(detection_app: Page):
    """Pressing 'a' should toggle all annotations."""
    focus_viewer(detection_app)

    detection_app.keyboard.press("a")
    checkboxes = detection_app.locator(".ann-checkbox")
    for i in range(checkboxes.count()):
        expect(checkboxes.nth(i)).not_to_be_checked()

    detection_app.keyboard.press("a")
    for i in range(checkboxes.count()):
        expect(checkboxes.nth(i)).to_be_checked()


# ── Reset ──


def test_reset_button_restores_state(detection_app: Page):
    """Reset button should restore initial state after changes."""
    detection_app.locator(".toggle-image-btn").click()
    detection_app.locator(".ann-checkbox").first.uncheck()

    btn = detection_app.locator(".toggle-image-btn")
    expect(btn).not_to_have_class(re.compile("active"))

    detection_app.locator(".reset-btn").click()

    expect(btn).to_have_class(re.compile("active"))
    checkboxes = detection_app.locator(".ann-checkbox")
    for i in range(checkboxes.count()):
        expect(checkboxes.nth(i)).to_be_checked()


def test_key_r_resets(detection_app: Page):
    """Pressing 'r' should reset state."""
    detection_app.locator(".toggle-image-btn").click()
    btn = detection_app.locator(".toggle-image-btn")
    expect(btn).not_to_have_class(re.compile("active"))

    focus_viewer(detection_app)
    detection_app.keyboard.press("r")
    expect(btn).to_have_class(re.compile("active"))


# ── Draw options ──


def test_draw_options_collapsed_by_default(detection_app: Page):
    """Draw options section should be collapsed by default."""
    body = detection_app.locator(".draw-options-body")
    expect(body).to_be_hidden()


def test_draw_options_toggle(detection_app: Page):
    """Clicking draw options toggle should expand/collapse."""
    toggle = detection_app.locator(".draw-options-toggle")
    body = detection_app.locator(".draw-options-body")

    toggle.click()
    expect(body).to_be_visible()

    toggle.click()
    expect(body).to_be_hidden()

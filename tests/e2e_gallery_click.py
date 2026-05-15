"""End-to-end browser test: prove the gallery's carousel actually opens
when a snapshot is clicked, and that arrow keys cycle through images.

Uses Playwright driving the system Chrome (no chromium bundle).
"""

import sys
from playwright.sync_api import sync_playwright

BASE = "http://localhost:5001"


def fail(msg):
    print(f"  ✗ FAIL: {msg}")
    sys.exit(1)


def step(msg):
    print(f"  • {msg}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
        )
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()

        # Capture any console errors so I see real JS failures (the
        # exact bug class that bit us last time was a silent SyntaxError
        # in an inline onclick).
        page.on("console", lambda m: m.type == "error" and print(f"  [console.error] {m.text}"))
        page.on("pageerror", lambda exc: print(f"  [pageerror] {exc}"))

        step(f"Navigating to {BASE}/snapshots")
        page.goto(f"{BASE}/snapshots", wait_until="networkidle")

        # ── Step 1: visitor tiles rendered ──────────────────────────────
        page.wait_for_selector(".visitor-tile", timeout=8000)
        tile_count = page.locator(".visitor-tile").count()
        step(f"Visitor tiles rendered: {tile_count}")
        if tile_count == 0:
            fail("no visitor tiles — gallery is empty")

        # ── Step 2: click the FIRST visitor tile to open the modal ──────
        step("Clicking first visitor tile…")
        page.locator(".visitor-tile").first.click()
        page.wait_for_selector("#visitorModal.open", timeout=4000)
        if "open" not in (page.locator("#visitorModal").get_attribute("class") or ""):
            fail("visitor modal did not open")
        step("✓ visitor modal opened")

        # ── Step 3: snap-cards appear inside the modal ──────────────────
        page.wait_for_selector("#vm-grid .snap-card", timeout=4000)
        snap_count = page.locator("#vm-grid .snap-card").count()
        step(f"Snap cards inside the modal: {snap_count}")
        if snap_count == 0:
            fail("no snap-cards rendered in the visitor modal")

        # ── Step 4: data attributes present (the bug we just fixed) ─────
        first_card = page.locator("#vm-grid .snap-card").first
        key = first_card.get_attribute("data-snap-key")
        idx = first_card.get_attribute("data-snap-idx")
        step(f"First card → data-snap-key={key!r}  data-snap-idx={idx!r}")
        if key is None or idx is None:
            fail("snap-card is missing data-snap-key / data-snap-idx")

        # ── Step 5: click the PHOTO of the first snap-card → carousel ───
        step("Clicking the snap-card's photo (the previously-broken path)…")
        first_card.locator(".snap-thumb").click()
        page.wait_for_selector("#carouselModal.open", timeout=4000)
        carousel_class = page.locator("#carouselModal").get_attribute("class") or ""
        if "open" not in carousel_class:
            fail(f"carousel modal did NOT open — class='{carousel_class}'")
        step("✓ carousel modal opened on photo click")

        carousel_img_src = page.locator("#carousel-img").get_attribute("src") or ""
        carousel_counter = page.locator("#carousel-counter").inner_text()
        step(f"  carousel img src    : {carousel_img_src}")
        step(f"  carousel counter    : {carousel_counter}")
        if not carousel_img_src.startswith("/snapshots/"):
            fail("carousel image src isn't pointing at a snapshot")

        # ── Step 6: arrow-right cycles to the next image ────────────────
        step("Pressing ArrowRight to advance to the next image…")
        first_src = carousel_img_src
        first_counter = carousel_counter
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(400)
        next_src = page.locator("#carousel-img").get_attribute("src") or ""
        next_counter = page.locator("#carousel-counter").inner_text()
        step(f"  new img src         : {next_src}")
        step(f"  new counter         : {next_counter}")

        # If the visitor has only one snapshot, wrap-around brings the
        # same image back. That's still correct behavior — verify by
        # checking the counter stays at "1 / 1".
        if int(carousel_counter.split("/")[1].strip()) == 1:
            if next_src != first_src:
                fail("single-snapshot visitor — ArrowRight should wrap to same image")
            step("  (visitor has only 1 snapshot — wrap to same image, correct)")
        else:
            if next_src == first_src or next_counter == first_counter:
                fail("ArrowRight did NOT advance to a new image")
            step("✓ ArrowRight advanced the image")

        # ── Step 7: arrow-left cycles backward ──────────────────────────
        step("Pressing ArrowLeft to go back…")
        page.keyboard.press("ArrowLeft")
        page.wait_for_timeout(400)
        back_src = page.locator("#carousel-img").get_attribute("src") or ""
        if back_src != first_src:
            fail("ArrowLeft did NOT return to the original image")
        step("✓ ArrowLeft returned to original image")

        # ── Step 8: Escape closes the carousel ──────────────────────────
        step("Pressing Escape to close the carousel…")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        carousel_class_after = page.locator("#carouselModal").get_attribute("class") or ""
        if "open" in carousel_class_after:
            fail("Escape did NOT close the carousel")
        step("✓ carousel closed; visitor modal still open underneath")

        # ── Step 9: NOW try clicking the View BUTTON ────────────────────
        step("Re-opening carousel via the View button on a card…")
        first_card.locator(".snap-view-btn").click()
        page.wait_for_selector("#carouselModal.open", timeout=4000)
        step("✓ View button also opens carousel")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        # Close the visitor modal too so the tile is clickable again.
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)

        # ── Step 10: a visible screenshot for the record ────────────────
        page.locator(".visitor-tile").first.click()
        page.wait_for_selector("#vm-grid .snap-card", timeout=4000)
        page.locator("#vm-grid .snap-card").first.locator(".snap-thumb").click()
        page.wait_for_selector("#carouselModal.open", timeout=4000)
        page.wait_for_timeout(500)
        page.screenshot(path="/tmp/carousel_open.png", full_page=False)
        step("Screenshot of open carousel saved → /tmp/carousel_open.png")

        browser.close()
        print()
        print("ALL E2E CHECKS PASSED ✓")


if __name__ == "__main__":
    main()

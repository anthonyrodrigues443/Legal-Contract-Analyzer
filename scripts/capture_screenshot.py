"""Capture a Streamlit app screenshot with real analysis populated.

Loads an example contract, clicks 'Analyze contract', waits for the result,
and saves a full-page screenshot to results/ui_screenshot.png.
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL = "http://localhost:8510"
OUT = Path("results/ui_screenshot.png")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(viewport={"width": 1440, "height": 2400})
        page = await context.new_page()
        await page.goto(URL)
        await page.wait_for_selector("text=Legal Contract Risk Analyzer", timeout=30_000)

        # Select a longer CUAD example (position 3 = median-length contract, more risky clauses)
        await page.click("[data-baseweb='select']")
        await page.wait_for_timeout(400)
        options = page.locator("[role='option']")
        n = await options.count()
        # index 0 is "— none —", so 5 is the longest CUAD example (most clauses)
        if n >= 6:
            await options.nth(5).click()
        elif n >= 2:
            await options.nth(n - 1).click()
        await page.wait_for_timeout(500)

        # Click "Analyze contract"
        await page.get_by_role("button", name="Analyze contract").click()

        # Wait for the risk score to render
        await page.wait_for_selector("text=Risk:", timeout=45_000)
        await page.wait_for_timeout(2500)  # let plotly finish

        OUT.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(OUT), full_page=True)
        print(f"Saved {OUT}")

        await browser.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"Screenshot capture failed: {exc}", file=sys.stderr)
        sys.exit(1)

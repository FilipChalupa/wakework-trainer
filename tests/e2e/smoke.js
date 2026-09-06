// Playwright smoke test: every tab renders its cards and the contributor page rejects a bad token.
const { chromium } = require("playwright");
const BASE = process.env.BASE_URL || "http://localhost:8000";

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ locale: "en-US" });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(BASE, { waitUntil: "load" });
  await page.waitForSelector(".MuiCard-root", { timeout: 20000 });
  const expected = {
    Data: ["Wake word configuration", "Sample recording", "Negative datasets"],
    Training: ["Model training", "Trained models"],
    Test: ["Test the model"],
    Deploy: ["Deploy to ESPHome"],
  };
  let cards = 0;
  for (const [tab, titles] of Object.entries(expected)) {
    await page.getByRole("tab", { name: tab }).click();
    await page.waitForTimeout(1500);
    const text = await page.locator("body").innerText();
    for (const title of titles) {
      if (!text.includes(title)) throw new Error(`Tab ${tab}: missing card ${title}`);
    }
    cards += await page.locator(".MuiCard-root").count();
  }
  await page.goto(`${BASE}/contribute?token=invalid`, { waitUntil: "load" });
  await page.waitForTimeout(1500);
  const contributeText = await page.locator("body").innerText();
  if (!/not valid/i.test(contributeText)) throw new Error("Contributor page did not reject an invalid token");
  if (errors.length) throw new Error(`Page errors: ${errors.join("; ")}`);
  console.log(`OK: ${cards} cards rendered across 4 tabs, contributor page guarded`);
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});

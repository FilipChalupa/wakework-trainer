// Playwright smoke test: the SPA renders all six cards and the contributor page rejects a bad token.
const { chromium } = require("playwright");
const BASE = process.env.BASE_URL || "http://localhost:8000";

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ locale: "en-US" });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(BASE, { waitUntil: "load" });
  await page.waitForSelector(".MuiCard-root", { timeout: 20000 });
  await page.waitForTimeout(2000);
  const cards = await page.locator(".MuiCard-root").count();
  const text = await page.locator("body").innerText();
  for (const expected of ["Wake word configuration", "Sample recording", "Negative datasets", "Model training", "Trained models", "Test the model"]) {
    if (!text.includes(expected)) throw new Error(`Missing card: ${expected}`);
  }
  if (cards < 6) throw new Error(`Expected >= 6 cards, got ${cards}`);
  await page.goto(`${BASE}/contribute?token=invalid`, { waitUntil: "load" });
  await page.waitForTimeout(1500);
  const contributeText = await page.locator("body").innerText();
  if (!/not valid/i.test(contributeText)) throw new Error("Contributor page did not reject an invalid token");
  if (errors.length) throw new Error(`Page errors: ${errors.join("; ")}`);
  console.log(`OK: ${cards} cards rendered, contributor page guarded`);
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});

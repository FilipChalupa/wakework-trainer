import { describe, expect, it } from "vitest";
import { detectLanguage, format } from "../i18n";

describe("format", () => {
  it("substitutes parameters and keeps unknown placeholders", () => {
    expect(format("Krok {step} / {total}", { step: 3, total: 10 })).toBe("Krok 3 / 10");
    expect(format("{a} and {b}", { a: "x" })).toBe("x and {b}");
    expect(format("plain")).toBe("plain");
  });
});

describe("detectLanguage", () => {
  it("prefers Czech/Slovak browsers, English otherwise", () => {
    Object.defineProperty(globalThis, "navigator", { value: { languages: ["sk-SK", "en-US"], language: "sk-SK" }, configurable: true });
    expect(detectLanguage()).toBe("cs");
    Object.defineProperty(globalThis, "navigator", { value: { languages: ["de-DE"], language: "de-DE" }, configurable: true });
    expect(detectLanguage()).toBe("en");
  });
});

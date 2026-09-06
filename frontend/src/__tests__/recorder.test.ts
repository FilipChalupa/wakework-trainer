// @vitest-environment node
import { describe, expect, it } from "vitest";
import { encodeWav, resample, waveformPeaks } from "../lib/recorder";

describe("encodeWav", () => {
  it("writes a 16 kHz mono 16-bit PCM header and clamps samples", async () => {
    const samples = new Float32Array([0, 0.5, -0.5, 2, -2]);
    const blob = encodeWav(samples, 16000);
    const buf = new DataView(await blob.arrayBuffer());
    expect(String.fromCharCode(buf.getUint8(0), buf.getUint8(1), buf.getUint8(2), buf.getUint8(3))).toBe("RIFF");
    expect(buf.getUint16(22, true)).toBe(1); // channels
    expect(buf.getUint32(24, true)).toBe(16000); // sample rate
    expect(buf.getUint16(34, true)).toBe(16); // bits
    expect(buf.getUint32(40, true)).toBe(samples.length * 2);
    expect(buf.getInt16(44 + 3 * 2, true)).toBe(32767); // clamped +2
    expect(buf.getInt16(44 + 4 * 2, true)).toBe(-32768); // clamped -2
  });
});

describe("resample fallback", () => {
  it("halves the length when converting 32 kHz to 16 kHz (no OfflineAudioContext in node)", async () => {
    const input = new Float32Array(3200).map((_, i) => Math.sin(i / 10));
    const out = await resample(input, 32000, 16000);
    expect(out.length).toBe(1600);
    expect(Math.abs(out[100] - input[200])).toBeLessThan(0.05);
  });
  it("returns the input untouched for equal rates", async () => {
    const input = new Float32Array([1, 2, 3]);
    expect(await resample(input, 16000, 16000)).toBe(input);
  });
});

describe("waveformPeaks", () => {
  it("returns the requested number of buckets with absolute peaks", () => {
    const samples = new Float32Array(800).map((_, i) => (i < 400 ? -0.2 : 0.9));
    const peaks = waveformPeaks(samples, 8);
    expect(peaks).toHaveLength(8);
    expect(peaks[0]).toBeCloseTo(0.2);
    expect(peaks[7]).toBeCloseTo(0.9);
  });
});

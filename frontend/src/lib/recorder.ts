/**
 * Microphone capture with the Web Audio API. Captures raw PCM through an AudioWorklet
 * (ScriptProcessor fallback), resamples to 16 kHz and encodes a mono 16-bit PCM WAV.
 */

export const TARGET_SAMPLE_RATE = 16000;

const WORKLET_SOURCE = `
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);
`;

export type LevelCallback = (info: { rms: number; peak: number; elapsed: number }) => void;

export class Recorder {
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private workletUrl: string | null = null;

  private deviceId: string | undefined;

  async init(deviceId?: string): Promise<void> {
    if (this.stream && deviceId === this.deviceId) return;
    if (this.stream) this.close();
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Prohlížeč nepodporuje přístup k mikrofonu (je stránka otevřená přes HTTPS nebo localhost?).");
    }
    this.deviceId = deviceId;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: true,
      },
    });
    this.context = new AudioContext();
    if (this.context.audioWorklet) {
      this.workletUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "application/javascript" }));
      await this.context.audioWorklet.addModule(this.workletUrl);
    }
  }

  get ready(): boolean {
    return !!this.stream && !!this.context;
  }

  /** Available microphones (labels are only populated after permission was granted). */
  static async listDevices(): Promise<{ deviceId: string; label: string }[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((d) => d.kind === "audioinput")
      .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Mikrofon ${i + 1}` }));
  }

  async record(durationS: number, onLevel?: LevelCallback): Promise<{ wav: Blob; samples: Float32Array; sampleRate: number }> {
    await this.init(this.deviceId);
    const context = this.context!;
    if (context.state === "suspended") await context.resume();
    const source = context.createMediaStreamSource(this.stream!);
    const chunks: Float32Array[] = [];
    const targetFrames = Math.ceil(durationS * context.sampleRate);
    let collected = 0;
    const started = performance.now();

    const push = (data: Float32Array) => {
      if (collected >= targetFrames) return;
      chunks.push(data);
      collected += data.length;
      if (onLevel) {
        let sum = 0;
        let peak = 0;
        for (let i = 0; i < data.length; i++) {
          const v = data[i];
          sum += v * v;
          if (Math.abs(v) > peak) peak = Math.abs(v);
        }
        onLevel({ rms: Math.sqrt(sum / data.length), peak, elapsed: (performance.now() - started) / 1000 });
      }
    };

    let cleanup: () => void;
    if (context.audioWorklet && this.workletUrl) {
      const node = new AudioWorkletNode(context, "capture-processor");
      node.port.onmessage = (e) => push(e.data as Float32Array);
      source.connect(node);
      const sink = context.createGain();
      sink.gain.value = 0;
      node.connect(sink).connect(context.destination);
      cleanup = () => {
        node.port.onmessage = null;
        source.disconnect();
        node.disconnect();
        sink.disconnect();
      };
    } else {
      const processor = context.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => push(e.inputBuffer.getChannelData(0).slice(0));
      source.connect(processor);
      processor.connect(context.destination);
      cleanup = () => {
        processor.onaudioprocess = null;
        source.disconnect();
        processor.disconnect();
      };
    }

    await new Promise<void>((resolve) => {
      const tick = () => {
        if (collected >= targetFrames) resolve();
        else setTimeout(tick, 30);
      };
      tick();
    });
    cleanup();

    const merged = new Float32Array(targetFrames);
    let offset = 0;
    for (const chunk of chunks) {
      const remaining = targetFrames - offset;
      if (remaining <= 0) break;
      merged.set(remaining >= chunk.length ? chunk : chunk.subarray(0, remaining), offset);
      offset += chunk.length;
    }

    const resampled = await resample(merged, context.sampleRate, TARGET_SAMPLE_RATE);
    return { wav: encodeWav(resampled, TARGET_SAMPLE_RATE), samples: resampled, sampleRate: TARGET_SAMPLE_RATE };
  }

  close(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.context?.close();
    this.context = null;
    if (this.workletUrl) URL.revokeObjectURL(this.workletUrl);
    this.workletUrl = null;
  }
}

export async function resample(samples: Float32Array, fromRate: number, toRate: number): Promise<Float32Array> {
  if (fromRate === toRate) return samples;
  const length = Math.round((samples.length * toRate) / fromRate);
  try {
    const offline = new OfflineAudioContext(1, length, toRate);
    const buffer = offline.createBuffer(1, samples.length, fromRate);
    buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
    const src = offline.createBufferSource();
    src.buffer = buffer;
    src.connect(offline.destination);
    src.start(0);
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0).slice(0);
  } catch {
    // linear interpolation fallback
    const out = new Float32Array(length);
    const ratio = fromRate / toRate;
    for (let i = 0; i < length; i++) {
      const pos = i * ratio;
      const i0 = Math.floor(pos);
      const i1 = Math.min(i0 + 1, samples.length - 1);
      const frac = pos - i0;
      out[i] = samples[i0] * (1 - frac) + samples[i1] * frac;
    }
    return out;
  }
}

export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);
  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/** Down-samples audio to a fixed number of peak values for a tiny waveform preview. */
export function waveformPeaks(samples: Float32Array, buckets = 80): number[] {
  const size = Math.max(1, Math.floor(samples.length / buckets));
  const peaks: number[] = [];
  for (let b = 0; b < buckets; b++) {
    let peak = 0;
    const start = b * size;
    for (let i = start; i < Math.min(samples.length, start + size); i++) {
      const v = Math.abs(samples[i]);
      if (v > peak) peak = v;
    }
    peaks.push(peak);
  }
  return peaks;
}

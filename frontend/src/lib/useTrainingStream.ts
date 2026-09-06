import { useEffect, useRef, useState } from "react";
import { api, type TrainingState } from "../api";

const EMPTY: TrainingState = {
  status: "idle",
  job_id: null,
  wake_word: null,
  stage: null,
  stage_key: null,
  message: null,
  message_key: null,
  message_params: null,
  progress: { current: 0, total: 0 },
  step: 0,
  total_steps: 0,
  eval_step_interval: 0,
  train_metrics: null,
  validation: [],
  best: null,
  final_metrics: null,
  model_url: null,
  manifest_url: null,
  started_at: null,
  finished_at: null,
  error: null,
};

const MAX_LOG = 400;

export function useTrainingStream() {
  const [state, setState] = useState<TrainingState>(EMPTY);
  const [log, setLog] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    let closed = false;
    const connect = () => {
      const source = new EventSource("/api/train/status");
      sourceRef.current = source;
      source.onopen = () => setConnected(true);
      source.addEventListener("snapshot", (e) => {
        const snap = JSON.parse((e as MessageEvent).data) as TrainingState;
        setState({ ...EMPTY, ...snap });
        setLog(snap.log_tail ?? []);
      });
      source.addEventListener("state", (e) => {
        const partial = JSON.parse((e as MessageEvent).data) as Partial<TrainingState>;
        setState((prev) => ({ ...prev, ...partial }));
      });
      source.addEventListener("log", (e) => {
        const { line } = JSON.parse((e as MessageEvent).data) as { line: string };
        setLog((prev) => (prev.length >= MAX_LOG ? [...prev.slice(prev.length - MAX_LOG + 1), line] : [...prev, line]));
      });
      source.onerror = () => {
        setConnected(false);
        // EventSource reconnects on its own; also refresh the snapshot so we don't miss state.
        api.trainingSnapshot().then((snap) => !closed && setState({ ...EMPTY, ...snap })).catch(() => undefined);
      };
    };
    connect();
    return () => {
      closed = true;
      sourceRef.current?.close();
    };
  }, []);

  return { state, log, connected, setState };
}

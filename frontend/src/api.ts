export type TrainingParams = {
  training_steps: number;
  learning_rate: number;
  batch_size: number;
  eval_step_interval: number;
  augmentations_per_sample: number;
  clip_duration_ms: number;
  negative_class_weight: number;
  positive_class_weight: number;
};

export type Project = {
  wake_word: string;
  sample_duration_s: number;
  training: TrainingParams;
};

export type QualityIssue = "cut_start" | "cut_end" | "too_short" | "silent" | "clipping" | "too_quiet" | "unreadable";

export type Recording = {
  id: string;
  kind: "positive" | "negative";
  duration: number;
  size: number;
  created: string;
  url: string;
  peaks: number[];
  quality: {
    peak?: number;
    rms_db?: number;
    speech_start?: number | null;
    speech_end?: number | null;
    issues: QualityIssue[];
  };
};

export type Dataset = {
  id: string;
  title: string;
  description: { cs: string; en: string };
  size_mb: number;
  required: boolean;
  installed: boolean;
  download: null | { state: string; received?: number; total?: number | null; error?: string | null };
};

export type ValidationEntry = {
  step: number;
  recall_at_no_faph: number;
  cutoff_for_no_faph: number;
  accuracy: number;
  recall: number;
  precision: number;
  ambient_false_positives: number;
  false_positives_per_hour: number;
  loss: number;
  auc: number;
  average_viable_recall: number;
};

export type FinalMetrics = { auc: number | null; cutoff: number; frr: number; faph: number; manifest_cutoff: number };

export type TrainingState = {
  status: "idle" | "downloading" | "preparing" | "training" | "converting" | "done" | "failed" | "cancelled";
  job_id: string | null;
  wake_word: string | null;
  stage: string | null;
  stage_key: string | null;
  message: string | null;
  message_key: string | null;
  message_params: Record<string, string | number> | null;
  progress: { current: number; total: number };
  step: number;
  total_steps: number;
  eval_step_interval: number;
  train_metrics: null | { accuracy: number; recall: number; precision: number; loss: number };
  validation: ValidationEntry[];
  best: null | { minimization: number; maximization: number };
  final_metrics: FinalMetrics | null;
  model_url: string | null;
  manifest_url: string | null;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  log_tail?: string[];
};

export type Job = {
  job_id: string;
  wake_word: string;
  slug: string;
  created_at: string;
  finished_at: string | null;
  status: string;
  positive_count: number;
  training: TrainingParams;
  final_metrics: FinalMetrics | null;
  model_url: string | null;
  manifest_url: string | null;
  model_size: number | null;
};

export class ApiError extends Error {
  code: string | null;
  constructor(message: string, code: string | null = null) {
    super(message);
    this.code = code;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body;
    } catch {
      /* ignore */
    }
    if (detail && typeof detail === "object" && "code" in (detail as object)) {
      const d = detail as { code: string; message?: string };
      throw new ApiError(d.message ?? d.code, d.code);
    }
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export type TestInfo = { job_id: string; model: string; probability_cutoff: number; sliding_window_size: number };

export type EvaluationItem = {
  id: string;
  kind: "positive" | "negative";
  url: string;
  max_probability: number | null;
  detections: number;
  error?: string;
};

export type Evaluation = {
  cutoff: number;
  window: number;
  items: EvaluationItem[];
  summary: { positive_total: number; positive_detected: number; negative_total: number; negative_triggered: number };
};

export const api = {
  getConfig: () => request<{ project: Project; defaults: TrainingParams }>("/api/config"),
  saveConfig: (update: Partial<Project>) =>
    request<{ project: Project; defaults: TrainingParams }>("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    }),
  listRecordings: (kind: string) => request<{ items: Recording[]; count: number }>(`/api/recordings?kind=${kind}`),
  uploadRecording: (kind: string, wav: Blob, filename = "sample.wav") => {
    const form = new FormData();
    form.append("kind", kind);
    form.append("file", wav, filename);
    return request<Recording>("/api/recordings", { method: "POST", body: form });
  },
  deleteRecording: (kind: string, id: string) => request<{ deleted: string }>(`/api/recordings/${kind}/${id}`, { method: "DELETE" }),
  restoreRecording: (kind: string, id: string) => request<Recording>(`/api/recordings/${kind}/${id}/restore`, { method: "POST" }),
  listDatasets: () => request<{ items: Dataset[] }>("/api/datasets"),
  downloadDataset: (id: string) => request<unknown>(`/api/datasets/${id}/download`, { method: "POST" }),
  deleteDataset: (id: string) => request<unknown>(`/api/datasets/${id}`, { method: "DELETE" }),
  startTraining: () => request<TrainingState>("/api/train", { method: "POST" }),
  cancelTraining: () => request<TrainingState>("/api/train/cancel", { method: "POST" }),
  trainingSnapshot: () => request<TrainingState>("/api/train"),
  listJobs: () => request<{ items: Job[] }>("/api/jobs"),
  testInfo: (jobId: string) => request<TestInfo>(`/api/jobs/${jobId}/test-info`),
  evaluateJob: (jobId: string, cutoff: number, window: number) =>
    request<Evaluation>(`/api/jobs/${jobId}/evaluate?cutoff=${cutoff}&window=${window}`, { method: "POST" }),
  deleteJob: (id: string) => request<unknown>(`/api/jobs/${id}`, { method: "DELETE" }),
};

export type TrainingTarget = "esphome" | "wyoming";

export type TrainingParams = {
  target: TrainingTarget;
  hard_negatives: boolean;
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
  id: string;
  name: string;
  wake_word: string;
  sample_duration_s: number;
  training: TrainingParams;
  share_token: string | null;
  contributor_target: number;
  webhook_url: string;
  created_at?: string;
};

export type SystemInfo = {
  version: string;
  latest_version: string | null;
  update_available: boolean;
  releases_url: string;
  gpu: null | { name: string; memory_total_mb: number; memory_used_mb: number; driver: string; utilization: number | null };
  gpu_available: boolean;
  tensorflow_cuda: boolean;
  cpu_count: number | null;
  last_training: null | { gpu: boolean; devices: string[]; tensorflow: string; at: string; job_id: string };
};

export type Contributor = { name: string; positive: number; negative: number };
export type Tag = "normal" | "far" | "noisy" | "whisper" | "loud";
export const TAGS: Tag[] = ["normal", "far", "noisy", "whisper", "loud"];

export type ProjectSummary = {
  id: string;
  name: string;
  wake_word: string;
  created_at: string | null;
  positive_count: number;
  negative_count: number;
  jobs: number;
  current: boolean;
  shared: boolean;
};

export type ContributeInfo = { project: string; wake_word: string; sample_duration_s: number; positive_count: number; contributor_target: number; tags: string[] };

export type QualityIssue = "cut_start" | "cut_end" | "too_short" | "silent" | "clipping" | "too_quiet" | "unreadable";

export type Recording = {
  id: string;
  kind: "positive" | "negative";
  duration: number;
  size: number;
  created: string;
  url: string;
  contributor: string | null;
  tag: Tag | null;
  review: boolean;
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

export type RocPoint = { cutoff: number; frr: number; faph: number };
export type AutoThreshold = {
  cutoff: number;
  window?: number;
  margin?: number;
  positives_total: number;
  positives_passed: number;
  positives_p05: number;
  negatives_total: number;
  negatives_max: number;
  negatives_triggered: number;
};
export type FinalMetrics = { auc: number | null; cutoff: number; frr: number; faph: number; manifest_cutoff: number; points?: RocPoint[]; auto_threshold?: AutoThreshold | null };

export type TrainingState = {
  status: "idle" | "downloading" | "preparing" | "training" | "converting" | "done" | "failed" | "cancelled" | "interrupted";
  job_id: string | null;
  project_id: string | null;
  label: string;
  queue: QueuedRun[];
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
  export_url: string | null;
  resumable: boolean;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  log_tail?: string[];
};

export type QueuedRun = { id: string; project_id: string; overrides: Partial<TrainingParams>; label: string; queued_at: string };

export type MonitorItem = Recording & { job_id?: string };
export type DeviceEvent = { at: string; device: string; wake_word: string; esphome_version: string; probability: number | null };
export type PublicUrls = { token: string; manifest_url: string; model_url: string; device_event_url: string; has_model: boolean; job_id: string | null; target: TrainingTarget; slug: string; minimum_esphome_version: string; snippet: string };

export type Job = {
  job_id: string;
  wake_word: string;
  label: string;
  target: TrainingTarget;
  overrides: Partial<TrainingParams>;
  slug: string;
  created_at: string;
  finished_at: string | null;
  status: string;
  positive_count: number;
  training: TrainingParams;
  final_metrics: FinalMetrics | null;
  validation_last: ValidationEntry | null;
  model_url: string | null;
  manifest_url: string | null;
  export_url: string | null;
  model_size: number | null;
  resumable: boolean;
};

export type RecordingsClient = {
  listRecordings: (kind: string) => Promise<{ items: Recording[]; count: number }>;
  uploadRecording: (kind: string, wav: Blob, filename?: string, tag?: string | null) => Promise<Recording>;
  deleteRecording: (kind: string, id: string) => Promise<{ deleted: string }>;
  restoreRecording: (kind: string, id: string) => Promise<Recording>;
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

export type TestInfo = { job_id: string; model: string; target: TrainingTarget; probability_cutoff: number; sliding_window_size: number };

export type EvaluationItem = {
  id: string;
  kind: "positive" | "negative";
  url: string;
  tag: Tag | null;
  contributor: string | null;
  outlier: boolean;
  max_probability: number | null;
  detections: number;
  error?: string;
};

export type Evaluation = {
  cutoff: number;
  window: number;
  items: EvaluationItem[];
  summary: {
    positive_total: number;
    positive_detected: number;
    negative_total: number;
    negative_triggered: number;
    by_tag: Record<string, { total: number; detected: number }>;
    outliers: number;
    median_positive: number;
  };
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
  uploadRecording: (kind: string, wav: Blob, filename = "sample.wav", tag: string | null = null) => {
    const form = new FormData();
    form.append("kind", kind);
    if (tag) form.append("tag", tag);
    form.append("file", wav, filename);
    return request<Recording>("/api/recordings", { method: "POST", body: form });
  },
  setTag: (kind: string, id: string, tag: string | null) =>
    request<Recording>(`/api/recordings/${kind}/${id}/tag`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tag }) }),
  contributors: () => request<{ items: Contributor[]; tags: string[] }>("/api/recordings/contributors"),
  system: () => request<SystemInfo>("/api/system"),
  exportProjectUrl: (id: string) => `/api/projects/${id}/export`,
  importProject: (file: File) => {
    const form = new FormData();
    form.append("file", file, file.name);
    return request<{ items: ProjectSummary[]; current: string }>("/api/projects/import", { method: "POST", body: form });
  },
  deleteRecording: (kind: string, id: string) => request<{ deleted: string }>(`/api/recordings/${kind}/${id}`, { method: "DELETE" }),
  restoreRecording: (kind: string, id: string) => request<Recording>(`/api/recordings/${kind}/${id}/restore`, { method: "POST" }),
  listDatasets: () => request<{ items: Dataset[] }>("/api/datasets"),
  downloadDataset: (id: string) => request<unknown>(`/api/datasets/${id}/download`, { method: "POST" }),
  deleteDataset: (id: string) => request<unknown>(`/api/datasets/${id}`, { method: "DELETE" }),
  startTraining: (overrides?: Partial<TrainingParams>, label?: string) =>
    request<TrainingState>("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ training: overrides ?? {}, label: label ?? "" }) }),
  startSweep: (param: keyof TrainingParams, values: number[], label?: string) =>
    request<TrainingState>("/api/train/sweep", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ param, values, label }) }),
  dropQueued: (id: string) => request<{ queue: QueuedRun[] }>(`/api/train/queue/${id}`, { method: "DELETE" }),
  setReview: (kind: string, id: string, review: boolean) =>
    request<Recording>(`/api/recordings/${kind}/${id}/review`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ review }) }),
  listMonitor: () => request<{ items: MonitorItem[] }>("/api/monitor"),
  monitorToNegative: (id: string) => request<Recording>(`/api/monitor/${id}/negative`, { method: "POST" }),
  deleteMonitor: (id: string) => request<{ deleted: string }>(`/api/monitor/${id}`, { method: "DELETE" }),
  clearMonitor: () => request<{ deleted: number }>("/api/monitor", { method: "DELETE" }),
  monitorAdoptAll: () => request<{ moved: number }>("/api/monitor/adopt-all", { method: "POST" }),
  deviceEvents: (since?: string) => request<{ items: DeviceEvent[]; minimum_esphome_version: string; outdated_versions: string[]; now: string }>(`/api/monitor/device-events${since ? `?since=${encodeURIComponent(since)}` : ""}`),
  publicUrls: (projectId: string) => request<PublicUrls>(`/api/projects/${projectId}/public-urls`),
  bundleUrl: (ids?: string[]) => `/api/bundle${ids && ids.length ? `?projects=${ids.join(",")}` : ""}`,
  resumeTraining: () => request<TrainingState>("/api/train/resume", { method: "POST" }),
  listProjects: () => request<{ items: ProjectSummary[]; current: string }>("/api/projects"),
  createProject: (name: string, wakeWord: string) =>
    request<{ items: ProjectSummary[]; current: string }>("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, wake_word: wakeWord }),
    }),
  selectProject: (id: string) => request<{ items: ProjectSummary[]; current: string }>(`/api/projects/${id}/select`, { method: "POST" }),
  deleteProject: (id: string) => request<{ items: ProjectSummary[]; current: string }>(`/api/projects/${id}`, { method: "DELETE" }),
  setShare: (id: string, enabled: boolean) =>
    request<{ share_token: string | null }>(`/api/projects/${id}/share`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    }),
  cancelTraining: () => request<TrainingState>("/api/train/cancel", { method: "POST" }),
  trainingSnapshot: () => request<TrainingState>("/api/train"),
  listJobs: () => request<{ items: Job[] }>("/api/jobs"),
  testInfo: (jobId: string) => request<TestInfo>(`/api/jobs/${jobId}/test-info`),
  evaluateJob: (jobId: string, cutoff: number, window: number) =>
    request<Evaluation>(`/api/jobs/${jobId}/evaluate?cutoff=${cutoff}&window=${window}`, { method: "POST" }),
  deleteJob: (id: string) => request<unknown>(`/api/jobs/${id}`, { method: "DELETE" }),
};

/** API client for the contributor page (token based, no login). */
export function contributeClient(token: string, name: string): RecordingsClient & { info: () => Promise<ContributeInfo> } {
  const q = `token=${encodeURIComponent(token)}&name=${encodeURIComponent(name)}`;
  return {
    info: () => request<ContributeInfo>(`/api/contribute/info?token=${encodeURIComponent(token)}`),
    listRecordings: (kind) => request(`/api/contribute/recordings?${q}&kind=${kind}`),
    uploadRecording: (kind, wav, filename = "sample.wav", tag = null) => {
      const form = new FormData();
      form.append("kind", kind);
      if (tag) form.append("tag", tag);
      form.append("file", wav, filename);
      return request(`/api/contribute/recordings?${q}`, { method: "POST", body: form });
    },
    deleteRecording: (kind, id) => request(`/api/contribute/recordings/${kind}/${id}?${q}`, { method: "DELETE" }),
    restoreRecording: (kind, id) => request(`/api/contribute/recordings/${kind}/${id}/restore?${q}`, { method: "POST" }),
  };
}

import { api, ApiRequestError, type TranscriptionJob } from "../api";
import { icon, type IconName } from "../ui/icons";
import type { Upload, Uploads } from "./file-transcription";
import { escapeHtml } from "./html";

export const JOBS_POLL_MS = 1000;

const KIND_ICONS: Record<TranscriptionJob["kind"], IconName> = {
  file: "file",
  meeting: "users",
  dictation: "mic",
};

const RUNNING: ReadonlySet<TranscriptionJob["stage"]> = new Set(["queued", "transcribing", "saving"]);

export const REMOVE_REFUSED = "Couldn't remove this — try again";
export const RETRY_REFUSED = "Couldn't restart this — try again";

/** Whether the job is still waiting or being turned into text. */
export function jobIsRunning(job: TranscriptionJob): boolean {
  return RUNNING.has(job.stage);
}

/** What the card's × says it does: a meeting's or dictation's recording is deleted with its card, a file is cancelled or dismissed. */
export function removeLabel(job: TranscriptionJob): string {
  if (job.kind !== "file") return "Delete this recording";
  return job.stage === "failed" ? "Dismiss" : "Cancel";
}

/** What the card says under its bar: the stage, the percentage when it can be told, or why it failed. */
export function jobStatusText(job: TranscriptionJob): string {
  switch (job.stage) {
    case "queued":
      return "Waiting";
    case "saving":
      return "Saving";
    case "failed":
      return job.error ?? `Couldn't turn this ${job.kind} into text`;
    default:
      return job.progress === null ? "Transcribing" : `Transcribing · ${Math.round(job.progress * 100)}%`;
  }
}

/** The card a job is drawn on, without its stage: name, bar, status line, Try again and ×. */
function jobCardShell(kind: TranscriptionJob["kind"], name: string): HTMLElement {
  const card = document.createElement("article");
  card.className = "entry entry--job";
  card.dataset.kind = kind;
  card.innerHTML = `
    <span class="entry-dot">${icon(KIND_ICONS[kind], "small")}</span>
    <p class="entry-job-name num">${escapeHtml(name)}</p>
    <div class="entry-progress"><i></i></div>
    <div class="entry-meta"><span class="entry-job-status" role="status"></span><span class="entry-actions">
      <button type="button" class="entry-retry" data-action="retry" hidden>Try again</button>
      <button type="button" data-action="remove">${icon("x", "small")}</button>
    </span></div>
  `;
  return card;
}

/** What an upload's card says under its bar. */
export function uploadStatusText(upload: Upload): string {
  if (upload.state === "waiting") return "Waiting";
  if (upload.state === "failed") return upload.error ?? "";
  return `Uploading · ${Math.round(upload.sentFraction * 100)}%`;
}

/** The card of a file still on its way to the backend, drawn as a job's card: × stops or
 *  clears it, Try again sends it once more. `renderUploadCard` paints its state. */
export function createUploadCard(upload: Upload, uploads: Pick<Uploads, "remove" | "retry">): HTMLElement {
  const card = jobCardShell("file", upload.name);
  card.querySelector('[data-action="remove"]')!.addEventListener("click", () => uploads.remove(upload));
  card.querySelector('[data-action="retry"]')!.addEventListener("click", () => uploads.retry(upload));
  return card;
}

export function renderUploadCard(card: HTMLElement, upload: Upload): void {
  const failed = upload.state === "failed";
  card.classList.toggle("entry--job-failed", failed);
  card.querySelector(".entry-job-status")!.textContent = uploadStatusText(upload);
  card.querySelector<HTMLElement>('[data-action="retry"]')!.hidden = !(failed && upload.retryable);
  card.querySelector('[data-action="remove"]')!.setAttribute("aria-label", failed ? "Dismiss" : "Cancel");
  card.querySelector<HTMLElement>(".entry-progress i")!.style.width = `${upload.sentFraction * 100}%`;
}

export interface JobCards {
  /** Reads the jobs now, and again every second for as long as one is running. */
  refresh(): Promise<void>;
  /** Stops reading until the next `refresh`. */
  pause(): void;
}

/**
 * The cards of files, meetings and dictations being transcribed. `show` receives them newest first
 * whenever the set changes. A failed job offers Try again. A finished job's card
 * stays, asking `entrySaved` to paint its row, until
 * `entryShown` says the row is painted, so the card turns into the entry where it stood. A job this
 * page removed is never painted again; a failed read is tried again a second later. A refused × or
 * Try again is said on the card's status line until the job moves on or the card is clicked again.
 */
export function createJobCards(
  show: (cards: HTMLElement[]) => void,
  entrySaved: () => Promise<void>,
  entryShown: (entryId: string) => boolean,
): JobCards {
  const cards = new Map<string, HTMLElement>();
  const gone = new Set<string>();
  const refusals = new Map<string, { stage: string; text: string }>();
  let timer: number | null = null;
  let generation = 0;

  function publish(): void {
    show([...cards.values()]);
  }

  function removeCard(id: string): void {
    gone.add(id);
    refusals.delete(id);
    if (cards.delete(id)) publish();
  }

  function sayRefused(card: HTMLElement, id: string, text: string): void {
    refusals.set(id, { stage: card.dataset.stage!, text });
    card.querySelector(".entry-job-status")!.textContent = text;
  }

  function createCard(job: TranscriptionJob): HTMLElement {
    const card = jobCardShell(job.kind, job.name);
    card.dataset.job = job.id;
    card.querySelector('[data-action="remove"]')!.addEventListener("click", async () => {
      refusals.delete(job.id);
      try {
        await api.removeJob(job.id);
        removeCard(job.id);
      } catch (err) {
        if (err instanceof ApiRequestError && err.status === 404) {
          removeCard(job.id);
          return;
        }
        console.error(err);
        sayRefused(card, job.id, REMOVE_REFUSED);
      }
    });
    card.querySelector('[data-action="retry"]')!.addEventListener("click", async () => {
      refusals.delete(job.id);
      try {
        await api.retryJob(job.id);
      } catch (err) {
        console.error(err);
        sayRefused(card, job.id, RETRY_REFUSED);
      }
      await refresh();
    });
    return card;
  }

  function paint(card: HTMLElement, job: TranscriptionJob): void {
    if (refusals.get(job.id)?.stage !== job.stage) refusals.delete(job.id);
    card.dataset.stage = job.stage;
    card.classList.toggle("entry--job-failed", job.stage === "failed");
    card.querySelector(".entry-job-status")!.textContent = refusals.get(job.id)?.text ?? jobStatusText(job);
    card.querySelector<HTMLElement>('[data-action="retry"]')!.hidden = job.stage !== "failed";
    card.querySelector<HTMLElement>('[data-action="remove"]')!.hidden =
      job.kind === "dictation" && job.stage === "transcribing";
    card.querySelector('[data-action="remove"]')!.setAttribute("aria-label", removeLabel(job));
    const bar = card.querySelector<HTMLElement>(".entry-progress i")!;
    if (job.progress !== null) {
      bar.style.width = `${job.progress * 100}%`;
    } else if (job.stage === "queued") {
      bar.style.width = "0";
    }
  }

  function lingers(job: TranscriptionJob, card: HTMLElement | undefined): card is HTMLElement {
    const entryId = job.entry_id;
    if (job.stage !== "done" || card === undefined || entryId === null) {
      gone.add(job.id);
      return false;
    }
    paint(card, job);
    void entrySaved().then(() => {
      if (entryShown(entryId)) removeCard(job.id);
    });
    return true;
  }

  function readAgainSoon(): void {
    timer = window.setTimeout(() => void refresh(), JOBS_POLL_MS);
  }

  async function refresh(): Promise<void> {
    pause();
    const current = generation;
    let jobs: TranscriptionJob[];
    try {
      jobs = await api.jobs();
    } catch (err) {
      console.error(err);
      if (current === generation) readAgainSoon();
      return;
    }
    if (current !== generation) return;
    const order: HTMLElement[] = [];
    let waiting = false;
    for (const job of jobs) {
      if (gone.has(job.id)) continue;
      const existing = cards.get(job.id);
      if (!jobIsRunning(job) && job.stage !== "failed") {
        if (lingers(job, existing)) {
          order.push(existing);
          waiting = true;
        }
        continue;
      }
      const card = existing ?? createCard(job);
      paint(card, job);
      order.push(card);
    }
    const painted = [...cards.values()];
    if (order.length !== painted.length || order.some((card, i) => painted[i] !== card)) {
      cards.clear();
      for (const card of order) cards.set(card.dataset.job!, card);
      publish();
    }
    if (waiting || jobs.some(jobIsRunning)) readAgainSoon();
  }

  function pause(): void {
    generation += 1;
    if (timer !== null) window.clearTimeout(timer);
    timer = null;
  }

  return { refresh, pause };
}

import { api, type FileJob } from "../api";
import { icon, type IconName } from "../ui/icons";
import { escapeHtml } from "./html";

export const JOBS_POLL_MS = 1000;

const KIND_ICONS: Record<FileJob["kind"], IconName> = { file: "file" };

const RUNNING: ReadonlySet<FileJob["stage"]> = new Set(["queued", "transcribing", "saving"]);

/** What the card says under its bar: the stage, the percentage when it can be told, or why it failed. */
export function jobStatusText(job: FileJob): string {
  switch (job.stage) {
    case "queued":
      return "Waiting";
    case "saving":
      return "Saving";
    case "failed":
      return job.error ?? "Couldn't turn this file into text";
    default:
      return job.progress === null ? "Transcribing" : `Transcribing · ${Math.round(job.progress * 100)}%`;
  }
}

export interface JobCards {
  /** Reads the jobs now, and again every second for as long as one is running. */
  refresh(): Promise<void>;
  /** Stops reading until the next `refresh`. */
  pause(): void;
}

/**
 * The cards of files being transcribed. `show` receives them newest first whenever the set
 * changes; a finished job waits for `entrySaved` to paint its row before its card goes, so the
 * card turns into the entry where it stood. A job this page removed is never painted again.
 */
export function createJobCards(
  show: (cards: HTMLElement[]) => void,
  entrySaved: () => Promise<void>,
): JobCards {
  const cards = new Map<string, HTMLElement>();
  const gone = new Set<string>();
  let timer: number | null = null;
  let generation = 0;

  function publish(): void {
    show([...cards.values()]);
  }

  function removeCard(id: string): void {
    gone.add(id);
    if (cards.delete(id)) publish();
  }

  function createCard(job: FileJob): HTMLElement {
    const card = document.createElement("article");
    card.className = "entry entry--job";
    card.dataset.job = job.id;
    card.innerHTML = `
      <span class="entry-dot">${icon(KIND_ICONS[job.kind], "small")}</span>
      <p class="entry-job-name num">${escapeHtml(job.name)}</p>
      <div class="entry-progress"><i></i></div>
      <div class="entry-meta"><span class="entry-job-status" role="status"></span><span class="entry-actions">
        <button type="button" data-action="remove">${icon("x", "small")}</button>
      </span></div>
    `;
    card.querySelector("button")!.addEventListener("click", async () => {
      try {
        await api.removeJob(job.id);
        removeCard(job.id);
      } catch (err) {
        console.error(err);
      }
    });
    return card;
  }

  function paint(card: HTMLElement, job: FileJob): void {
    card.classList.toggle("entry--job-failed", job.stage === "failed");
    card.querySelector(".entry-job-status")!.textContent = jobStatusText(job);
    card.querySelector("button")!.setAttribute("aria-label", job.stage === "failed" ? "Dismiss" : "Cancel");
    if (job.progress !== null) {
      card.querySelector<HTMLElement>(".entry-progress i")!.style.width = `${job.progress * 100}%`;
    }
  }

  function finish(job: FileJob, card: HTMLElement | undefined): HTMLElement | null {
    gone.add(job.id);
    if (job.stage !== "done" || card === undefined) return null;
    paint(card, job);
    void entrySaved().finally(() => removeCard(job.id));
    return card;
  }

  async function refresh(): Promise<void> {
    pause();
    const current = generation;
    let jobs: FileJob[];
    try {
      jobs = await api.jobs();
    } catch (err) {
      console.error(err);
      return;
    }
    if (current !== generation) return;
    const order: HTMLElement[] = [];
    for (const job of jobs) {
      if (gone.has(job.id)) continue;
      const existing = cards.get(job.id);
      if (!RUNNING.has(job.stage) && job.stage !== "failed") {
        const lingering = finish(job, existing);
        if (lingering) order.push(lingering);
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
    if (jobs.some((job) => RUNNING.has(job.stage))) {
      timer = window.setTimeout(() => void refresh(), JOBS_POLL_MS);
    }
  }

  function pause(): void {
    generation += 1;
    if (timer !== null) window.clearTimeout(timer);
    timer = null;
  }

  return { refresh, pause };
}

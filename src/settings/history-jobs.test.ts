// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FileJob } from "../api";

const apiMock = {
  jobs: vi.fn(),
  removeJob: vi.fn(),
};

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, api: apiMock };
});

const { createJobCards, jobStatusText, JOBS_POLL_MS } = await import("./history-jobs");

function job(overrides: Partial<FileJob> = {}): FileJob {
  return {
    id: "j1",
    kind: "file",
    name: "interview.m4a",
    stage: "transcribing",
    progress: 0.36,
    error: null,
    entry_id: null,
    ...overrides,
  };
}

let shown: HTMLElement[];
const show = (cards: HTMLElement[]) => (shown = cards);
const entrySaved = vi.fn();

function status(card: HTMLElement): string {
  return card.querySelector(".entry-job-status")!.textContent!;
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  shown = [];
  entrySaved.mockResolvedValue(undefined);
  apiMock.removeJob.mockResolvedValue({ outcome: "cancelled" });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("jobStatusText", () => {
  it("says the stage, the percentage when known, and the reason for a failure", () => {
    expect(jobStatusText(job())).toBe("Transcribing · 36%");
    expect(jobStatusText(job({ progress: null }))).toBe("Transcribing");
    expect(jobStatusText(job({ stage: "queued", progress: null }))).toBe("Waiting");
    expect(jobStatusText(job({ stage: "saving", progress: null }))).toBe("Saving");
    expect(jobStatusText(job({ stage: "failed", error: "We didn't hear any speech in this file" }))).toBe(
      "We didn't hear any speech in this file",
    );
  });
});

describe("createJobCards", () => {
  it("paints a running job with its name, bar and status, and reads again every second", async () => {
    apiMock.jobs.mockResolvedValue([job()]);
    const cards = createJobCards(show, entrySaved);

    await cards.refresh();

    expect(shown).toHaveLength(1);
    expect(shown[0].querySelector(".entry-job-name")!.textContent).toBe("interview.m4a");
    expect(shown[0].querySelector<HTMLElement>(".entry-progress i")!.style.width).toBe("36%");
    expect(status(shown[0])).toBe("Transcribing · 36%");

    apiMock.jobs.mockResolvedValue([job({ progress: 0.5 })]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    expect(apiMock.jobs).toHaveBeenCalledTimes(2);
    expect(status(shown[0])).toBe("Transcribing · 50%");
  });

  it("stops reading once nothing is running, and while paused", async () => {
    apiMock.jobs.mockResolvedValue([]);
    const idle = createJobCards(show, entrySaved);
    await idle.refresh();
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS * 5);
    expect(apiMock.jobs).toHaveBeenCalledTimes(1);

    apiMock.jobs.mockResolvedValue([job()]);
    const running = createJobCards(show, entrySaved);
    await running.refresh();
    running.pause();
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS * 5);
    expect(apiMock.jobs).toHaveBeenCalledTimes(2);
  });

  it("keeps a finished card until its entry is painted, then lets it go", async () => {
    apiMock.jobs.mockResolvedValue([job()]);
    const cards = createJobCards(show, entrySaved);
    await cards.refresh();
    let paintEntry!: () => void;
    entrySaved.mockReturnValue(new Promise<void>((resolve) => (paintEntry = resolve)));

    apiMock.jobs.mockResolvedValue([job({ stage: "done", progress: 1, entry_id: "e1" })]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    expect(entrySaved).toHaveBeenCalledTimes(1);
    expect(shown).toHaveLength(1);

    paintEntry();
    await vi.advanceTimersByTimeAsync(0);
    expect(shown).toEqual([]);
  });

  it("does not paint a job that finished while the page was away", async () => {
    apiMock.jobs.mockResolvedValue([job({ stage: "done", progress: 1, entry_id: "e1" }), job({ id: "j2", stage: "cancelled" })]);
    const cards = createJobCards(show, entrySaved);

    await cards.refresh();

    expect(shown).toEqual([]);
    expect(entrySaved).not.toHaveBeenCalled();
  });

  it("cancels from the ×, and a later read listing the job as cancelled paints nothing", async () => {
    apiMock.jobs.mockResolvedValue([job()]);
    const cards = createJobCards(show, entrySaved);
    await cards.refresh();
    const button = shown[0].querySelector<HTMLButtonElement>("button")!;
    expect(button.getAttribute("aria-label")).toBe("Cancel");

    button.click();
    await vi.advanceTimersByTimeAsync(0);
    expect(apiMock.removeJob).toHaveBeenCalledWith("j1");
    expect(shown).toEqual([]);

    apiMock.jobs.mockResolvedValue([job({ stage: "cancelled" })]);
    await cards.refresh();
    expect(shown).toEqual([]);
  });

  it("keeps a card whose cancel the backend refused", async () => {
    apiMock.jobs.mockResolvedValue([job({ stage: "saving", progress: null })]);
    apiMock.removeJob.mockRejectedValue(new Error("This file is already being saved to History"));
    vi.spyOn(console, "error").mockImplementation(() => {});
    const cards = createJobCards(show, entrySaved);
    await cards.refresh();

    shown[0].querySelector<HTMLButtonElement>("button")!.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(shown).toHaveLength(1);
  });

  it("shows a failure's reason with a Dismiss that removes it", async () => {
    apiMock.jobs.mockResolvedValue([job({ stage: "failed", progress: null, error: "Add an API key in Settings" })]);
    apiMock.removeJob.mockResolvedValue({ outcome: "dismissed" });
    const cards = createJobCards(show, entrySaved);
    await cards.refresh();

    const card = shown[0];
    expect(card.classList.contains("entry--job-failed")).toBe(true);
    expect(status(card)).toBe("Add an API key in Settings");
    card.querySelector<HTMLButtonElement>('button[aria-label="Dismiss"]')!.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(shown).toEqual([]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS * 3);
    expect(apiMock.jobs).toHaveBeenCalledTimes(1);
  });

  it("drops what a read left in flight when the page paused", async () => {
    let answer!: (jobs: FileJob[]) => void;
    apiMock.jobs.mockReturnValue(new Promise((resolve) => (answer = resolve)));
    const cards = createJobCards(show, entrySaved);

    const reading = cards.refresh();
    cards.pause();
    answer([job()]);
    await reading;

    expect(shown).toEqual([]);
  });
});

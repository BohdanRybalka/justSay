// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TranscriptionJob } from "../api";

const apiMock = {
  jobs: vi.fn(),
  removeJob: vi.fn(),
  retryJob: vi.fn(),
};

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, api: apiMock };
});

const {
  createJobCards,
  createUploadCard,
  jobIsRunning,
  jobStatusText,
  renderUploadCard,
  uploadStatusText,
  JOBS_POLL_MS,
  REMOVE_REFUSED,
  RETRY_REFUSED,
} = await import("./history-jobs");
const { ApiRequestError } = await import("../api");
type Upload = import("./file-transcription").Upload;
const { icon } = await import("../ui/icons");

function job(overrides: Partial<TranscriptionJob> = {}): TranscriptionJob {
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
let painted: string[];
const entryShown = (id: string) => painted.includes(id);

function status(card: HTMLElement): string {
  return card.querySelector(".entry-job-status")!.textContent!;
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  shown = [];
  painted = [];
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
    const cards = createJobCards(show, entrySaved, entryShown);

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
    const idle = createJobCards(show, entrySaved, entryShown);
    await idle.refresh();
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS * 5);
    expect(apiMock.jobs).toHaveBeenCalledTimes(1);

    apiMock.jobs.mockResolvedValue([job()]);
    const running = createJobCards(show, entrySaved, entryShown);
    await running.refresh();
    running.pause();
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS * 5);
    expect(apiMock.jobs).toHaveBeenCalledTimes(2);
  });

  it("keeps a finished card until its entry is painted, then lets it go", async () => {
    apiMock.jobs.mockResolvedValue([job()]);
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();
    let paintEntry!: () => void;
    entrySaved.mockReturnValue(new Promise<void>((resolve) => (paintEntry = resolve)));

    apiMock.jobs.mockResolvedValue([job({ stage: "done", progress: 1, entry_id: "e1" })]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    expect(entrySaved).toHaveBeenCalledTimes(1);
    expect(shown).toHaveLength(1);

    painted.push("e1");
    paintEntry();
    await vi.advanceTimersByTimeAsync(0);
    expect(shown).toEqual([]);
  });

  it("asks for the entry again while it has not been painted, and keeps the card meanwhile", async () => {
    apiMock.jobs.mockResolvedValue([job()]);
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();

    apiMock.jobs.mockResolvedValue([job({ stage: "done", progress: 1, entry_id: "e1" })]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    expect(entrySaved).toHaveBeenCalledTimes(2);
    expect(shown).toHaveLength(1);

    painted.push("e1");
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    expect(shown).toEqual([]);
  });

  it("reads again a second after a failed read, even before any card is up", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    apiMock.jobs.mockRejectedValueOnce(new Error("the backend did not answer /jobs within 15 seconds"));
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();

    apiMock.jobs.mockResolvedValue([job()]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);

    expect(shown).toHaveLength(1);
  });

  it("keeps reading after a failed read while a card is up", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    apiMock.jobs.mockResolvedValue([job()]);
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();

    apiMock.jobs.mockRejectedValueOnce(new Error("the backend did not answer /jobs within 15 seconds"));
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    apiMock.jobs.mockResolvedValue([job({ stage: "failed", progress: null, error: "Add an API key in Settings" })]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);

    expect(status(shown[0])).toBe("Add an API key in Settings");
  });

  it("does not paint a job that finished while the page was away", async () => {
    apiMock.jobs.mockResolvedValue([job({ stage: "done", progress: 1, entry_id: "e1" }), job({ id: "j2", stage: "cancelled" })]);
    const cards = createJobCards(show, entrySaved, entryShown);

    await cards.refresh();

    expect(shown).toEqual([]);
    expect(entrySaved).not.toHaveBeenCalled();
  });

  it("cancels from the ×, and a later read listing the job as cancelled paints nothing", async () => {
    apiMock.jobs.mockResolvedValue([job()]);
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();
    const button = shown[0].querySelector<HTMLButtonElement>('[data-action="remove"]')!;
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
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();

    shown[0].querySelector<HTMLButtonElement>('[data-action="remove"]')!.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(shown).toHaveLength(1);
    expect(status(shown[0])).toBe(REMOVE_REFUSED);
  });

  it("keeps saying the × was refused while the job stays where it was, and lets go once it moves on", async () => {
    apiMock.jobs.mockResolvedValue([job()]);
    apiMock.removeJob.mockRejectedValue(new TypeError("Failed to fetch"));
    vi.spyOn(console, "error").mockImplementation(() => {});
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();

    shown[0].querySelector<HTMLButtonElement>('[data-action="remove"]')!.click();
    await vi.advanceTimersByTimeAsync(0);
    expect(status(shown[0])).toBe(REMOVE_REFUSED);

    apiMock.jobs.mockResolvedValue([job({ progress: 0.5 })]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    expect(status(shown[0])).toBe(REMOVE_REFUSED);

    apiMock.jobs.mockResolvedValue([job({ stage: "saving", progress: null })]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    expect(status(shown[0])).toBe("Saving");
  });

  it("lets a card go when its × finds the job already gone", async () => {
    apiMock.jobs.mockResolvedValue([job({ stage: "failed", progress: null, error: "Add an API key in Settings" })]);
    apiMock.removeJob.mockRejectedValue(new ApiRequestError("No such job", 404));
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();

    shown[0].querySelector<HTMLButtonElement>('[data-action="remove"]')!.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(shown).toEqual([]);
  });

  it("says on the card that Try again was refused, until it is clicked again", async () => {
    const failed = job({ kind: "meeting", stage: "failed", progress: null, error: "Couldn't turn this meeting into text" });
    apiMock.jobs.mockResolvedValue([failed]);
    apiMock.retryJob.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    vi.spyOn(console, "error").mockImplementation(() => {});
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();
    const retry = shown[0].querySelector<HTMLButtonElement>('[data-action="retry"]')!;

    retry.click();
    await vi.advanceTimersByTimeAsync(0);
    expect(status(shown[0])).toBe(RETRY_REFUSED);
    expect(retry.hidden).toBe(false);

    apiMock.retryJob.mockResolvedValueOnce({ id: "j1" });
    retry.click();
    await vi.advanceTimersByTimeAsync(0);
    expect(status(shown[0])).toBe("Couldn't turn this meeting into text");
  });

  it("shows a failure's reason with a Dismiss that removes it", async () => {
    apiMock.jobs.mockResolvedValue([job({ stage: "failed", progress: null, error: "Add an API key in Settings" })]);
    apiMock.removeJob.mockResolvedValue({ outcome: "dismissed" });
    const cards = createJobCards(show, entrySaved, entryShown);
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

  it("offers a failed meeting Try again, which queues it and reads the jobs again", async () => {
    const failed = job({ kind: "meeting", name: "Meeting · 8 Oct 14:32", stage: "failed", progress: null, error: "Couldn't turn this meeting into text" });
    apiMock.jobs.mockResolvedValue([failed]);
    apiMock.retryJob.mockResolvedValue({ id: "j1" });
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();

    const card = shown[0];
    const retry = card.querySelector<HTMLButtonElement>('[data-action="retry"]')!;
    expect(retry.hidden).toBe(false);
    expect(card.querySelector('[data-action="remove"]')!.getAttribute("aria-label")).toBe("Delete this recording");
    const meetingIcon = document.createElement("span");
    meetingIcon.innerHTML = icon("users", "small");
    expect(card.querySelector(".entry-dot")!.innerHTML).toBe(meetingIcon.innerHTML);

    apiMock.jobs.mockResolvedValue([{ ...failed, stage: "queued", error: null }]);
    retry.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.retryJob).toHaveBeenCalledWith("j1");
    expect(status(shown[0])).toBe("Waiting");
    expect(retry.hidden).toBe(true);
  });

  it("offers a failed file Try again, since its copy is kept", async () => {
    apiMock.jobs.mockResolvedValue([job({ stage: "failed", progress: null, error: "Add an API key in Settings" })]);
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();

    expect(shown[0].querySelector<HTMLButtonElement>('[data-action="retry"]')!.hidden).toBe(false);
  });

  it("offers a failed dictation Try again under a mic, and says Delete for the audio it keeps", async () => {
    const failed = job({ kind: "dictation", name: "Dictation · 8 Oct 14:32", stage: "failed", progress: null, error: "Couldn't turn this dictation into text. Try again" });
    apiMock.jobs.mockResolvedValue([failed]);
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();

    const card = shown[0];
    expect(card.querySelector<HTMLButtonElement>('[data-action="retry"]')!.hidden).toBe(false);
    expect(card.querySelector('[data-action="remove"]')!.getAttribute("aria-label")).toBe("Delete this recording");
    const micIcon = document.createElement("span");
    micIcon.innerHTML = icon("mic", "small");
    expect(card.querySelector(".entry-dot")!.innerHTML).toBe(micIcon.innerHTML);
  });

  it("offers no × on a dictation being turned into text, which cannot be cancelled", async () => {
    apiMock.jobs.mockResolvedValue([job({ kind: "dictation", name: "Dictation · 8 Oct 14:32", stage: "transcribing", progress: null })]);
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();

    expect(shown[0].querySelector<HTMLButtonElement>('[data-action="remove"]')!.hidden).toBe(true);
  });

  it("empties the bar of a meeting queued again after it failed part way", async () => {
    apiMock.jobs.mockResolvedValue([job({ kind: "meeting", progress: 0.5 })]);
    const cards = createJobCards(show, entrySaved, entryShown);
    await cards.refresh();
    const bar = shown[0].querySelector<HTMLElement>(".entry-progress i")!;
    expect(bar.style.width).toBe("50%");

    apiMock.jobs.mockResolvedValue([job({ kind: "meeting", stage: "queued", progress: null })]);
    await cards.refresh();

    expect(bar.style.width).toBe("0px");
  });

  it("drops what a read left in flight when the page paused", async () => {
    let answer!: (jobs: TranscriptionJob[]) => void;
    apiMock.jobs.mockReturnValue(new Promise((resolve) => (answer = resolve)));
    const cards = createJobCards(show, entrySaved, entryShown);

    const reading = cards.refresh();
    cards.pause();
    answer([job()]);
    await reading;

    expect(shown).toEqual([]);
  });
});

describe("jobIsRunning", () => {
  it("is true while a job waits or is being turned into text, and false once it ended", () => {
    expect(["queued", "transcribing", "saving"].map((stage) => jobIsRunning(job({ stage } as Partial<TranscriptionJob>)))).toEqual([
      true,
      true,
      true,
    ]);
    expect(["done", "failed", "cancelled"].map((stage) => jobIsRunning(job({ stage } as Partial<TranscriptionJob>)))).toEqual([
      false,
      false,
      false,
    ]);
  });
});

describe("upload cards", () => {
  function upload(overrides: Partial<Upload> = {}): Upload {
    return { name: "interview.m4a", state: "uploading", sentFraction: 0.45, error: null, retryable: false, ...overrides };
  }

  function drawn(of: Upload) {
    const actions = { remove: vi.fn(), retry: vi.fn() };
    const card = createUploadCard(of, actions);
    renderUploadCard(card, of);
    return { card, actions };
  }

  const retryButton = (card: HTMLElement) => card.querySelector<HTMLButtonElement>('[data-action="retry"]')!;
  const removeButton = (card: HTMLElement) => card.querySelector<HTMLButtonElement>('[data-action="remove"]')!;

  it("says how much has gone, that it waits its turn, or why it failed", () => {
    expect(uploadStatusText(upload({ sentFraction: 0.456 }))).toBe("Uploading · 46%");
    expect(uploadStatusText(upload({ state: "waiting", sentFraction: 0 }))).toBe("Waiting");
    expect(uploadStatusText(upload({ state: "failed", error: "The upload stopped" }))).toBe("The upload stopped");
  });

  it("is drawn as a file job's card, with its bar as full as the upload", () => {
    const { card } = drawn(upload());

    expect(card.classList.contains("entry--job")).toBe(true);
    expect(card.dataset.kind).toBe("file");
    expect(card.querySelector(".entry-job-name")!.textContent).toBe("interview.m4a");
    expect(status(card)).toBe("Uploading · 45%");
    expect(card.querySelector<HTMLElement>(".entry-progress i")!.style.width).toBe("45%");
    expect(retryButton(card).hidden).toBe(true);
    expect(removeButton(card).getAttribute("aria-label")).toBe("Cancel");
  });

  it("cancels the upload from the ×", () => {
    const of = upload();
    const { card, actions } = drawn(of);

    removeButton(card).click();

    expect(actions.remove).toHaveBeenCalledWith(of);
  });

  it("offers Try again on a failed upload, and the × dismisses it", () => {
    const of = upload({ state: "failed", error: "The upload stopped", retryable: true });
    const { card, actions } = drawn(of);

    expect(card.classList.contains("entry--job-failed")).toBe(true);
    expect(retryButton(card).hidden).toBe(false);
    expect(removeButton(card).getAttribute("aria-label")).toBe("Dismiss");
    retryButton(card).click();
    expect(actions.retry).toHaveBeenCalledWith(of);
  });

  it("offers no Try again on a file that was refused", () => {
    const { card } = drawn(upload({ name: "notes.txt", state: "failed", error: "That's not an audio file" }));

    expect(status(card)).toBe("That's not an audio file");
    expect(retryButton(card).hidden).toBe(true);
  });

  it("goes back to an uploading card when Try again sends it", () => {
    const of = { ...upload({ state: "failed", error: "The upload stopped", retryable: true }) };
    const { card } = drawn(of);

    renderUploadCard(card, { ...of, state: "uploading", sentFraction: 0, error: null });

    expect(card.classList.contains("entry--job-failed")).toBe(false);
    expect(status(card)).toBe("Uploading · 0%");
    expect(retryButton(card).hidden).toBe(true);
  });
});

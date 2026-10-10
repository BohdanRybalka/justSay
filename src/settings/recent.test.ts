// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryEntry, TranscriptionJob } from "../api";
import { buildEntry, FakeResizeObserver, pageOf } from "./history-page-stub.test-helper";

const apiMock = {
  getHistory: vi.fn(),
  jobs: vi.fn(),
  removeJob: vi.fn(),
  retryJob: vi.fn(),
  deleteHistoryEntry: vi.fn(),
  setHistoryStarred: vi.fn(),
};

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, api: apiMock };
});

const { mountRecent, RECENT_POLL_MS, RECENT_SHOWN } = await import("./recent");
const { JOBS_POLL_MS } = await import("./history-jobs");

function entryOf(id: string, source: HistoryEntry["source"] = "meeting"): HistoryEntry {
  return { ...buildEntry(id), source };
}

function jobOf(id: string, overrides: Partial<TranscriptionJob> = {}): TranscriptionJob {
  return {
    id,
    kind: "meeting",
    name: `Meeting ${id}`,
    stage: "transcribing",
    progress: 0.4,
    error: null,
    entry_id: null,
    ...overrides,
  };
}

function mount(kind: "meeting" | "file" = "meeting", openHistory = vi.fn()) {
  const container = document.createElement("div");
  document.body.append(container);
  const recent = mountRecent(container, kind, openHistory);
  return { container, recent, openHistory };
}

function shown(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll<HTMLElement>(".timeline > .entry")).map(
    (card) => card.dataset.id ?? `job:${card.dataset.job}`,
  );
}

async function flush(): Promise<void> {
  await vi.advanceTimersByTimeAsync(0);
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  apiMock.getHistory.mockResolvedValue(pageOf([], 0, null));
  apiMock.jobs.mockResolvedValue([]);
});

afterEach(() => {
  document.body.replaceChildren();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("mountRecent — what is listed", () => {
  it("asks for the three newest entries of its own kind, starred or not", async () => {
    mount("file");
    await flush();

    expect(apiMock.getHistory).toHaveBeenCalledWith(RECENT_SHOWN, null, { source: "file", starred: false });
  });

  it("names the section and lists the entries newest first", async () => {
    apiMock.getHistory.mockResolvedValue(pageOf([entryOf("b"), entryOf("a")], 2, null));
    const { container } = mount("meeting");
    await flush();

    expect(container.querySelector(".group-label")!.textContent).toBe("RECENT MEETINGS");
    expect(shown(container)).toEqual(["b", "a"]);
  });

  it("names the files section for files", async () => {
    const { container } = mount("file");
    await flush();

    expect(container.querySelector(".group-label")!.textContent).toBe("RECENT FILES");
  });

  it("puts running and failed jobs of its kind above the entries, three in all", async () => {
    apiMock.getHistory.mockResolvedValue(pageOf([entryOf("c"), entryOf("b"), entryOf("a")], 3, null));
    apiMock.jobs.mockResolvedValue([
      jobOf("j1"),
      jobOf("j2", { stage: "failed", progress: null, error: "No key" }),
      jobOf("other", { kind: "file" }),
    ]);
    const { container } = mount("meeting");
    await flush();

    expect(shown(container)).toEqual(["job:j1", "job:j2", "c"]);
  });

  it("shows no more than three jobs, and then no entries", async () => {
    apiMock.getHistory.mockResolvedValue(pageOf([entryOf("a")], 1, null));
    apiMock.jobs.mockResolvedValue([jobOf("j1"), jobOf("j2"), jobOf("j3"), jobOf("j4")]);
    const { container } = mount("meeting");
    await flush();

    expect(shown(container)).toEqual(["job:j1", "job:j2", "job:j3"]);
  });

  it("says there is nothing yet only once the first read has landed", async () => {
    const { container } = mount("meeting");
    const empty = container.querySelector<HTMLElement>(".history-empty")!;
    expect(empty.hidden).toBe(true);

    await flush();

    expect(empty.hidden).toBe(false);
    expect(empty.textContent).toBe("No meetings yet.");
    expect(container.querySelector<HTMLElement>(".timeline")!.hidden).toBe(true);
  });

  it("keeps quiet about nothing while the first read is out, even when a job of another kind has landed", async () => {
    let answer: (page: ReturnType<typeof pageOf>) => void = () => {};
    apiMock.getHistory.mockReturnValue(new Promise((resolve) => (answer = resolve)));
    apiMock.jobs.mockResolvedValue([jobOf("j1", { kind: "file", name: "a.mp3" })]);
    const { container } = mount("meeting");
    await flush();

    expect(container.querySelector<HTMLElement>(".history-empty")!.hidden).toBe(true);

    answer(pageOf([], 0, null));
    await flush();

    expect(container.querySelector<HTMLElement>(".history-empty")!.hidden).toBe(false);
  });

  it("does not say there is nothing while a job of its kind is running", async () => {
    apiMock.jobs.mockResolvedValue([jobOf("j1", { kind: "file", name: "a.mp3" })]);
    const { container } = mount("file");
    await flush();

    expect(container.querySelector<HTMLElement>(".history-empty")!.hidden).toBe(true);
    expect(shown(container)).toEqual(["job:j1"]);
  });

  it("says no files for an empty files section", async () => {
    const { container } = mount("file");
    await flush();

    expect(container.querySelector(".history-empty")!.textContent).toBe("No files yet.");
  });
});

describe("mountRecent — a job becoming its entry", () => {
  it("turns a finished job's card into its entry where it stood", async () => {
    apiMock.getHistory.mockResolvedValue(pageOf([entryOf("old")], 1, null));
    apiMock.jobs.mockResolvedValue([jobOf("j1")]);
    const { container } = mount("meeting");
    await flush();
    expect(shown(container)).toEqual(["job:j1", "old"]);

    apiMock.getHistory.mockResolvedValue(pageOf([entryOf("saved"), entryOf("old")], 2, null));
    apiMock.jobs.mockResolvedValue([jobOf("j1", { stage: "done", progress: 1, entry_id: "saved" })]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    await flush();

    expect(shown(container)).toEqual(["saved", "old"]);
  });

  it("keeps the card while the entry cannot be read, and lets it go once it can", async () => {
    apiMock.jobs.mockResolvedValue([jobOf("j1")]);
    const { container } = mount("meeting");
    await flush();
    vi.spyOn(console, "error").mockImplementation(() => {});

    apiMock.getHistory.mockRejectedValue(new Error("offline"));
    apiMock.jobs.mockResolvedValue([jobOf("j1", { stage: "done", progress: 1, entry_id: "saved" })]);
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    await flush();
    expect(shown(container)).toEqual(["job:j1"]);

    apiMock.getHistory.mockResolvedValue(pageOf([entryOf("saved")], 1, null));
    await vi.advanceTimersByTimeAsync(JOBS_POLL_MS);
    await flush();

    expect(shown(container)).toEqual(["saved"]);
  });

  it("reads the jobs at once when one is started elsewhere", async () => {
    const { recent, container } = mount("file");
    await flush();
    expect(apiMock.jobs).toHaveBeenCalledTimes(1);

    apiMock.jobs.mockResolvedValue([jobOf("j9", { kind: "file", name: "new.mp3" })]);
    recent.jobStarted();
    await flush();

    expect(apiMock.jobs).toHaveBeenCalledTimes(2);
    expect(shown(container)).toEqual(["job:j9"]);
  });
});

describe("mountRecent — staying current", () => {
  it("picks up a newer entry on the next poll and leaves the cards alone when nothing changed", async () => {
    apiMock.getHistory.mockResolvedValue(pageOf([entryOf("a")], 1, null));
    const { container } = mount("meeting");
    await flush();
    const first = container.querySelector(".timeline > .entry")!;

    await vi.advanceTimersByTimeAsync(RECENT_POLL_MS);
    expect(container.querySelector(".timeline > .entry")).toBe(first);

    apiMock.getHistory.mockResolvedValue(pageOf([entryOf("b"), entryOf("a")], 2, null));
    await vi.advanceTimersByTimeAsync(RECENT_POLL_MS);

    expect(shown(container)).toEqual(["b", "a"]);
  });

  it("stops reading while released and reads again on resume", async () => {
    const { recent } = mount("meeting");
    await flush();
    const reads = apiMock.getHistory.mock.calls.length;

    recent.releaseResources();
    await vi.advanceTimersByTimeAsync(RECENT_POLL_MS * 3);
    expect(apiMock.getHistory).toHaveBeenCalledTimes(reads);

    recent.resumeResources();
    await flush();
    expect(apiMock.getHistory).toHaveBeenCalledTimes(reads + 1);
    await vi.advanceTimersByTimeAsync(RECENT_POLL_MS);
    expect(apiMock.getHistory).toHaveBeenCalledTimes(reads + 2);
  });

  it("stops reading for good once destroyed", async () => {
    const { recent } = mount("meeting");
    await flush();
    const reads = apiMock.getHistory.mock.calls.length;

    recent.destroy();
    await vi.advanceTimersByTimeAsync(RECENT_POLL_MS * 3);

    expect(apiMock.getHistory).toHaveBeenCalledTimes(reads);
  });

  it("drops a deleted entry at once and fills the place from the next read", async () => {
    apiMock.getHistory.mockResolvedValue(pageOf([entryOf("c"), entryOf("b"), entryOf("a")], 4, null));
    apiMock.deleteHistoryEntry.mockResolvedValue(undefined);
    const { container } = mount("meeting");
    await flush();

    let answer: (page: ReturnType<typeof pageOf>) => void = () => {};
    apiMock.getHistory.mockReturnValue(new Promise((resolve) => (answer = resolve)));
    const first = container.querySelector<HTMLElement>(".timeline > .entry")!;
    first.querySelector<HTMLButtonElement>('[data-action="more"]')!.click();
    first.querySelector<HTMLButtonElement>('[data-action="delete"]')!.click();
    await flush();

    expect(apiMock.deleteHistoryEntry).toHaveBeenCalledWith("c");
    expect(shown(container)).toEqual(["b", "a"]);

    answer(pageOf([entryOf("b"), entryOf("a"), entryOf("z")], 3, null));
    await flush();

    expect(shown(container)).toEqual(["b", "a", "z"]);
  });
});

describe("mountRecent — All in History", () => {
  it("opens History with its own kind", async () => {
    const { container, openHistory } = mount("meeting");
    await flush();

    container.querySelector<HTMLButtonElement>(".see-all")!.click();

    expect(openHistory).toHaveBeenCalledExactlyOnceWith("meeting");
  });

  it("opens History with files for the files section", async () => {
    const { container, openHistory } = mount("file");
    await flush();

    container.querySelector<HTMLButtonElement>(".see-all")!.click();

    expect(openHistory).toHaveBeenCalledExactlyOnceWith("file");
  });
});

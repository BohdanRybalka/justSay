// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryEntry, HistoryPageResponse, UserSettings } from "../../api";
import { detectShortcutPlatform, formatAccelerator } from "../../accelerator";
import {
  buildEntry,
  FakeObserver,
  FakeResizeObserver,
  newerByCursor,
  pageOf,
  pagesByCursor,
} from "../history-page-stub.test-helper";

const copyToClipboardMock = vi.fn();

vi.mock("../../clipboard", () => ({
  copyToClipboard: copyToClipboardMock,
}));

const apiMock = {
  getHistory: vi.fn(),
  getNewerHistory: vi.fn(),
  searchHistory: vi.fn(),
  deleteHistoryEntry: vi.fn(),
  setHistoryStarred: vi.fn(),
};

/**
 * Only `api` is replaced. Everything else in the module -- `SidecarTooOldError`
 * above all -- stays the real export, so the `instanceof` branch under test is
 * tied to the class `api.getHistory` actually throws.
 */
vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, api: apiMock };
});

const { SidecarTooOldError, MalformedResponseError } = await import("../../api");
const { sidecarTooOldText } = await import("../history-list");
const { renderHistory, NEWER_POLL_MS } = await import("./history");

const SETTINGS = { shortcut: "Ctrl+Alt+KeyV" } as UserSettings;

/** Local noon on 1 August 2026, the "now" every test runs at. */
const NOW = new Date(2026, 7, 1, 12, 0);

function at(day: number, hour: number, minute = 0): string {
  return new Date(2026, 7, day, hour, minute).toISOString();
}

function mount(container = document.createElement("div")) {
  document.body.append(container);
  return { container, lifecycle: renderHistory(container, SETTINGS) };
}

function countText(container: HTMLElement): string {
  return container.querySelector("#history-count")!.textContent!;
}

function cards(container: HTMLElement): NodeListOf<HTMLElement> {
  return container.querySelectorAll<HTMLElement>(".entry:not(.entry--skeleton)");
}

function sentinel(container: HTMLElement): HTMLElement {
  return container.querySelector<HTMLElement>("#history-more")!;
}

async function renderWith(total: number): Promise<HTMLElement> {
  const entries = Array.from({ length: total }, (_, index) => buildEntry(String(index + 1)));
  apiMock.getHistory.mockResolvedValue(pageOf(entries, total, null));
  const { container } = mount();
  await vi.waitFor(() => {
    expect(countText(container)).toBe(`${total} recording${total !== 1 ? "s" : ""}`);
  });
  return container;
}

async function renderPaged(total: number): Promise<HTMLElement> {
  const all = Array.from({ length: total }, (_, index) => buildEntry(String(index + 1)));
  apiMock.getHistory.mockImplementation(pagesByCursor(all));
  const { container } = mount();
  await vi.waitFor(() => expect(countText(container)).not.toBe("Loading..."));
  return container;
}

async function renderAndWait(entries: HistoryEntry[], total = entries.length, next = null) {
  apiMock.getHistory.mockResolvedValue(pageOf(entries, total, next));
  const mounted = mount();
  await vi.waitFor(() => expect(countText(mounted.container)).not.toBe("Loading..."));
  return mounted;
}

/** jsdom lays nothing out, so a card's text is given the heights a browser would measure. */
function layOut(card: HTMLElement, scrollHeight: number, clientHeight: number): void {
  const text = card.querySelector(".entry-text")!;
  Object.defineProperty(text, "scrollHeight", { configurable: true, value: scrollHeight });
  Object.defineProperty(text, "clientHeight", { configurable: true, value: clientHeight });
}

function cross(): void {
  FakeObserver.latest!.cross();
}

/**
 * The search box's own debounce, restated so the clock can be advanced by
 * exactly it. Ordering is decided by the `deferred()` handles rather than by
 * elapsed time.
 */
const SEARCH_DEBOUNCE_MS = 300;

/** Lets every already-resolved promise settle without any real time passing. */
async function flush(): Promise<void> {
  await vi.advanceTimersByTimeAsync(0);
}

function deferred<T>(): { promise: Promise<T>; release: (value: T) => void } {
  let release: (value: T) => void = () => {};
  const promise = new Promise<T>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

async function typeQuery(container: HTMLElement, value: string): Promise<void> {
  const search = container.querySelector<HTMLInputElement>("#history-search")!;
  search.value = value;
  search.dispatchEvent(new Event("input"));
  await vi.advanceTimersByTimeAsync(SEARCH_DEBOUNCE_MS);
}

function openMenuOf(card: HTMLElement): HTMLButtonElement {
  const more = card.querySelector<HTMLButtonElement>('[data-action="more"]')!;
  more.click();
  return more;
}

function starOf(card: HTMLElement): HTMLButtonElement {
  return card.querySelector<HTMLButtonElement>('[data-action="star"]')!;
}

async function showStarred(container: HTMLElement): Promise<void> {
  const starred = Array.from(container.querySelectorAll<HTMLButtonElement>("#history-filter button")).find(
    (button) => button.textContent === "Starred"
  )!;
  starred.click();
  await flush();
}

async function deleteFirstRow(container: HTMLElement): Promise<void> {
  openMenuOf(cards(container)[0]);
  cards(container)[0].querySelector<HTMLButtonElement>('[data-action="delete"]')!.click();
  await vi.waitFor(() => expect(apiMock.deleteHistoryEntry).toHaveBeenCalledTimes(1));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  vi.setSystemTime(NOW);
  vi.stubGlobal("IntersectionObserver", FakeObserver);
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  apiMock.getNewerHistory.mockResolvedValue(pageOf([], 0, null, { newest_cursor: null }));
});

afterEach(() => {
  document.body.replaceChildren();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("renderHistory — paging as the end scrolls into view", () => {
  it("asks for 30 recordings on the first paint and keeps the sentinel up", async () => {
    const container = await renderPaged(40);

    expect(apiMock.getHistory.mock.calls[0]).toEqual([30, null, false]);
    expect(cards(container)).toHaveLength(30);
    expect(sentinel(container).hidden).toBe(false);
  });

  it("one crossing brings in the rest and hides the sentinel", async () => {
    const container = await renderPaged(40);

    cross();

    await vi.waitFor(() => expect(cards(container)).toHaveLength(40));
    expect(apiMock.getHistory.mock.calls[1]).toEqual([30, { ts: 30, id: "30" }, false]);
    expect(sentinel(container).hidden).toBe(true);
  });

  it("shows three skeleton cards while the first page is read", () => {
    apiMock.getHistory.mockReturnValue(new Promise(() => {}));
    const { container } = mount();

    expect(sentinel(container).classList.contains("timeline-more--reading")).toBe(true);
    expect(container.querySelectorAll(".entry--skeleton")).toHaveLength(3);
  });
});

describe("renderHistory — the timeline", () => {
  it("groups entries under one header per local day, newest day first", async () => {
    const { container } = await renderAndWait([
      { ...buildEntry("a"), timestamp: at(1, 0, 10) },
      { ...buildEntry("b"), timestamp: at(0, 23, 50) },
      { ...buildEntry("c"), timestamp: at(-2, 9) },
    ]);

    const heads = Array.from(container.querySelectorAll(".day-head b")).map((b) => b.textContent);
    expect(heads).toEqual(["Today", "Yesterday", "Wednesday, 29 July"]);
    const groups = container.querySelectorAll(".day-group");
    expect(Array.from(groups[0].querySelectorAll<HTMLElement>(".entry")).map((el) => el.dataset.id)).toEqual(["a"]);
    expect(Array.from(groups[1].querySelectorAll<HTMLElement>(".entry")).map((el) => el.dataset.id)).toEqual(["b"]);
  });

  it("reads each header's totals from the day, not from the cards loaded", async () => {
    apiMock.getHistory.mockResolvedValue(
      pageOf([{ ...buildEntry("a"), timestamp: at(1, 10) }], 40, null, {
        days: [{ date: "2026-08-01", recordings: 4, words: 7142 }],
      })
    );
    const { container } = mount();
    await vi.waitFor(() => expect(cards(container)).toHaveLength(1));

    expect(container.querySelector(".day-head")!.textContent).toBe(
      "Today·4 recordings·7,142 words"
    );
  });

  it("shows the text first and one dim line of time, length and words under it", async () => {
    const { container } = await renderAndWait([
      {
        ...buildEntry("a"),
        timestamp: at(1, 18, 26),
        audio_duration_seconds: 60,
        word_count: 1212,
        language: "en",
        duration_ms: 1234,
      },
    ]);

    const card = cards(container)[0];
    expect(card.firstElementChild!.nextElementSibling!.firstElementChild!.className).toBe("entry-text");
    const meta = card.querySelector(".entry-meta")!;
    expect(meta.textContent!.replace(/\s+/g, " ").trim()).toBe("18:26·1:00·1,212 words");
    expect(card.textContent).not.toContain("process");
    expect(card.textContent).not.toMatch(/\ben\b/);
  });

  it("styles each source apart: a file carries its name, a meeting says so", async () => {
    const { container } = await renderAndWait([
      buildEntry("dictated"),
      { ...buildEntry("file"), source: "file", source_name: "<podcast>.mp3" },
      { ...buildEntry("meeting"), source: "meeting" },
    ]);

    const [dictated, file, meeting] = Array.from(cards(container));
    expect(dictated.classList.contains("entry--dictation")).toBe(true);
    expect(dictated.querySelector(".entry-source")).toBeNull();
    expect(dictated.querySelector(".entry-dot use")!.getAttribute("href")).toBe("#mic");
    expect(file.classList.contains("entry--file")).toBe(true);
    expect(file.querySelector(".entry-source")!.textContent).toBe("<podcast>.mp3");
    expect(file.querySelector(".entry-dot use")!.getAttribute("href")).toBe("#file");
    expect(meeting.classList.contains("entry--meeting")).toBe(true);
    expect(meeting.querySelector(".entry-source")!.textContent).toBe("meeting");
    expect(meeting.querySelector(".entry-dot use")!.getAttribute("href")).toBe("#users");
  });

  it("offers Show more only on a card whose text is cut off", async () => {
    const { container } = await renderAndWait([buildEntry("long"), buildEntry("short")]);
    const [long, short] = Array.from(cards(container));
    layOut(long, 120, 42);
    layOut(short, 21, 21);

    FakeResizeObserver.latest!.resize();

    expect(long.classList.contains("entry--long")).toBe(true);
    expect(long.querySelector(".entry-more")!.textContent).toBe("… more");
    expect(short.classList.contains("entry--long")).toBe(false);
  });

  it("expands a long card from its more, and collapses it from its less or from the text", async () => {
    const { container } = await renderAndWait([buildEntry("a")]);
    const card = cards(container)[0];
    layOut(card, 120, 42);
    FakeResizeObserver.latest!.resize();

    card.querySelector<HTMLElement>(".entry-more")!.click();
    expect(card.classList.contains("entry--expanded")).toBe(true);

    layOut(card, 120, 120);
    FakeResizeObserver.latest!.resize();
    expect(card.classList.contains("entry--long")).toBe(true);

    card.querySelector<HTMLElement>(".entry-less")!.click();
    expect(card.classList.contains("entry--expanded")).toBe(false);

    card.querySelector<HTMLElement>(".entry-text")!.click();
    expect(card.classList.contains("entry--expanded")).toBe(true);
    card.querySelector<HTMLElement>(".entry-text")!.click();
    expect(card.classList.contains("entry--expanded")).toBe(false);
  });

  it("measures every card a search paints, so a cut-off match offers Show more", async () => {
    apiMock.searchHistory.mockResolvedValue({ entries: [buildEntry("9")], total: 1 });
    const { container } = await renderAndWait([buildEntry("1")]);
    await typeQuery(container, "test");
    await vi.waitFor(() => expect(countText(container)).toBe("1 match"));
    const match = cards(container)[0];
    expect(FakeResizeObserver.latest!.watched()).toContain(match.querySelector(".entry-text"));
    layOut(match, 120, 42);

    FakeResizeObserver.latest!.resize();

    expect(match.classList.contains("entry--long")).toBe(true);
  });

  it("lets go of a card's text once it has left the page, and of everything on teardown", async () => {
    apiMock.searchHistory.mockResolvedValue({ entries: [buildEntry("9")], total: 1 });
    const { container, lifecycle } = await renderAndWait([buildEntry("1")]);
    const first = cards(container)[0].querySelector(".entry-text")!;
    await typeQuery(container, "test");
    await vi.waitFor(() => expect(countText(container)).toBe("1 match"));

    FakeResizeObserver.latest!.resize();
    expect(FakeResizeObserver.latest!.watched()).not.toContain(first);

    lifecycle.destroy();
    expect(FakeResizeObserver.latest!.watched()).toHaveLength(0);
  });

  it("leaves a short card alone when its text is clicked", async () => {
    const { container } = await renderAndWait([buildEntry("a")]);
    const card = cards(container)[0];
    layOut(card, 21, 21);
    FakeResizeObserver.latest!.resize();

    card.querySelector<HTMLElement>(".entry-text")!.click();

    expect(card.classList.contains("entry--expanded")).toBe(false);
  });

  it("marks matched words with the backend's own markup, never escaped again", async () => {
    apiMock.searchHistory.mockResolvedValue({
      entries: [{ ...buildEntry("9"), highlighted_text: "a <mark>test</mark> &amp; more" }],
      total: 1,
    });
    const { container } = await renderAndWait([buildEntry("1")]);

    await typeQuery(container, "test");
    await vi.waitFor(() => expect(countText(container)).toBe("1 match"));

    const text = cards(container)[0].querySelector(".entry-text")!;
    expect(text.querySelector("mark")!.textContent).toBe("test");
    expect(text.firstChild!.textContent).toBe("a ");
    expect(text.textContent).toBe("a test & moreless");
  });

  it("says how to start when there is nothing yet", async () => {
    const { container } = await renderAndWait([]);

    const shortcut = formatAccelerator(SETTINGS.shortcut, detectShortcutPlatform(navigator));
    expect(container.querySelector(".history-empty")!.textContent).toBe(
      `Nothing here yet. Hold ${shortcut} anywhere and talk.`
    );
  });
});

describe("renderHistory — new recordings appear while it is open", () => {
  it("puts a recording made while the window is open at the top of Today within one poll", async () => {
    const store = [{ ...buildEntry("old"), timestamp: at(1, 9) }];
    apiMock.getHistory.mockResolvedValue(pageOf([...store], 1, null));
    apiMock.getNewerHistory.mockImplementation(newerByCursor(store));
    const { container } = mount();
    await vi.waitFor(() => expect(cards(container)).toHaveLength(1));

    store.unshift({ ...buildEntry("new"), timestamp: at(1, 11) });
    await vi.advanceTimersByTimeAsync(NEWER_POLL_MS);

    expect(Array.from(cards(container)).map((el) => el.dataset.id)).toEqual(["new", "old"]);
    expect(countText(container)).toBe("2 recordings");
  });

  it("stops asking while the window is hidden and asks at once when it is shown", async () => {
    const { lifecycle } = await renderAndWait([buildEntry("a")]);
    lifecycle.releaseResources!();

    await vi.advanceTimersByTimeAsync(NEWER_POLL_MS * 3);
    expect(apiMock.getNewerHistory).not.toHaveBeenCalled();

    lifecycle.resumeResources!();
    await flush();
    expect(apiMock.getNewerHistory).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(NEWER_POLL_MS);
    expect(apiMock.getNewerHistory).toHaveBeenCalledTimes(2);
  });

  it("stops asking once the panel is gone", async () => {
    const { lifecycle } = await renderAndWait([buildEntry("a")]);
    lifecycle.destroy();

    await vi.advanceTimersByTimeAsync(NEWER_POLL_MS * 3);

    expect(apiMock.getNewerHistory).not.toHaveBeenCalled();
  });
});

describe("renderHistory — Copy", () => {
  it("hands the transcript to the clipboard command and flashes a check", async () => {
    copyToClipboardMock.mockResolvedValue(true);
    const container = await renderWith(1);

    container.querySelector<HTMLButtonElement>('[data-action="copy"]')!.click();

    await vi.waitFor(() => {
      expect(container.querySelector('[data-action="copy"]')!.getAttribute("aria-label")).toBe(
        "Copied"
      );
    });
    expect(container.querySelector('[data-action="copy"] use')!.getAttribute("href")).toBe("#check");
    expect(copyToClipboardMock).toHaveBeenCalledExactlyOnceWith(buildEntry("1").text);
  });

  it("says Copy failed when the clipboard command fails", async () => {
    copyToClipboardMock.mockResolvedValue(false);
    const container = await renderWith(1);

    container.querySelector<HTMLButtonElement>('[data-action="copy"]')!.click();

    await vi.waitFor(() => {
      expect(container.querySelector('[data-action="copy"]')!.getAttribute("aria-label")).toBe(
        "Copy failed"
      );
    });
  });
});

describe("renderHistory — Star", () => {
  it("stars an entry and unstars it again, filling the star while it is on", async () => {
    apiMock.setHistoryStarred.mockImplementation(async (_id: string, starred: boolean) => ({ starred }));
    const container = await renderWith(1);
    const star = starOf(cards(container)[0]);

    star.click();
    await flush();
    expect(apiMock.setHistoryStarred).toHaveBeenLastCalledWith("1", true);
    expect(star.getAttribute("aria-pressed")).toBe("true");
    expect(star.classList.contains("entry-star--on")).toBe(true);

    star.click();
    await flush();
    expect(apiMock.setHistoryStarred).toHaveBeenLastCalledWith("1", false);
    expect(star.getAttribute("aria-pressed")).toBe("false");
  });

  it("shows an entry that is already starred as starred", async () => {
    const { container } = await renderAndWait([{ ...buildEntry("a"), starred: true }]);

    expect(starOf(cards(container)[0]).getAttribute("aria-pressed")).toBe("true");
  });

  it("puts the star back when saving it failed", async () => {
    let fail: (error: Error) => void = () => {};
    apiMock.setHistoryStarred.mockReturnValue(
      new Promise((_resolve, reject) => {
        fail = reject;
      })
    );
    const container = await renderWith(1);
    const star = starOf(cards(container)[0]);

    star.click();
    await flush();
    expect(star.getAttribute("aria-pressed")).toBe("true");

    fail(new Error("offline"));
    await flush();
    expect(star.getAttribute("aria-pressed")).toBe("false");
    expect(star.classList.contains("entry-star--on")).toBe(false);
  });
});

describe("renderHistory — All / Starred", () => {
  it("reloads with starred entries only, and says so when there are none", async () => {
    const container = await renderWith(2);
    apiMock.getHistory.mockResolvedValue(pageOf([], 0, null));

    await showStarred(container);

    expect(apiMock.getHistory).toHaveBeenLastCalledWith(30, null, true);
    await vi.waitFor(() =>
      expect(container.querySelector(".history-empty")!.textContent).toBe("Nothing starred yet.")
    );
  });

  it("keeps a search to starred entries once Starred is on", async () => {
    const container = await renderWith(1);
    apiMock.searchHistory.mockResolvedValue({ entries: [], total: 0 });

    await typeQuery(container, "budget");
    expect(apiMock.searchHistory).toHaveBeenLastCalledWith("budget", 30, false);

    await showStarred(container);
    expect(apiMock.searchHistory).toHaveBeenLastCalledWith("budget", 30, true);
  });

  it("asks the poll for starred entries only while Starred is on", async () => {
    const container = await renderWith(1);
    apiMock.getHistory.mockResolvedValue(pageOf([buildEntry("s")], 1, null));
    await showStarred(container);

    await vi.advanceTimersByTimeAsync(NEWER_POLL_MS);

    expect(apiMock.getNewerHistory).toHaveBeenLastCalledWith(30, expect.anything(), true);
  });
});

describe("renderHistory — the more menu", () => {
  it("opens under More with Delete in it, and Escape closes it back onto More", async () => {
    const container = await renderWith(1);
    const more = openMenuOf(cards(container)[0]);

    expect(more.getAttribute("aria-expanded")).toBe("true");
    expect(document.activeElement!.textContent).toBe("Delete");

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));

    expect(container.querySelector(".entry-menu")).toBeNull();
    expect(more.getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(more);
  });

  it("closes on a click anywhere outside it", async () => {
    const container = await renderWith(1);
    openMenuOf(cards(container)[0]);

    document.body.dispatchEvent(new Event("pointerdown", { bubbles: true }));

    expect(container.querySelector(".entry-menu")).toBeNull();
  });

  it("keeps one menu open at a time", async () => {
    const container = await renderWith(2);
    openMenuOf(cards(container)[0]);
    openMenuOf(cards(container)[1]);

    expect(container.querySelectorAll(".entry-menu")).toHaveLength(1);
    expect(cards(container)[1].querySelector(".entry-menu")).not.toBeNull();
  });
});

describe("renderHistory — the count names recordings", () => {
  it("reads '1 recording' for a single entry", async () => {
    const container = await renderWith(1);
    expect(countText(container)).toBe("1 recording");
  });

  it("reads '2 recordings' for two entries", async () => {
    const container = await renderWith(2);
    expect(countText(container)).toBe("2 recordings");
  });
});

describe("renderHistory — teardown", () => {
  it("a response arriving after teardown writes nothing", async () => {
    const page = deferred<HistoryPageResponse>();
    apiMock.getHistory.mockReturnValue(page.promise);
    const { container, lifecycle } = mount();

    const before = countText(container);
    lifecycle.destroy();
    page.release(pageOf([buildEntry("1")], 1, null));
    await flush();

    expect(countText(container)).toBe(before);
    expect(cards(container)).toHaveLength(0);
  });

  it("a search resolving after teardown writes nothing", async () => {
    const search = deferred<{ entries: HistoryEntry[]; total: number }>();
    apiMock.searchHistory.mockReturnValue(search.promise);
    const { container, lifecycle } = await renderAndWait([buildEntry("1"), buildEntry("2")]);

    await typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));

    const countBefore = countText(container);
    lifecycle.destroy();
    search.release({ entries: [buildEntry("9")], total: 1 });
    await flush();

    expect(countText(container)).toBe(countBefore);
    expect(cards(container)).toHaveLength(2);
  });
});

describe("renderHistory — a reload and a search cannot both own the rows", () => {
  it("a search that answers first keeps its matches when the reload arrives after it", async () => {
    const entries = [buildEntry("1"), buildEntry("2")];
    const reload = deferred<HistoryPageResponse>();
    let pages = 0;
    apiMock.getHistory.mockImplementation(async () => {
      pages += 1;
      return pages === 1 ? pageOf(entries, 2, null) : reload.promise;
    });
    const search = deferred<{ entries: HistoryEntry[]; total: number }>();
    apiMock.searchHistory.mockReturnValue(search.promise);
    apiMock.deleteHistoryEntry.mockResolvedValue({ deleted: true });

    const { container } = mount();
    await vi.waitFor(() => expect(countText(container)).toBe("2 recordings"));

    await typeQuery(container, "");
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    await typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));

    search.release({ entries: [buildEntry("9")], total: 1 });
    await vi.waitFor(() => expect(countText(container)).toBe("1 match"));

    reload.release(pageOf(entries, 2, { ts: 1, id: "1" }));
    await flush();

    expect(countText(container)).toBe("1 match");
    expect(cards(container)).toHaveLength(1);
    expect(sentinel(container).hidden).toBe(true);

    await deleteFirstRow(container);
    expect(countText(container)).toBe("1 match");
  });

  it("a search that failed without repainting leaves the delete moving the total", async () => {
    apiMock.searchHistory.mockRejectedValue(new Error("503 store busy"));
    apiMock.deleteHistoryEntry.mockResolvedValue({ deleted: true });
    const { container } = await renderAndWait([buildEntry("1"), buildEntry("2")]);

    await typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));
    await flush();

    expect(cards(container)).toHaveLength(2);

    await deleteFirstRow(container);

    expect(countText(container)).toBe("1 recording");
    expect(cards(container)).toHaveLength(1);
  });

  it("emptying the box is what gives paging back after a search that never answers", async () => {
    const entries = Array.from({ length: 30 }, (_, index) => buildEntry(String(index + 1)));
    apiMock.getHistory.mockResolvedValue(pageOf(entries, 60, { ts: 30, id: "30" }));
    apiMock.searchHistory.mockReturnValue(new Promise(() => {}));
    const { container } = mount();
    await vi.waitFor(() => expect(countText(container)).toBe("60 recordings"));

    await typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));
    expect(container.querySelector("#history-search-hint")!.textContent).toBe("Searching...");
    cross();
    await flush();
    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);

    await typeQuery(container, "");
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    await flush();

    expect(container.querySelector("#history-search-hint")!.textContent).toBe("");
    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(3));
  });

  it("a reload that takes the rows over drops the search answer that arrives after it", async () => {
    const entries = [buildEntry("1"), buildEntry("2")];
    const reload = deferred<HistoryPageResponse>();
    let pages = 0;
    apiMock.getHistory.mockImplementation(async () => {
      pages += 1;
      return pages === 1 ? pageOf(entries, 2, null) : reload.promise;
    });
    const search = deferred<{ entries: HistoryEntry[]; total: number }>();
    apiMock.searchHistory.mockReturnValue(search.promise);
    apiMock.deleteHistoryEntry.mockResolvedValue({ deleted: true });

    const { container } = mount();
    await vi.waitFor(() => expect(countText(container)).toBe("2 recordings"));

    await typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));
    await typeQuery(container, "");
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));

    reload.release(pageOf(entries, 2, null));
    await vi.waitFor(() => expect(cards(container)).toHaveLength(2));

    search.release({ entries: [buildEntry("9")], total: 1 });
    await flush();

    expect(countText(container)).toBe("2 recordings");
    expect(cards(container)).toHaveLength(2);

    await deleteFirstRow(container);
    expect(countText(container)).toBe("1 recording");
  });

  it("a search that answers over an append in flight keeps the append from painting or paging", async () => {
    const entries = Array.from({ length: 30 }, (_, index) => buildEntry(String(index + 1)));
    const append = deferred<HistoryPageResponse>();
    let pages = 0;
    apiMock.getHistory.mockImplementation(async () => {
      pages += 1;
      return pages === 1 ? pageOf(entries, 60, { ts: 30, id: "30" }) : append.promise;
    });
    const search = deferred<{ entries: HistoryEntry[]; total: number }>();
    apiMock.searchHistory.mockReturnValue(search.promise);

    const { container } = mount();
    await vi.waitFor(() => expect(countText(container)).toBe("60 recordings"));
    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));

    await typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));
    search.release({ entries: [], total: 0 });
    append.release(pageOf([buildEntry("31")], 60, null));
    await flush();

    expect(container.querySelector(".history-empty")!.textContent).toBe("No matches");
    expect(sentinel(container).hidden).toBe(true);
    cross();
    await flush();
    expect(apiMock.getHistory).toHaveBeenCalledTimes(2);
  });

  it("the poll leaves search results alone", async () => {
    apiMock.searchHistory.mockResolvedValue({ entries: [buildEntry("9")], total: 1 });
    const { container } = await renderAndWait([buildEntry("1")]);

    await typeQuery(container, "hello");
    await vi.waitFor(() => expect(countText(container)).toBe("1 match"));
    await vi.advanceTimersByTimeAsync(NEWER_POLL_MS * 2);

    expect(apiMock.getNewerHistory).not.toHaveBeenCalled();
    expect(Array.from(cards(container)).map((el) => el.dataset.id)).toEqual(["9"]);
  });
});

describe("renderHistory — paging while a search owns the rows", () => {
  it("refuses a crossing while the search is outstanding, without discarding the search", async () => {
    const entries = Array.from({ length: 30 }, (_, index) => buildEntry(String(index + 1)));
    apiMock.getHistory.mockResolvedValue(pageOf(entries, 60, { ts: 30, id: "30" }));
    const search = deferred<{ entries: HistoryEntry[]; total: number }>();
    apiMock.searchHistory.mockReturnValue(search.promise);
    const { container } = mount();
    await vi.waitFor(() => expect(countText(container)).toBe("60 recordings"));

    await typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));
    cross();
    await flush();
    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);

    search.release({ entries: [buildEntry("9")], total: 1 });
    await vi.waitFor(() => expect(countText(container)).toBe("1 match"));
    expect(sentinel(container).hidden).toBe(true);
  });

  it("keeps paging when the search fails over rows it never repainted", async () => {
    apiMock.getHistory.mockImplementation(
      pagesByCursor(Array.from({ length: 60 }, (_, index) => buildEntry(String(index + 1))))
    );
    apiMock.searchHistory.mockRejectedValue(new Error("503 store busy"));
    const { container } = mount();
    await vi.waitFor(() => expect(countText(container)).toBe("60 recordings"));

    await typeQuery(container, "hello");
    await vi.waitFor(() => {
      expect(container.querySelector("#history-search-hint")!.textContent).toBe("503 store busy");
    });

    expect(sentinel(container).hidden).toBe(false);
    cross();
    await vi.waitFor(() => expect(cards(container)).toHaveLength(60));
  });
});

describe("renderHistory — the search hint belongs to the lane that put it up", () => {
  it("keeps a failing search's own message, which the lane still owns", async () => {
    apiMock.searchHistory.mockRejectedValue(new Error("503 store busy"));
    const { container } = await renderAndWait([buildEntry("1"), buildEntry("2")]);

    await typeQuery(container, "hello");
    await vi.waitFor(() => {
      expect(container.querySelector("#history-search-hint")!.textContent).toBe("503 store busy");
    });
    await flush();

    expect(container.querySelector("#history-search-hint")!.textContent).toBe("503 store busy");
  });

  it("leaves a newer search's error message alone when the one it superseded settles", async () => {
    const superseded = deferred<{ entries: HistoryEntry[]; total: number }>();
    let searches = 0;
    apiMock.searchHistory.mockImplementation(() => {
      searches += 1;
      return searches === 1 ? superseded.promise : Promise.reject(new Error("invalid query"));
    });
    const { container } = await renderAndWait([buildEntry("1"), buildEntry("2")]);

    await typeQuery(container, "first");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));
    await typeQuery(container, "second");
    await vi.waitFor(() => {
      expect(container.querySelector("#history-search-hint")!.textContent).toBe(
        "Invalid search query"
      );
    });

    superseded.release({ entries: [buildEntry("9")], total: 1 });
    await flush();

    expect(container.querySelector("#history-search-hint")!.textContent).toBe(
      "Invalid search query"
    );
  });

  it("leaves a newer search's Searching... alone when the one it superseded settles first", async () => {
    const superseded = deferred<{ entries: HistoryEntry[]; total: number }>();
    const inFlight = deferred<{ entries: HistoryEntry[]; total: number }>();
    let searches = 0;
    apiMock.searchHistory.mockImplementation(() => {
      searches += 1;
      return searches === 1 ? superseded.promise : inFlight.promise;
    });
    const { container } = await renderAndWait([buildEntry("1"), buildEntry("2")]);

    await typeQuery(container, "first");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));
    await typeQuery(container, "second");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(2));

    superseded.release({ entries: [buildEntry("9")], total: 1 });
    await flush();

    expect(container.querySelector("#history-search-hint")!.textContent).toBe("Searching...");
  });

  it("names the version skew from the error class rather than from the words in its message", async () => {
    apiMock.searchHistory.mockRejectedValue(
      new SidecarTooOldError("/history/search answered HTTP 404")
    );
    const { container } = await renderAndWait([buildEntry("1"), buildEntry("2")]);

    await typeQuery(container, "hello");

    await vi.waitFor(() => {
      expect(container.querySelector("#history-search-hint")!.textContent).toBe(
        sidecarTooOldText("Search")
      );
    });
  });

  it("keeps the endpoint out of the hint when the search reply itself was malformed", async () => {
    apiMock.searchHistory.mockRejectedValue(
      new MalformedResponseError("/history/search returned a total that is not a number")
    );
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const { container } = await renderAndWait([buildEntry("1"), buildEntry("2")]);

    await typeQuery(container, "hello");

    await vi.waitFor(() => {
      expect(container.querySelector("#history-search-hint")!.textContent).toBe("Search failed");
    });
    expect(countText(container)).toBe("2 recordings");
    expect(container.textContent).not.toContain("undefined");
    expect(container.textContent).not.toContain("/history");
    expect(logged).toHaveBeenCalled();
    logged.mockRestore();
  });

  it("reports an ordinary failure verbatim even when it happens to say 'not found'", async () => {
    apiMock.searchHistory.mockRejectedValue(new Error("transcript not found"));
    const { container } = await renderAndWait([buildEntry("1"), buildEntry("2")]);

    await typeQuery(container, "hello");

    await vi.waitFor(() => {
      expect(container.querySelector("#history-search-hint")!.textContent).toBe(
        "transcript not found"
      );
    });
  });
});

describe("renderHistory — the version-skew message names this tab", () => {
  it("says History, the name this tab hands the shared list", async () => {
    apiMock.getHistory.mockRejectedValue(new SidecarTooOldError("no next_cursor"));
    const { container } = mount();

    await vi.waitFor(() => {
      expect(countText(container)).toBe("History needs the latest backend — please update JustSay.");
    });
  });
});

describe("renderHistory — a delete", () => {
  it("decrements the total after an ordinary load", async () => {
    apiMock.deleteHistoryEntry.mockResolvedValue({ deleted: true });
    const container = await renderWith(2);

    await deleteFirstRow(container);

    expect(countText(container)).toBe("1 recording");
  });

  it("takes the deleted card out of its day header's totals", async () => {
    apiMock.deleteHistoryEntry.mockResolvedValue({ deleted: true });
    apiMock.getHistory.mockResolvedValue(
      pageOf(
        [
          { ...buildEntry("a"), timestamp: at(1, 10), word_count: 7 },
          { ...buildEntry("b"), timestamp: at(1, 9), word_count: 5 },
        ],
        2,
        null,
        { days: [{ date: "2026-08-01", recordings: 2, words: 12 }] }
      )
    );
    const { container } = mount();
    await vi.waitFor(() => expect(cards(container)).toHaveLength(2));

    await deleteFirstRow(container);
    await flush();

    expect(container.querySelector(".day-head")!.textContent).toBe("Today·1 recording·5 words");
    expect(container.querySelector(".entry-menu")).toBeNull();
  });

  it("takes an emptied day off the timeline", async () => {
    apiMock.deleteHistoryEntry.mockResolvedValue({ deleted: true });
    const { container } = await renderAndWait([
      { ...buildEntry("today"), timestamp: at(1, 9) },
      { ...buildEntry("yesterday"), timestamp: at(0, 9) },
    ]);

    await deleteFirstRow(container);
    await flush();

    const heads = Array.from(container.querySelectorAll(".day-head b")).map((b) => b.textContent);
    expect(heads).toEqual(["Yesterday"]);
  });
});

describe("renderHistory — a record whose recording time could not be recovered", () => {
  it("sits under its own header with a dash where the time goes and the transcript in full", async () => {
    const dated = { ...buildEntry("dated"), timestamp: at(1, 9) };
    const undated: HistoryEntry = { ...buildEntry("undated"), timestamp: null };
    const { container } = await renderAndWait([dated, undated]);

    const heads = Array.from(container.querySelectorAll(".day-head b")).map((b) => b.textContent);
    expect(heads).toEqual(["Today", "Date unknown"]);
    const undatedRow = cards(container)[1];
    expect(undatedRow.querySelector(".entry-meta span")!.textContent).toBe("—");
    expect(undatedRow.textContent).toContain("transcript undated");
    expect(undatedRow.textContent).not.toContain("Invalid Date");
    expect(undatedRow.textContent).not.toContain("1970");
  });
});

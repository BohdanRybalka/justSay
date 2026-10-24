// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import { buildEntry, FakeResizeObserver } from "./history-page-stub.test-helper";

const { deleteHistoryEntryMock } = vi.hoisted(() => ({ deleteHistoryEntryMock: vi.fn() }));
vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  api: { deleteHistoryEntry: deleteHistoryEntryMock },
}));

const { ApiRequestError } = await import("../api");
const { createEntryCards, DELETE_REFUSED, matchExcerpt } = await import("./history-entry");

describe("matchExcerpt — the text around the first match", () => {
  it("starts a few whole words before a match deep in the text, keeping the backend's markup", () => {
    const excerpt = matchExcerpt(`${"alpha ".repeat(20)}R&amp;D <mark>test</mark> &lt;b&gt;`)!;

    expect(excerpt).toMatch(/^…alpha /);
    expect(excerpt).toContain("R&amp;D <mark>test</mark> &lt;b&gt;");
    expect(excerpt.match(/alpha/g)!.length).toBeLessThan(20);
  });

  it("leaves a text alone when its first match is near the start or there is none", () => {
    expect(matchExcerpt("a <mark>test</mark> here")).toBeNull();
    expect(matchExcerpt("<mark>test</mark> here")).toBeNull();
    expect(matchExcerpt("no mark ".repeat(20))).toBeNull();
  });
});

describe("Delete on a card", () => {
  const deleted = vi.fn();

  function pickDelete(card: HTMLElement): void {
    card.querySelector<HTMLButtonElement>('[data-action="more"]')!.click();
    card.querySelector<HTMLButtonElement>('[data-action="delete"]')!.click();
  }

  function refusal(card: HTMLElement): string | null {
    return card.querySelector(".entry-refusal")?.textContent ?? null;
  }

  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("says on the card that the backend refused it, and keeps the card", async () => {
    deleteHistoryEntryMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const card = createEntryCards(deleted).create(buildEntry("1"));
    document.body.append(card);

    pickDelete(card);

    await vi.waitFor(() => expect(refusal(card)).toBe(DELETE_REFUSED));
    expect(card.querySelector(".entry-refusal")!.getAttribute("role")).toBe("alert");
    expect(deleted).not.toHaveBeenCalled();
  });

  it("takes the message away when Delete is picked again, and says it once if refused again", async () => {
    deleteHistoryEntryMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const card = createEntryCards(deleted).create(buildEntry("1"));
    document.body.append(card);
    pickDelete(card);
    await vi.waitFor(() => expect(refusal(card)).toBe(DELETE_REFUSED));

    let answer!: () => void;
    deleteHistoryEntryMock.mockReturnValue(new Promise<void>((_, reject) => (answer = () => reject(new TypeError("Failed to fetch")))));
    pickDelete(card);
    expect(refusal(card)).toBeNull();

    answer();
    await vi.waitFor(() => expect(refusal(card)).toBe(DELETE_REFUSED));
    expect(card.querySelectorAll(".entry-refusal")).toHaveLength(1);
  });

  it("sends no second Delete while the first is on its way", async () => {
    let answer!: () => void;
    deleteHistoryEntryMock.mockReturnValueOnce(new Promise<void>((resolve) => (answer = resolve)));
    deleteHistoryEntryMock.mockRejectedValue(new ApiRequestError("Entry not found", 404));
    const card = createEntryCards(deleted).create(buildEntry("1"));
    document.body.append(card);

    pickDelete(card);
    pickDelete(card);
    answer();

    await vi.waitFor(() => expect(deleted).toHaveBeenCalledOnce());
    expect(deleteHistoryEntryMock).toHaveBeenCalledOnce();
  });

  it("lets the card go with no message when the entry is already gone", async () => {
    deleteHistoryEntryMock.mockRejectedValue(new ApiRequestError("Entry not found", 404));
    const card = createEntryCards(deleted).create(buildEntry("1"));
    document.body.append(card);

    pickDelete(card);

    await vi.waitFor(() => expect(deleted).toHaveBeenCalledWith(card));
    expect(refusal(card)).toBeNull();
  });
});

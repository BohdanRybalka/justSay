// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EVENT_FILE_PICKED, MAX_UPLOAD_BYTES } from "../contracts";

const apiMock = {
  startFileJob: vi.fn(),
};

vi.mock("../api", () => ({
  api: apiMock,
}));

const { invokeMock, shellListeners, stopHearing } = vi.hoisted(() => ({
  invokeMock: vi.fn(),
  shellListeners: new Map<string, (event: { payload: unknown }) => unknown>(),
  stopHearing: vi.fn(),
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (event: string, handler: (event: { payload: unknown }) => unknown) => {
    shellListeners.set(event, handler);
    return stopHearing;
  }),
}));

const { mountFileTranscription, refusalOf, DROP_AREA_CLASS, DROP_HINT, REFUSAL_SHOWN_MS } = await import("./file-transcription");

const onStarted = vi.fn();
let mounted: { pick: () => void; destroy: () => void } = { pick: () => {}, destroy: () => {} };

function buildFile(name: string, size: number): File {
  const file = new File(["audio-bytes"], name, { type: "audio/wav" });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

function drag(type: string, target: EventTarget, carried: string[], files: File[] = []): Event {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(event, "dataTransfer", { value: { types: carried, files } });
  target.dispatchEvent(event);
  return event;
}

function picker(): HTMLInputElement {
  return document.querySelector<HTMLInputElement>('input[type="file"]')!;
}

function pick(file: File): void {
  Object.defineProperty(picker(), "files", { value: [file], configurable: true });
  picker().dispatchEvent(new Event("change"));
}

function overlay(): HTMLElement {
  return document.querySelector<HTMLElement>(".drop-overlay")!;
}

function shownText(): string {
  return overlay().querySelector(".drop-overlay-title")!.textContent ?? "";
}

beforeEach(() => {
  vi.clearAllMocks();
  apiMock.startFileJob.mockResolvedValue({ id: "job-1" });
  document.body.innerHTML = `<main id="pane"><div id="child"><span id="grandchild"></span></div></main>`;
  mounted = mountFileTranscription(document.body, onStarted);
});

afterEach(() => {
  mounted.destroy();
  vi.useRealTimers();
});

describe("refusalOf — why a file cannot be transcribed", () => {
  it("says a file of another kind is not audio", () => {
    expect(refusalOf(buildFile("notes.txt", 2048))).toBe("That's not an audio file");
    expect(refusalOf(buildFile("README", 2048))).toBe("That's not an audio file");
  });

  it("says an empty file is empty", () => {
    expect(refusalOf(buildFile("silence.wav", 0))).toBe("This file is empty");
  });

  it("names the ceiling a file is over", () => {
    expect(refusalOf(buildFile("long.mp3", MAX_UPLOAD_BYTES + 1))).toBe("This file is over 500 MB");
    expect(refusalOf(buildFile("edge.mp3", MAX_UPLOAD_BYTES))).toBeNull();
  });

  it("takes an audio file whatever the case of its extension", () => {
    expect(refusalOf(buildFile("Call.M4A", 2048))).toBeNull();
  });
});

describe("the overlay while a file is dragged over the window", () => {
  it("is hidden until a drag arrives", () => {
    expect(overlay().hidden).toBe(true);
  });

  it("dims the window under the drop hint when a file is dragged anywhere on it", () => {
    const enter = drag("dragenter", document.getElementById("grandchild")!, ["Files"]);

    expect(overlay().hidden).toBe(false);
    expect(shownText()).toBe("Drop to transcribe");
    expect(overlay().querySelector(".drop-overlay-hint")!.textContent).toBe(DROP_HINT);
    expect(DROP_HINT).toBe("mp3, wav, m4a, mp4 and 7 more · up to 500 MB");
    expect(enter.defaultPrevented, "the webview refuses a drag whose enter is not cancelled").toBe(true);
  });

  it("stays lit while the drag crosses from one element onto another", () => {
    const pane = document.getElementById("pane")!;
    const child = document.getElementById("child")!;

    drag("dragenter", pane, ["Files"]);
    drag("dragenter", child, ["Files"]);
    drag("dragleave", pane, ["Files"]);

    expect(overlay().hidden, "a leave fires at every element boundary; one leave must not hide it").toBe(false);

    drag("dragleave", child, ["Files"]);

    expect(overlay().hidden).toBe(true);
  });

  it("comes back when the drag leaves the window and enters again", () => {
    drag("dragenter", document.body, ["Files"]);
    drag("dragleave", document.body, ["Files"]);
    drag("dragleave", document.body, ["Files"]);
    expect(overlay().hidden).toBe(true);

    drag("dragenter", document.body, ["Files"]);

    expect(overlay().hidden, "an unmatched leave must not push the count below zero").toBe(false);
  });

  it("cancels dragover for a file, so the drop lands in the page", () => {
    drag("dragenter", document.body, ["Files"]);

    expect(drag("dragover", document.body, ["Files"]).defaultPrevented).toBe(true);
  });
});

describe("a drag that carries no file", () => {
  it("shows nothing and is left to the page, so a dragged path still lands in a field", () => {
    const enter = drag("dragenter", document.body, ["text/plain"]);
    const over = drag("dragover", document.body, ["text/plain"]);
    const drop = drag("drop", document.body, ["text/plain"]);

    expect(overlay().hidden).toBe(true);
    expect(enter.defaultPrevented).toBe(false);
    expect(over.defaultPrevented).toBe(false);
    expect(drop.defaultPrevented).toBe(false);
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });
});

describe("dropping a file", () => {
  it("sends it off as a job, hides the overlay and reports the start", async () => {
    drag("dragenter", document.body, ["Files"]);
    const drop = drag("drop", document.getElementById("child")!, ["Files"], [buildFile("note.wav", 2048)]);

    expect(drop.defaultPrevented, "an unprevented drop navigates the webview to the file").toBe(true);
    expect(overlay().hidden).toBe(true);
    await vi.waitFor(() => expect(onStarted).toHaveBeenCalledOnce());
    expect(apiMock.startFileJob.mock.calls[0][1]).toBe("note.wav");
  });

  it("shows why a refused file was refused, without sending it, then goes away", async () => {
    vi.useFakeTimers();
    drag("dragenter", document.body, ["Files"]);
    drag("drop", document.body, ["Files"], [buildFile("notes.txt", 2048)]);

    await vi.waitFor(() => expect(overlay().hidden).toBe(false));
    expect(shownText()).toBe("That's not an audio file");
    expect(overlay().classList.contains("drop-overlay--refused")).toBe(true);
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
    expect(onStarted).not.toHaveBeenCalled();

    vi.advanceTimersByTime(REFUSAL_SHOWN_MS);

    expect(overlay().hidden).toBe(true);
  });

  it("a click puts a refusal away at once", async () => {
    drag("drop", document.body, ["Files"], [buildFile("empty.wav", 0)]);
    await vi.waitFor(() => expect(shownText()).toBe("This file is empty"));

    overlay().click();

    expect(overlay().hidden).toBe(true);
  });

  it("a new drag after a refusal shows the drop hint again", async () => {
    drag("drop", document.body, ["Files"], [buildFile("big.wav", MAX_UPLOAD_BYTES + 1)]);
    await vi.waitFor(() => expect(shownText()).toBe("This file is over 500 MB"));

    drag("dragenter", document.body, ["Files"]);

    expect(shownText()).toBe("Drop to transcribe");
    expect(overlay().classList.contains("drop-overlay--refused")).toBe(false);
  });

  it("shows the backend's reason when it refuses the upload, and reports no start", async () => {
    apiMock.startFileJob.mockRejectedValue(new Error("Backend is not running"));

    drag("drop", document.body, ["Files"], [buildFile("note.wav", 2048)]);

    await vi.waitFor(() => expect(shownText()).toBe("Backend is not running"));
    expect(overlay().hidden).toBe(false);
    expect(onStarted).not.toHaveBeenCalled();
  });
});

describe("picking a file with pick()", () => {
  it("opens the system file dialog, offering audio files", () => {
    const opened = vi.spyOn(picker(), "click").mockImplementation(() => {});

    mounted.pick();

    expect(opened).toHaveBeenCalledOnce();
    expect(picker().accept).toContain(".m4a");
  });

  it("sends the picked file off as a job and reports the start", async () => {
    pick(buildFile("interview.m4a", 2048));

    await vi.waitFor(() => expect(onStarted).toHaveBeenCalledOnce());
    expect(apiMock.startFileJob.mock.calls[0][1]).toBe("interview.m4a");
    expect(overlay().hidden).toBe(true);
  });

  it("explains a picked file it refuses, the way it explains a dropped one", async () => {
    pick(buildFile("notes.txt", 2048));

    await vi.waitFor(() => expect(shownText()).toBe("That's not an audio file"));
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });
});

describe("a file picked from the ring or the tray", () => {
  async function shellPicks(name: string, size: number): Promise<void> {
    await vi.waitFor(() => expect(shellListeners.get(EVENT_FILE_PICKED)).toBeTypeOf("function"));
    shellListeners.get(EVENT_FILE_PICKED)!({ payload: { token: "token-1", name, size } });
  }

  it("fetches the bytes from the shell by token and sends them off as a job", async () => {
    const bytes = new ArrayBuffer(4);
    invokeMock.mockResolvedValue(bytes);

    await shellPicks("Interview 3.m4a", 2048);

    await vi.waitFor(() => expect(onStarted).toHaveBeenCalledOnce());
    expect(invokeMock).toHaveBeenCalledWith("take_picked_file", { token: "token-1" });
    expect(apiMock.startFileJob).toHaveBeenCalledWith(bytes, "Interview 3.m4a");
  });

  it("refuses a file over the cap in the drop's words without reading it", async () => {
    await shellPicks("All hands.m4a", MAX_UPLOAD_BYTES + 1);

    await vi.waitFor(() => expect(shownText()).toBe("This file is over 500 MB"));
    expect(invokeMock).not.toHaveBeenCalled();
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });

  it("shows the shell's reason when the file cannot be read, and reports no start", async () => {
    invokeMock.mockRejectedValue("This file can't be read");

    await shellPicks("Interview 3.m4a", 2048);

    await vi.waitFor(() => expect(shownText()).toBe("This file can't be read"));
    expect(onStarted).not.toHaveBeenCalled();
  });

  it("stops hearing the shell on teardown", async () => {
    await vi.waitFor(() => expect(shellListeners.get(EVENT_FILE_PICKED)).toBeTypeOf("function"));

    mounted.destroy();
    mounted = { pick: () => {}, destroy: () => {} };

    expect(stopHearing).toHaveBeenCalledOnce();
  });
});

describe("a drop area on the page", () => {
  function showDropArea(): void {
    document.getElementById("pane")!.innerHTML = `<div class="${DROP_AREA_CLASS}"></div>`;
  }

  const lit = () => document.body.classList.contains("dragging-file");

  it("lights the drop area instead of dimming the window when a file is dragged in", () => {
    showDropArea();

    const enter = drag("dragenter", document.getElementById("pane")!, ["Files"]);

    expect(lit()).toBe(true);
    expect(overlay().hidden).toBe(true);
    expect(enter.defaultPrevented).toBe(true);
  });

  it("puts the drop area back when the drag leaves the window", () => {
    showDropArea();
    drag("dragenter", document.body, ["Files"]);

    drag("dragleave", document.body, ["Files"]);

    expect(lit()).toBe(false);
  });

  it("starts the job of a file dropped anywhere and puts the drop area back", async () => {
    showDropArea();
    drag("dragenter", document.body, ["Files"]);

    drag("drop", document.getElementById("pane")!, ["Files"], [buildFile("note.wav", 2048)]);

    expect(lit()).toBe(false);
    await vi.waitFor(() => expect(onStarted).toHaveBeenCalledOnce());
  });

  it("still explains a refusal on the overlay", async () => {
    showDropArea();
    drag("dragenter", document.body, ["Files"]);

    drag("drop", document.body, ["Files"], [buildFile("notes.txt", 2048)]);

    await vi.waitFor(() => expect(shownText()).toBe("That's not an audio file"));
    expect(overlay().hidden).toBe(false);
  });

  it("takes a refusal away when the next drag arrives", async () => {
    showDropArea();
    drag("drop", document.body, ["Files"], [buildFile("empty.wav", 0)]);
    await vi.waitFor(() => expect(overlay().hidden).toBe(false));

    drag("dragenter", document.body, ["Files"]);

    expect(overlay().hidden).toBe(true);
    expect(lit()).toBe(true);
  });

  it("is not lit by a drag that carries no file", () => {
    showDropArea();

    drag("dragenter", document.body, ["text/plain"]);

    expect(lit()).toBe(false);
  });
});

describe("teardown", () => {
  it("removes the overlay and the dialog, and stops listening to the window", () => {
    mounted.destroy();
    mounted = { pick: () => {}, destroy: () => {} };

    const enter = drag("dragenter", document.body, ["Files"]);

    expect(document.querySelector(".drop-overlay")).toBeNull();
    expect(document.querySelector('input[type="file"]')).toBeNull();
    expect(enter.defaultPrevented).toBe(false);
  });

  it("puts a lit drop area back", () => {
    document.getElementById("pane")!.innerHTML = `<div class="${DROP_AREA_CLASS}"></div>`;
    drag("dragenter", document.body, ["Files"]);

    mounted.destroy();
    mounted = { pick: () => {}, destroy: () => {} };

    expect(document.body.classList.contains("dragging-file")).toBe(false);
  });
});

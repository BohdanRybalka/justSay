// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";
import { MAX_UPLOAD_BYTES } from "../../contracts";

const apiMock = {
  processFile: vi.fn(),
};

vi.mock("../../api", () => ({
  api: apiMock,
}));

const unlisten = vi.fn();
const onDragDropEvent = vi.fn(async () => unlisten);

vi.mock("@tauri-apps/api/webview", () => ({
  getCurrentWebview: () => ({ onDragDropEvent }),
}));

const { renderTranscribe } = await import("./transcribe");

const writeText = vi.fn();

const UNREAD_SETTINGS = {} as UserSettings;

function render(): { container: HTMLElement; teardown: () => void } {
  const container = document.createElement("div");
  const teardown = renderTranscribe(container, UNREAD_SETTINGS);
  return { container, teardown };
}

function buildFile(name: string, size: number): File {
  const file = new File(["audio-bytes"], name, { type: "audio/wav" });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

function dropFile(container: HTMLElement, file: File): void {
  const event = new Event("drop", { bubbles: true });
  Object.defineProperty(event, "dataTransfer", {
    value: { files: [file], getData: () => "" },
  });
  container.querySelector("#dropzone")!.dispatchEvent(event);
}

function status(container: HTMLElement): HTMLElement {
  return container.querySelector<HTMLElement>("#result-status")!;
}

function resultText(container: HTMLElement): HTMLElement {
  return container.querySelector<HTMLElement>("#result-text")!;
}

beforeEach(() => {
  vi.clearAllMocks();
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText },
    configurable: true,
  });
});

describe("renderTranscribe — the buttons the markup declares", () => {
  it("btn-pick opens the hidden file input", () => {
    const { container } = render();
    const input = container.querySelector<HTMLInputElement>("#file-input")!;
    const click = vi.spyOn(input, "click").mockImplementation(() => {});

    container.querySelector<HTMLButtonElement>("#btn-pick")!.click();

    expect(click).toHaveBeenCalledTimes(1);
  });

  it("btn-copy puts the transcript on the clipboard and says so", async () => {
    apiMock.processFile.mockResolvedValue({
      text: "hello there",
      duration_ms: 1500,
      copied_to_clipboard: false,
    });
    const { container } = render();
    dropFile(container, buildFile("note.wav", 2048));
    await vi.waitFor(() => {
      expect(resultText(container).textContent).toBe("hello there");
    });

    container.querySelector<HTMLButtonElement>("#btn-copy")!.click();

    await vi.waitFor(() => {
      expect(writeText).toHaveBeenCalledWith("hello there");
    });
    expect(container.querySelector("#btn-copy")!.textContent).toBe("Copied!");
  });

  it("btn-reset hides the result panel and empties it", async () => {
    apiMock.processFile.mockResolvedValue({
      text: "hello there",
      duration_ms: 1500,
      copied_to_clipboard: false,
    });
    const { container } = render();
    dropFile(container, buildFile("note.wav", 2048));
    await vi.waitFor(() => {
      expect(resultText(container).textContent).toBe("hello there");
    });

    container.querySelector<HTMLButtonElement>("#btn-reset")!.click();

    const group = container.querySelector<HTMLElement>("#result-group")!;
    expect(group.style.display).toBe("none");
    expect(resultText(container).textContent).toBe("");
    expect(status(container).textContent).toBe("");
  });
});

describe("renderTranscribe — a file is refused before it is uploaded", () => {
  it("an extension the backend does not accept names the extension", async () => {
    const { container } = render();

    dropFile(container, buildFile("notes.txt", 2048));

    await vi.waitFor(() => {
      expect(status(container).textContent).toBe("Unsupported format: txt");
    });
    expect(apiMock.processFile).not.toHaveBeenCalled();
  });

  it("an empty file is refused", async () => {
    const { container } = render();

    dropFile(container, buildFile("empty.wav", 0));

    await vi.waitFor(() => {
      expect(status(container).textContent).toBe("Empty file");
    });
    expect(apiMock.processFile).not.toHaveBeenCalled();
  });

  it("a file over the upload ceiling names the ceiling", async () => {
    const { container } = render();

    dropFile(container, buildFile("long.wav", MAX_UPLOAD_BYTES + 1));

    await vi.waitFor(() => {
      expect(status(container).textContent).toContain("File too large");
    });
    expect(status(container).textContent).toContain("25 MB limit");
    expect(apiMock.processFile).not.toHaveBeenCalled();
  });
});

describe("renderTranscribe — the result panel", () => {
  it("a transcript arrives with its elapsed time and the clipboard note", async () => {
    apiMock.processFile.mockResolvedValue({
      text: "hello there",
      duration_ms: 1500,
      copied_to_clipboard: true,
    });
    const { container } = render();

    dropFile(container, buildFile("note.wav", 2048));

    await vi.waitFor(() => {
      expect(status(container).textContent).toBe("Done in 1.50s · copied to clipboard");
    });
    expect(resultText(container).textContent).toBe("hello there");
    expect(status(container).className).toBe("result-status ok");
  });

  it("an empty transcript is labelled instead of left blank", async () => {
    apiMock.processFile.mockResolvedValue({
      text: "",
      duration_ms: 900,
      copied_to_clipboard: false,
    });
    const { container } = render();

    dropFile(container, buildFile("silence.wav", 2048));

    await vi.waitFor(() => {
      expect(resultText(container).textContent).toBe("(empty result)");
    });
    expect(status(container).textContent).toBe("Done in 0.90s");
  });

  it("a failed transcription shows the backend's message as an error", async () => {
    apiMock.processFile.mockRejectedValue(new Error("Backend is not running"));
    const { container } = render();

    dropFile(container, buildFile("note.wav", 2048));

    await vi.waitFor(() => {
      expect(status(container).textContent).toBe("Backend is not running");
    });
    expect(status(container).className).toBe("result-status error");
  });
});

describe("renderTranscribe — the drop zone", () => {
  it("dragging over it marks it active and leaving clears the mark", () => {
    const { container } = render();
    const dropzone = container.querySelector<HTMLElement>("#dropzone")!;

    dropzone.dispatchEvent(new Event("dragenter", { bubbles: true }));
    expect(dropzone.classList.contains("active")).toBe(true);

    dropzone.dispatchEvent(new Event("dragleave", { bubbles: true }));
    expect(dropzone.classList.contains("active")).toBe(false);
  });

  it("a drop carrying a path instead of a file says to use the picker", async () => {
    const { container } = render();
    const event = new Event("drop", { bubbles: true });
    Object.defineProperty(event, "dataTransfer", {
      value: { files: [], getData: () => "C:\\audio\\note.wav" },
    });

    container.querySelector("#dropzone")!.dispatchEvent(event);

    await vi.waitFor(() => {
      expect(status(container).textContent).toContain("Use the picker instead");
    });
    expect(apiMock.processFile).not.toHaveBeenCalled();
  });

  it("the teardown unsubscribes the shell's own drag-drop listener", async () => {
    const { teardown } = render();
    await vi.waitFor(() => {
      expect(onDragDropEvent).toHaveBeenCalledTimes(1);
    });

    teardown();

    expect(unlisten).toHaveBeenCalledTimes(1);
  });
});

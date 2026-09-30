// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MAX_UPLOAD_BYTES } from "../../contracts";

const apiMock = {
  startFileJob: vi.fn(),
};

vi.mock("../../api", () => ({
  api: apiMock,
}));

const { renderTranscribe } = await import("./transcribe");

const onStarted = vi.fn();

function render(): { container: HTMLElement; teardown: () => void } {
  const container = document.createElement("div");
  const teardown = renderTranscribe(container, onStarted);
  return { container, teardown };
}

function buildFile(name: string, size: number): File {
  const file = new File(["audio-bytes"], name, { type: "audio/wav" });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

function dropFile(container: HTMLElement, file: File): Event {
  const event = new Event("drop", { bubbles: true, cancelable: true });
  Object.defineProperty(event, "dataTransfer", {
    value: { files: [file], getData: () => "" },
  });
  container.querySelector("#dropzone")!.dispatchEvent(event);
  return event;
}

function status(container: HTMLElement): HTMLElement {
  return container.querySelector<HTMLElement>("#result-status")!;
}

beforeEach(() => {
  vi.clearAllMocks();
  apiMock.startFileJob.mockResolvedValue({ id: "job-1" });
});

describe("renderTranscribe — the buttons the markup declares", () => {
  it("btn-pick opens the hidden file input", () => {
    const { container } = render();
    const input = container.querySelector<HTMLInputElement>("#file-input")!;
    const click = vi.spyOn(input, "click").mockImplementation(() => {});

    container.querySelector<HTMLButtonElement>("#btn-pick")!.click();

    expect(click).toHaveBeenCalledTimes(1);
  });
});

describe("renderTranscribe — a file is refused before it is uploaded", () => {
  it("an extension the backend does not accept names the extension", async () => {
    const { container } = render();

    dropFile(container, buildFile("notes.txt", 2048));

    await vi.waitFor(() => {
      expect(status(container).textContent).toBe("Unsupported format: txt");
    });
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });

  it("an empty file is refused", async () => {
    const { container } = render();

    dropFile(container, buildFile("empty.wav", 0));

    await vi.waitFor(() => {
      expect(status(container).textContent).toBe("Empty file");
    });
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });

  it("a file over the upload ceiling names the ceiling", async () => {
    const { container } = render();

    dropFile(container, buildFile("long.wav", MAX_UPLOAD_BYTES + 1));

    await vi.waitFor(() => {
      expect(status(container).textContent).toContain("File too large");
    });
    expect(status(container).textContent).toContain("25 MB limit");
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });
});

describe("renderTranscribe — sending the file off", () => {
  it("a file is started as a job, History is told, and the zone goes quiet", async () => {
    const { container } = render();

    dropFile(container, buildFile("note.wav", 2048));

    await vi.waitFor(() => {
      expect(onStarted).toHaveBeenCalledTimes(1);
    });
    expect(apiMock.startFileJob.mock.calls[0][1]).toBe("note.wav");
    expect(container.querySelector<HTMLElement>("#result-group")!.style.display).toBe("none");
  });

  it("a refused upload shows the backend's message as an error and tells History nothing", async () => {
    apiMock.startFileJob.mockRejectedValue(new Error("Backend is not running"));
    const { container } = render();

    dropFile(container, buildFile("note.wav", 2048));

    await vi.waitFor(() => {
      expect(status(container).textContent).toBe("Backend is not running");
    });
    expect(status(container).className).toBe("result-status error");
    expect(onStarted).not.toHaveBeenCalled();
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

  it("crossing onto the zone's own children keeps the mark lit", () => {
    const { container } = render();
    const dropzone = container.querySelector<HTMLElement>("#dropzone")!;
    const title = container.querySelector<HTMLElement>(".dropzone-title")!;

    dropzone.dispatchEvent(new Event("dragenter", { bubbles: true }));
    const ontoAChild = new Event("dragleave", { bubbles: true });
    Object.defineProperty(ontoAChild, "relatedTarget", { value: title });
    dropzone.dispatchEvent(ontoAChild);

    expect(
      dropzone.classList.contains("active"),
      "dragleave fires at every child boundary, so clearing the mark here strobes it",
    ).toBe(true);
  });

  it("a drop it handles never reaches the window, which would swallow it", async () => {
    const { container } = render();
    document.body.appendChild(container);
    const reachedTheWindow = vi.fn();
    window.addEventListener("drop", reachedTheWindow);

    dropFile(container, buildFile("note.wav", 2048));
    await vi.waitFor(() => {
      expect(apiMock.startFileJob).toHaveBeenCalledTimes(1);
    });
    window.removeEventListener("drop", reachedTheWindow);
    container.remove();

    expect(
      reachedTheWindow,
      "a file the zone took must not bubble on to the window guard, which cancels the drop",
    ).not.toHaveBeenCalled();
  });

  it("a drop carrying text instead of a file says what it carried", async () => {
    const { container } = render();
    const event = new Event("drop", { bubbles: true });
    Object.defineProperty(event, "dataTransfer", {
      value: { files: [], getData: () => "C:\\audio\\note.wav" },
    });

    container.querySelector("#dropzone")!.dispatchEvent(event);

    await vi.waitFor(() => {
      expect(status(container).textContent).toBe(
        "That drag carried text, not a file. Drop an audio file or use the picker.",
      );
    });
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });

  it("the zone keeps the webview from navigating to the file it was handed", async () => {
    const { container } = render();

    const event = dropFile(container, buildFile("note.wav", 2048));

    await vi.waitFor(() => {
      expect(apiMock.startFileJob).toHaveBeenCalledTimes(1);
    });
    expect(apiMock.startFileJob.mock.calls[0][1]).toBe("note.wav");
    expect(
      event.defaultPrevented,
      "an unprevented drop is a browser navigation to the file, which replaces the UI",
    ).toBe(true);
  });

  it("an upload still in flight at teardown tells History nothing", async () => {
    let finish!: (result: unknown) => void;
    apiMock.startFileJob.mockReturnValue(new Promise((resolve) => (finish = resolve)));
    const { container, teardown } = render();

    dropFile(container, buildFile("inflight.wav", 2048));
    await vi.waitFor(() => {
      expect(apiMock.startFileJob).toHaveBeenCalledTimes(1);
    });

    teardown();
    finish({ id: "job-1" });
    await new Promise((resolve) => setTimeout(resolve, 25));

    expect(onStarted).not.toHaveBeenCalled();
  });

  it("a drop delivered after teardown transcribes nothing", async () => {
    const { container, teardown } = render();

    dropFile(container, buildFile("before.wav", 2048));
    await vi.waitFor(() => {
      expect(apiMock.startFileJob).toHaveBeenCalledTimes(1);
    });

    teardown();
    dropFile(container, buildFile("after.wav", 2048));
    await new Promise((resolve) => setTimeout(resolve, 25));

    expect(apiMock.startFileJob).toHaveBeenCalledTimes(1);
    expect(apiMock.startFileJob.mock.calls[0][1]).toBe("before.wav");
  });
});

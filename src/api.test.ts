// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

function installBridge() {
  (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {};
}

function removeBridge() {
  delete (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__;
}

function okJson(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

function errJson(status: number, detail: string) {
  return { ok: false, status, statusText: "Error", json: async () => ({ detail }) };
}

function headerOf(callIndex: number): Record<string, string> {
  const opts = fetchMock.mock.calls[callIndex][1] as RequestInit;
  return (opts.headers ?? {}) as Record<string, string>;
}

beforeEach(() => {
  vi.resetModules();
  invokeMock.mockReset();
  fetchMock.mockReset();
  installBridge();
});

afterEach(() => {
  removeBridge();
  vi.useRealTimers();
});

describe("token injection when a backend token is available", () => {
  beforeEach(() => {
    invokeMock.mockResolvedValue("secret-token");
  });

  it("request() attaches X-JustSay-Token", async () => {
    const { api } = await import("./api");
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    await api.health();

    expect(invokeMock).toHaveBeenCalledWith("get_backend_token");
    expect(headerOf(0)["X-JustSay-Token"]).toBe("secret-token");
  });

  it("processFile() attaches X-JustSay-Token", async () => {
    const { api } = await import("./api");
    fetchMock.mockResolvedValue(okJson({ text: "hi" }));

    await api.processFile(new ArrayBuffer(4), "clip.wav");

    expect(headerOf(0)["X-JustSay-Token"]).toBe("secret-token");
  });

  it("levelStream() attaches X-JustSay-Token", async () => {
    const { levelStream } = await import("./api");
    fetchMock.mockResolvedValue({ ok: false, status: 500, body: null });

    levelStream(
      () => {},
      () => {},
      () => {},
    );

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(headerOf(0)["X-JustSay-Token"]).toBe("secret-token");
  });

  it("reports an ok bridge diagnosis", async () => {
    const { api, lastBridgeDiagnosis } = await import("./api");
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    await api.health();

    expect(lastBridgeDiagnosis()).toEqual({ kind: "ok" });
  });
});

describe("open mode when invoke is unavailable", () => {
  beforeEach(() => {
    invokeMock.mockRejectedValue(new Error("not running in tauri"));
  });

  it("request() sends no token header and does not throw", async () => {
    const { api } = await import("./api");
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    await expect(api.health()).resolves.toEqual({ status: "ok" });

    expect(headerOf(0)["X-JustSay-Token"]).toBeUndefined();
  });
});

describe("token cache does not poison on a transient failure", () => {
  it("retries invoke on a later request after a failed invoke (does not cache null forever)", async () => {
    const { api } = await import("./api");
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    invokeMock
      .mockRejectedValueOnce(new Error("transient"))
      .mockResolvedValue("recovered-token");
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    await api.health();
    expect(headerOf(0)["X-JustSay-Token"]).toBeUndefined();
    expect(warnSpy).toHaveBeenCalled();

    await api.health();
    expect(invokeMock).toHaveBeenCalledTimes(2);
    expect(headerOf(1)["X-JustSay-Token"]).toBe("recovered-token");

    warnSpy.mockRestore();
  });

  it("caches a successful token: invoke runs once across multiple requests", async () => {
    const { api } = await import("./api");
    invokeMock.mockResolvedValue("secret-token");
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    await api.health();
    await api.health();
    await api.audioStatus();

    expect(invokeMock).toHaveBeenCalledTimes(1);
    expect(headerOf(2)["X-JustSay-Token"]).toBe("secret-token");
  });
});

describe("a hung IPC transport strands at most one token call", () => {
  it("repeated polls against a never-settling invoke issue exactly one invoke", async () => {
    vi.useFakeTimers();
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { api, lastBridgeDiagnosis } = await import("./api");
    invokeMock.mockReturnValue(new Promise(() => {}));
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    for (let poll = 0; poll < 4; poll++) {
      const inFlight = api.health();
      await vi.advanceTimersByTimeAsync(3000);
      await inFlight;
    }

    expect(lastBridgeDiagnosis()).toEqual({ kind: "invoke-timeout" });
    expect(invokeMock).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });

  it("a call still unanswered past the reuse window is abandoned for a fresh one", async () => {
    vi.useFakeTimers();
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { api } = await import("./api");
    invokeMock.mockReturnValue(new Promise(() => {}));
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    const first = api.health();
    await vi.advanceTimersByTimeAsync(3000);
    await first;
    expect(invokeMock).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(60_000);

    const later = api.health();
    await vi.advanceTimersByTimeAsync(3000);
    await later;

    expect(invokeMock).toHaveBeenCalledTimes(2);
    warnSpy.mockRestore();
  });

  it("a hung call that finally answers does not wedge later requests", async () => {
    vi.useFakeTimers();
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { api } = await import("./api");
    let answer: (token: string) => void = () => {};
    invokeMock.mockReturnValueOnce(
      new Promise<string>((resolve) => {
        answer = resolve;
      }),
    );
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    const timedOut = api.health();
    await vi.advanceTimersByTimeAsync(3000);
    await timedOut;
    expect(headerOf(0)["X-JustSay-Token"]).toBeUndefined();

    answer("late-token");
    await vi.advanceTimersByTimeAsync(0);
    invokeMock.mockResolvedValue("late-token");

    await api.health();

    expect(headerOf(1)["X-JustSay-Token"]).toBe("late-token");
    warnSpy.mockRestore();
  });
});

describe("bridge diagnosis", () => {
  it("bridge-missing: Tauri's injected globals never ran, so invoke is not even attempted", async () => {
    removeBridge();
    const { api, lastBridgeDiagnosis } = await import("./api");
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    await api.health();

    expect(lastBridgeDiagnosis()).toEqual({ kind: "bridge-missing" });
    expect(invokeMock).not.toHaveBeenCalled();
    expect(headerOf(0)["X-JustSay-Token"]).toBeUndefined();
  });

  it("invoke-timeout: an invoke that never settles releases the caller instead of hanging", async () => {
    vi.useFakeTimers();
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { api, lastBridgeDiagnosis } = await import("./api");
    invokeMock.mockReturnValue(new Promise(() => {}));
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    const inFlight = api.health();
    await vi.advanceTimersByTimeAsync(3000);
    await inFlight;

    expect(lastBridgeDiagnosis()).toEqual({ kind: "invoke-timeout" });
    expect(headerOf(0)["X-JustSay-Token"]).toBeUndefined();
    warnSpy.mockRestore();
  });

  it("invoke-failed: carries the underlying error message as the detail", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { api, lastBridgeDiagnosis } = await import("./api");
    invokeMock.mockRejectedValue(new Error("command get_backend_token not found"));
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    await api.health();

    expect(lastBridgeDiagnosis()).toEqual({
      kind: "invoke-failed",
      detail: "command get_backend_token not found",
    });
    warnSpy.mockRestore();
  });

  it("the three failure diagnoses are distinguishable, not one collapsed state", async () => {
    const kinds = new Set<string>();
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    removeBridge();
    let mod = await import("./api");
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));
    await mod.api.health();
    kinds.add(mod.lastBridgeDiagnosis().kind);

    vi.resetModules();
    installBridge();
    vi.useFakeTimers();
    mod = await import("./api");
    invokeMock.mockReturnValue(new Promise(() => {}));
    const inFlight = mod.api.health();
    await vi.advanceTimersByTimeAsync(3000);
    await inFlight;
    kinds.add(mod.lastBridgeDiagnosis().kind);
    vi.useRealTimers();

    vi.resetModules();
    mod = await import("./api");
    invokeMock.mockReset();
    invokeMock.mockRejectedValue(new Error("boom"));
    await mod.api.health();
    kinds.add(mod.lastBridgeDiagnosis().kind);

    expect([...kinds].sort()).toEqual(["bridge-missing", "invoke-failed", "invoke-timeout"]);
    warnSpy.mockRestore();
  });
});

describe("401 handling", () => {
  it("request() throws ApiAuthError carrying the diagnosis, and sawAuthFailure() flips", async () => {
    removeBridge();
    const { api, ApiAuthError, sawAuthFailure } = await import("./api");
    fetchMock.mockResolvedValue(errJson(401, "Missing or invalid API token"));

    expect(sawAuthFailure()).toBe(false);
    await expect(api.getSettings()).rejects.toBeInstanceOf(ApiAuthError);
    expect(sawAuthFailure()).toBe(true);

    fetchMock.mockResolvedValue(errJson(401, "Missing or invalid API token"));
    const error = await api.getSettings().catch((e) => e);
    expect(error).toBeInstanceOf(ApiAuthError);
    expect(error.message).toBe("Missing or invalid API token");
    expect(error.diagnosis).toEqual({ kind: "bridge-missing" });
  });

  it("processFile() throws ApiAuthError on 401 too", async () => {
    const { api, ApiAuthError } = await import("./api");
    invokeMock.mockResolvedValue("secret-token");
    fetchMock.mockResolvedValue(errJson(401, "Missing or invalid API token"));

    await expect(api.processFile(new ArrayBuffer(4), "clip.wav")).rejects.toBeInstanceOf(
      ApiAuthError,
    );
  });

  it("a non-401 failure stays a plain Error and does not flip sawAuthFailure()", async () => {
    const { api, ApiAuthError, sawAuthFailure } = await import("./api");
    invokeMock.mockResolvedValue("secret-token");
    fetchMock.mockResolvedValue(errJson(500, "boom"));

    const error = await api.getSettings().catch((e) => e);
    expect(error).toBeInstanceOf(Error);
    expect(error).not.toBeInstanceOf(ApiAuthError);
    expect(sawAuthFailure()).toBe(false);
  });

  it("an open backend with no bridge never reports an auth failure", async () => {
    removeBridge();
    const { api, sawAuthFailure } = await import("./api");
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    await api.health();
    await api.getSettings();

    expect(sawAuthFailure()).toBe(false);
  });
});

describe("sawAuthFailure() clears as well as sets", () => {
  async function observeA401() {
    const mod = await import("./api");
    fetchMock.mockResolvedValueOnce(errJson(401, "Missing or invalid API token"));
    await expect(mod.api.getSettings()).rejects.toBeInstanceOf(mod.ApiAuthError);
    expect(mod.sawAuthFailure()).toBe(true);
    return mod;
  }

  beforeEach(() => {
    invokeMock.mockResolvedValue("secret-token");
  });

  it("a 2xx from a token-gated path clears an observed auth failure", async () => {
    const { api, sawAuthFailure } = await observeA401();

    fetchMock.mockResolvedValue(okJson({ language: "uk" }));
    await api.getSettings();

    expect(sawAuthFailure()).toBe(false);
  });

  it("a 200 from GET /health never clears it — the route is exempt from the gate", async () => {
    const { api, sawAuthFailure } = await observeA401();

    fetchMock.mockResolvedValue(okJson({ status: "ok" }));
    await api.health();
    await api.health();

    expect(sawAuthFailure()).toBe(true);
  });

  it("levelStream() reports its outcome through the same flag", async () => {
    const { levelStream, sawAuthFailure } = await import("./api");
    fetchMock.mockResolvedValue({ ok: false, status: 401, body: null });

    levelStream(
      () => {},
      () => {},
      () => {},
    );

    await vi.waitFor(() => expect(sawAuthFailure()).toBe(true));
  });
});

describe("a backend that accepts a request and never answers", () => {
  beforeEach(() => {
    invokeMock.mockResolvedValue("secret-token");
    vi.useFakeTimers();
  });

  /** A `fetch` that behaves like the real one against a backend that has stopped
   *  answering: it holds the connection open and settles only when aborted. */
  function deafFetch() {
    return (_url: string, opts: RequestInit) =>
      new Promise<Response>((_, reject) => {
        opts.signal?.addEventListener("abort", () =>
          reject(new DOMException("The operation was aborted.", "AbortError")),
        );
      });
  }

  /** The half-answer a headers-only budget cannot catch: the status line and the
   *  headers arrive, so `fetch` settles, and then the body never comes. */
  function headersThenSilenceFetch() {
    return (_url: string, opts: RequestInit) =>
      Promise.resolve({
        ok: true,
        status: 200,
        json: () =>
          new Promise((_, reject) => {
            opts.signal?.addEventListener("abort", () =>
              reject(new DOMException("The operation was aborted.", "AbortError")),
            );
          }),
      } as unknown as Response);
  }

  it("abandons an ordinary request at the budget instead of never settling", async () => {
    const { api, REQUEST_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.health();
    const settled = vi.fn();
    pending.then(settled, settled);

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS - 1);
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(2);
    await expect(pending).rejects.toThrow("the backend did not answer /health within 15 seconds");
  });

  it("aborts the request it gave up on, so the socket is not left open", async () => {
    const { api, REQUEST_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.audioStart();
    pending.catch(() => {});
    await vi.advanceTimersByTimeAsync(0);
    const signal = (fetchMock.mock.calls[0][1] as RequestInit).signal!;
    expect(signal.aborted).toBe(false);

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);
    expect(signal.aborted).toBe(true);
  });

  it("keeps the query string out of the message, since it carries what was typed", async () => {
    const { api, REQUEST_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.searchHistory("my private note");
    pending.catch(() => {});
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);

    await expect(pending).rejects.toThrow(
      "the backend did not answer /history/search within 15 seconds",
    );
    const message = await pending.then(
      () => "resolved",
      (e: Error) => e.message,
    );
    expect(message).not.toContain(encodeURIComponent("my private note"));
    expect(message).not.toContain("?q=");
  });

  it("leaks not even a one-word query, which percent-encoding passes through verbatim", async () => {
    const { api, REQUEST_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.searchHistory("secret");
    pending.catch(() => {});
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);

    const message = await pending.then(
      () => "resolved",
      (e: Error) => e.message,
    );
    expect(encodeURIComponent("secret")).toBe("secret");
    expect(message).not.toContain("secret");
    expect(message).not.toContain("?q=");
  });

  it("gives transcription the long budget rather than the short one", async () => {
    const { api, REQUEST_TIMEOUT_MS, LONG_REQUEST_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.dictate("uk");
    const settled = vi.fn();
    pending.then(settled, settled);

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS * 2);
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(LONG_REQUEST_TIMEOUT_MS);
    await expect(pending).rejects.toThrow("within 600 seconds");
  });

  it("gives a file upload the long budget, since a 25 MB upload is minutes of audio", async () => {
    const { api, LONG_REQUEST_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.processFile(new ArrayBuffer(8), "call.wav");
    const settled = vi.fn();
    pending.then(settled, settled);

    await vi.advanceTimersByTimeAsync(300_000);
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(LONG_REQUEST_TIMEOUT_MS);
    await expect(pending).rejects.toThrow(
      "the backend did not answer /pipeline/process-file within 600 seconds",
    );
  });

  it("gives the local model load the long budget, since a first run downloads it", async () => {
    const { api, LONG_REQUEST_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.sttLocalLoad();
    const settled = vi.fn();
    pending.then(settled, settled);

    await vi.advanceTimersByTimeAsync(300_000);
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(LONG_REQUEST_TIMEOUT_MS);
    await expect(pending).rejects.toThrow(
      "the backend did not answer /stt/local/load within 600 seconds",
    );
  });

  it("gives a status read the short budget, so a recovery read does not cost a second full one", async () => {
    const { api, STATUS_TIMEOUT_MS, REQUEST_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.audioStatus();
    const settled = vi.fn();
    pending.then(settled, settled);

    await vi.advanceTimersByTimeAsync(STATUS_TIMEOUT_MS - 1);
    expect(settled).not.toHaveBeenCalled();
    expect(STATUS_TIMEOUT_MS).toBeLessThan(REQUEST_TIMEOUT_MS);

    await vi.advanceTimersByTimeAsync(2);
    await expect(pending).rejects.toThrow(
      "the backend did not answer /audio/status within 3 seconds",
    );
  });

  it("gives the meeting status read the same short budget, since it reads the same in-memory state", async () => {
    const { api, STATUS_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.getMeetingStatus();
    const settled = vi.fn();
    pending.then(settled, settled);

    await vi.advanceTimersByTimeAsync(STATUS_TIMEOUT_MS - 1);
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(2);
    await expect(pending).rejects.toThrow(
      "the backend did not answer /audio/meeting/status within 3 seconds",
    );
  });

  it("holds the budget through the body, not only through the headers", async () => {
    const { api, REQUEST_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(headersThenSilenceFetch());

    const pending = api.health();
    const settled = vi.fn();
    pending.then(settled, settled);

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS - 1);
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(2);
    await expect(pending).rejects.toThrow("the backend did not answer /health within 15 seconds");
  });

  it("rejects with a TimedOutError carrying the budget and the path, so callers can branch", async () => {
    const { api, REQUEST_TIMEOUT_MS } = await import("./api");
    const { TimedOutError } = await import("./timeout");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.audioStart();
    pending.catch(() => {});
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);

    const error = await pending.then(
      () => null,
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(TimedOutError);
    expect((error as InstanceType<typeof TimedOutError>).budgetMs).toBe(REQUEST_TIMEOUT_MS);
    expect((error as InstanceType<typeof TimedOutError>).subject).toBe("/audio/start");
  });

  it("carries the same identity when it is the body that stops part-way", async () => {
    const { api, REQUEST_TIMEOUT_MS } = await import("./api");
    const { TimedOutError } = await import("./timeout");
    fetchMock.mockImplementation(headersThenSilenceFetch());

    const pending = api.audioStatus();
    pending.catch(() => {});
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);

    const error = await pending.then(
      () => null,
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(TimedOutError);
    expect((error as InstanceType<typeof TimedOutError>).subject).toBe("/audio/status");
  });

  it("lets the widget's intent queue recover instead of wedging on one dead request", async () => {
    const { api, REQUEST_TIMEOUT_MS } = await import("./api");
    const { createRecordingIntentQueue } = await import("./widget/recording-intent");
    fetchMock.mockImplementation(deafFetch());

    let recording = false;
    const errors: unknown[] = [];
    const queue = createRecordingIntentQueue({
      isRecording: () => recording,
      isBusy: () => false,
      startRecording: async () => {
        await api.audioStart();
        recording = true;
      },
      stopRecording: async () => {
        await api.dictate("uk");
        recording = false;
      },
      reportError: (e) => errors.push(e),
    });

    const firstPress = queue.request("start");
    firstPress.catch(() => {});
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);
    await firstPress;

    expect(errors).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const secondPress = queue.request("start");
    secondPress.catch(() => {});
    await vi.advanceTimersByTimeAsync(0);
    expect(secondPress).not.toBe(firstPress);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);
    await secondPress;
    expect(errors).toHaveLength(2);
  });
});

describe("the level stream's handshake, which is bounded while the stream is not", () => {
  beforeEach(() => {
    invokeMock.mockResolvedValue("secret-token");
    vi.useFakeTimers();
  });

  function streamOf(chunks: string[], hold: Promise<void>) {
    let index = 0;
    const encoder = new TextEncoder();
    return {
      ok: true,
      status: 200,
      body: {
        getReader: () => ({
          read: async () => {
            if (index < chunks.length) {
              return { done: false, value: encoder.encode(chunks[index++]) };
            }
            await hold;
            return { done: true, value: undefined };
          },
        }),
      },
    } as unknown as Response;
  }

  it("reports an error when the backend never sends response headers", async () => {
    const { levelStream, REQUEST_TIMEOUT_MS } = await import("./api");
    fetchMock.mockImplementation(
      (_url: string, opts: RequestInit) =>
        new Promise<Response>((_, reject) => {
          opts.signal?.addEventListener("abort", () =>
            reject(new DOMException("The operation was aborted.", "AbortError")),
          );
        }),
    );
    const onError = vi.fn();

    levelStream(
      () => {},
      () => {},
      onError,
    );

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS - 1);
    expect(onError).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(2);
    expect(onError).toHaveBeenCalledOnce();
    expect(onError.mock.calls[0][0]).toContain("/audio/level-stream");
  });

  it("never cuts off a stream that has already delivered its first chunk", async () => {
    const { levelStream, REQUEST_TIMEOUT_MS } = await import("./api");
    let release = () => {};
    const hold = new Promise<void>((resolve) => {
      release = resolve;
    });
    fetchMock.mockResolvedValue(
      streamOf(['event: level\ndata: {"level_db":-20,"is_recording":true}\n\n'], hold),
    );
    const onLevel = vi.fn();
    const onError = vi.fn();

    const controller = levelStream(onLevel, () => {}, onError);

    await vi.advanceTimersByTimeAsync(0);
    expect(onLevel).toHaveBeenCalledOnce();

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS * 4);
    expect(onError).not.toHaveBeenCalled();
    expect(controller.signal.aborted).toBe(false);

    release();
  });

  it("stays silent when the caller aborts the stream itself", async () => {
    const { levelStream } = await import("./api");
    fetchMock.mockImplementation(
      (_url: string, opts: RequestInit) =>
        new Promise<Response>((_, reject) => {
          opts.signal?.addEventListener("abort", () =>
            reject(new DOMException("The operation was aborted.", "AbortError")),
          );
        }),
    );
    const onError = vi.fn();

    const controller = levelStream(
      () => {},
      () => {},
      onError,
    );

    await vi.advanceTimersByTimeAsync(0);
    controller.abort();
    await vi.advanceTimersByTimeAsync(0);

    expect(onError).not.toHaveBeenCalled();
  });
});

describe("a token wait that never ends, because the bridge module never loads", () => {
  beforeEach(() => {
    invokeMock.mockResolvedValue("secret-token");
    vi.useFakeTimers();
    vi.doMock("@tauri-apps/api/core", () => new Promise(() => {}));
  });

  afterEach(() => {
    vi.doMock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
    vi.resetModules();
  });

  function deafFetch() {
    return (_url: string, opts: RequestInit) =>
      new Promise<Response>((_, reject) => {
        opts.signal?.addEventListener("abort", () =>
          reject(new DOMException("The operation was aborted.", "AbortError")),
        );
      });
  }

  it("gives the bridge import its own budget, so the shared token promise settles rather than retaining every later caller", async () => {
    const { api, lastBridgeDiagnosis, REQUEST_TIMEOUT_MS } = await import("./api");
    const { TimedOutError } = await import("./timeout");
    fetchMock.mockImplementation(deafFetch());

    const pending = api.health();
    const settled = vi.fn();
    pending.then(settled, settled);

    await vi.advanceTimersByTimeAsync(0);
    expect(fetchMock).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS - 1);
    expect(lastBridgeDiagnosis()).toEqual({ kind: "bridge-timeout" });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][1].headers).not.toHaveProperty("X-JustSay-Token");
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(2);
    await expect(pending).rejects.toBeInstanceOf(TimedOutError);
  });

  it("frees a later caller after a token wait that never ends, rather than pooling them on it", async () => {
    const { api, REQUEST_TIMEOUT_MS } = await import("./api");
    const { TimedOutError } = await import("./timeout");
    fetchMock.mockImplementation(deafFetch());

    const first = api.audioStart();
    first.catch(() => {});
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);
    await expect(first).rejects.toBeInstanceOf(TimedOutError);

    const second = api.audioStart();
    second.catch(() => {});
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);
    await expect(second).rejects.toBeInstanceOf(TimedOutError);
  });
});

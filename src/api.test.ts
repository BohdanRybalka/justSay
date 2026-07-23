// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// getToken() checks for Tauri's injected bridge, then dynamically imports
// @tauri-apps/api/core and calls invoke("get_backend_token"). Mock the module
// so we can drive every branch, and install/remove the bridge marker per test
// — its absence is itself one of the states under test. The token is cached at
// module scope in api.ts, so each test resets the module registry
// (vi.resetModules) and re-imports ./api to get a fresh cache.
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
    // Resolve a non-ok response so the SSE reader loop returns immediately.
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
    // Simulates running outside a Tauri WebView (invoke rejects / module
    // absent). getToken() must swallow this and resolve null.
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
    // First fetch of the token fails; a later one succeeds.
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

// The three diagnoses below are the whole point of spec 042: on macOS they are
// the only signal available about which layer failed, so they must stay
// distinguishable from one another.
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
    // The real hang mode: invoke() settles only through a registered callback
    // and its send path has no reject channel (ADR 028). Without the timeout
    // in getToken() this pending promise hangs the suite.
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

  // The negative case the diagnosis must never fire on: a manually launched
  // backend with no token configured returns 200 everywhere, so nothing is
  // wrong even though the bridge is absent.
  it("an open backend with no bridge never reports an auth failure", async () => {
    removeBridge();
    const { api, sawAuthFailure } = await import("./api");
    fetchMock.mockResolvedValue(okJson({ status: "ok" }));

    await api.health();
    await api.getSettings();

    expect(sawAuthFailure()).toBe(false);
  });
});

// sawAuthFailure() means "the outcome of the most recent token-gated request",
// so it has to move in both directions — a flag that only latches on makes the
// badge stop tracking reality just as surely as one that never fires.
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

  // AC 10, and the whole point of TOKEN_EXEMPT_PATHS: GET /health is exempt
  // from the backend's token gate, so a 200 from it is not evidence the token
  // was accepted. Clearing on it would let the 5 s health poll repaint the
  // badge green over a completely unauthorized app.
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

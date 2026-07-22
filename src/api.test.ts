// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";

// getToken() dynamically imports @tauri-apps/api/core and calls
// invoke("get_backend_token"). Mock it so we can drive both the "token
// available" and "invoke unavailable" branches. The token is cached at module
// scope in api.ts, so each test resets the module registry (vi.resetModules)
// and re-imports ./api to get a fresh cache.
const invokeMock = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

function okJson(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

function headerOf(callIndex: number): Record<string, string> {
  const opts = fetchMock.mock.calls[callIndex][1] as RequestInit;
  return (opts.headers ?? {}) as Record<string, string>;
}

beforeEach(() => {
  vi.resetModules();
  invokeMock.mockReset();
  fetchMock.mockReset();
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

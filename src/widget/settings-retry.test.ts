import { describe, expect, it, vi } from "vitest";
import { TOKEN_CALL_REUSE_MS } from "../api";
import {
  createSettingsRetry,
  SETTINGS_RETRY_DELAYS_MS,
  type SettingsRetryActions,
  type WidgetSettings,
} from "./settings-retry";

const FETCHED: WidgetSettings = { language: "uk", shortcut: "Alt+Space" };

function refused() {
  return new Error("HTTP 401 Missing or invalid API token");
}

function clock() {
  let value = 0;
  return {
    now: () => value,
    advance: (ms: number) => {
      value += ms;
    },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function actions(overrides: Partial<SettingsRetryActions> = {}) {
  const spies = {
    now: vi.fn(() => 0),
    fetchSettings: vi.fn(async () => FETCHED),
    applySettings: vi.fn(async () => {}),
    applyFallbackShortcut: vi.fn(async () => {}),
    reportAttemptFailed: vi.fn(),
    reportGaveUp: vi.fn(),
  };
  return Object.assign(spies, overrides);
}

function alwaysRefused(time: ReturnType<typeof clock>) {
  return actions({
    now: time.now,
    fetchSettings: vi.fn(async () => {
      throw refused();
    }),
  });
}

async function exhaustTheSchedule(time: ReturnType<typeof clock>, deps: SettingsRetryActions) {
  const retry = createSettingsRetry(deps);
  await retry.load();
  for (let tick = 0; tick < 20; tick += 1) {
    time.advance(200_000);
    await retry.retryIfDue();
  }
  return retry;
}

describe("the bounded settings retry", () => {
  it("applies the fetched language and shortcut on a successful first load", async () => {
    const deps = actions();

    await createSettingsRetry(deps).load();

    expect(deps.applySettings).toHaveBeenCalledWith(FETCHED);
    expect(deps.applyFallbackShortcut).not.toHaveBeenCalled();
    expect(deps.reportAttemptFailed).not.toHaveBeenCalled();
  });

  it("applies the fallback shortcut when the first load fails", async () => {
    const time = clock();
    const deps = alwaysRefused(time);

    await createSettingsRetry(deps).load();

    expect(deps.applyFallbackShortcut).toHaveBeenCalledOnce();
    expect(deps.applySettings).not.toHaveBeenCalled();
    expect(deps.reportAttemptFailed).toHaveBeenCalledOnce();
    expect(deps.reportGaveUp).not.toHaveBeenCalled();
  });

  it("applies the real settings once a retry succeeds", async () => {
    const time = clock();
    const deps = actions({
      now: time.now,
      fetchSettings: vi.fn().mockRejectedValueOnce(refused()).mockResolvedValueOnce(FETCHED),
    });
    const retry = createSettingsRetry(deps);

    await retry.load();
    time.advance(SETTINGS_RETRY_DELAYS_MS[0]);
    await retry.retryIfDue();

    expect(deps.fetchSettings).toHaveBeenCalledTimes(2);
    expect(deps.applySettings).toHaveBeenCalledWith(FETCHED);
  });

  it("does not fetch again before the scheduled gap has elapsed", async () => {
    const time = clock();
    const deps = alwaysRefused(time);
    const retry = createSettingsRetry(deps);

    await retry.load();
    time.advance(SETTINGS_RETRY_DELAYS_MS[0] - 1);
    await retry.retryIfDue();

    expect(deps.fetchSettings).toHaveBeenCalledOnce();
  });

  it("requests a permanently refused endpoint exactly six times", async () => {
    const time = clock();
    const deps = alwaysRefused(time);

    await exhaustTheSchedule(time, deps);

    expect(deps.fetchSettings).toHaveBeenCalledTimes(6);
    expect(deps.fetchSettings).toHaveBeenCalledTimes(1 + SETTINGS_RETRY_DELAYS_MS.length);
  });

  it("applies the fallback once and reports giving up once across the whole schedule", async () => {
    const time = clock();
    const deps = alwaysRefused(time);

    await exhaustTheSchedule(time, deps);

    expect(deps.applyFallbackShortcut).toHaveBeenCalledOnce();
    expect(deps.reportGaveUp).toHaveBeenCalledOnce();
    expect(deps.reportAttemptFailed).toHaveBeenCalledTimes(6);
  });

  it("starts no second fetch while one is already in flight", async () => {
    const time = clock();
    const gate = deferred<WidgetSettings>();
    const deps = actions({
      now: time.now,
      fetchSettings: vi
        .fn()
        .mockRejectedValueOnce(refused())
        .mockImplementationOnce(() => gate.promise),
    });
    const retry = createSettingsRetry(deps);

    await retry.load();
    time.advance(SETTINGS_RETRY_DELAYS_MS[0]);
    const inFlight = retry.retryIfDue();
    time.advance(SETTINGS_RETRY_DELAYS_MS[1]);
    const overlapping = retry.retryIfDue();
    gate.resolve(FETCHED);
    await Promise.all([inFlight, overlapping]);

    expect(deps.fetchSettings).toHaveBeenCalledTimes(2);
  });

  it("keeps retrying when the fetched settings cannot be applied", async () => {
    const time = clock();
    const deps = actions({
      now: time.now,
      applySettings: vi.fn(async () => {
        throw new Error("the shortcut could not be registered");
      }),
    });
    const retry = createSettingsRetry(deps);

    await retry.load();
    time.advance(SETTINGS_RETRY_DELAYS_MS[0]);
    await retry.retryIfDue();

    expect(deps.fetchSettings).toHaveBeenCalledTimes(2);
    expect(deps.reportAttemptFailed).toHaveBeenCalledTimes(2);
  });

  it("reports giving up once, even when a later manual load fails again", async () => {
    const time = clock();
    const deps = alwaysRefused(time);
    const retry = await exhaustTheSchedule(time, deps);

    await retry.load();

    expect(deps.reportGaveUp).toHaveBeenCalledOnce();
  });

  it("keeps the last gap longer than the token-call reuse window", () => {
    const lastGap = SETTINGS_RETRY_DELAYS_MS[SETTINGS_RETRY_DELAYS_MS.length - 1];
    expect(lastGap).toBeGreaterThan(TOKEN_CALL_REUSE_MS);
  });

  it("still fetches on a load requested after the retries gave up", async () => {
    const time = clock();
    const deps = alwaysRefused(time);
    const retry = await exhaustTheSchedule(time, deps);

    await retry.load();

    expect(deps.fetchSettings).toHaveBeenCalledTimes(7);
  });
});

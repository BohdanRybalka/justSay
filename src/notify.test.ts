import { beforeEach, describe, expect, it, vi } from "vitest";

const sendNotificationMock = vi.fn();
vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn(async () => true),
  requestPermission: vi.fn(async () => "granted"),
  sendNotification: sendNotificationMock,
}));

const {
  ERROR_WITHOUT_READABLE_TEXT,
  displayableError,
  nextConnectionCheckState,
  notifyError,
  onConnectivityChange,
} = await import("./notify");

const UNREADABLE = [
  ["an empty string", ""],
  ["spaces alone", "   "],
  ["a newline alone", "\n"],
  ["a tab and a carriage return", "\t \r\n"],
  ["a byte-order mark", "﻿"],
  ["a zero-width space", "​"],
  ["what `str(KeyError(\"\"))` produces", "''"],
  ["punctuation alone", "-->"],
] as const;

const READABLE = [
  ["a plain sentence", "The model file is missing.", "The model file is missing."],
  ["a padded sentence", "  The model file is missing.\n", "The model file is missing."],
  ["a Ukrainian message", "Не вдалося завантажити модель", "Не вдалося завантажити модель"],
  ["a Japanese message", "モデルを読み込めません", "モデルを読み込めません"],
  ["a bare digit", "0", "0"],
] as const;

beforeEach(() => {
  vi.clearAllMocks();
});

describe("onConnectivityChange", () => {
  it("online -> online: stays online, no notify", () => {
    expect(onConnectivityChange(false, false)).toEqual({ offline: false, shouldNotify: false });
  });

  it("online -> offline: edge triggers notify", () => {
    expect(onConnectivityChange(false, true)).toEqual({ offline: true, shouldNotify: true });
  });

  it("offline -> offline: repeated poll, no re-notify", () => {
    expect(onConnectivityChange(true, true)).toEqual({ offline: true, shouldNotify: false });
  });

  it("offline -> online: recovery resets the edge, no notify", () => {
    expect(onConnectivityChange(true, false)).toEqual({ offline: false, shouldNotify: false });
  });
});

describe("nextConnectionCheckState", () => {
  it("first-ever call succeeding: no notify, firstCheckDone flips true", () => {
    expect(nextConnectionCheckState({ offline: false, firstCheckDone: false }, true)).toEqual({
      offline: false,
      firstCheckDone: true,
      shouldNotify: false,
    });
  });

  it("first-ever call failing: no notify despite the offline edge, firstCheckDone flips true", () => {
    expect(nextConnectionCheckState({ offline: false, firstCheckDone: false }, false)).toEqual({
      offline: true,
      firstCheckDone: true,
      shouldNotify: false,
    });
  });

  it("a later failure after a prior success: notifies", () => {
    expect(nextConnectionCheckState({ offline: false, firstCheckDone: true }, false)).toEqual({
      offline: true,
      firstCheckDone: true,
      shouldNotify: true,
    });
  });

  it("a continued outage: no repeat notify", () => {
    expect(nextConnectionCheckState({ offline: true, firstCheckDone: true }, false)).toEqual({
      offline: true,
      firstCheckDone: true,
      shouldNotify: false,
    });
  });

  it("recovery: offline flips back to false, no notify", () => {
    expect(nextConnectionCheckState({ offline: true, firstCheckDone: true }, true)).toEqual({
      offline: false,
      firstCheckDone: true,
      shouldNotify: false,
    });
  });
});

describe("displayableError", () => {
  it("returns null for null, and for nothing else", () => {
    expect(displayableError(null)).toBeNull();
    const nulled = [...UNREADABLE, ...READABLE]
      .map(([label, raw]) => [label, displayableError(raw)] as const)
      .filter(([, shown]) => shown === null)
      .map(([label]) => label);
    expect(nulled).toEqual([]);
  });

  it.each(UNREADABLE)("replaces %s with a sentence the toast can draw", (_label, raw) => {
    const shown = displayableError(raw);
    expect(shown).toBe(ERROR_WITHOUT_READABLE_TEXT);
    expect(shown).toMatch(/[\p{L}\p{N}]/u);
  });

  it.each(READABLE)("keeps %s", (_label, raw, expected) => {
    const shown = displayableError(raw);
    expect(shown).toBe(expected);
    expect(shown).toMatch(/[\p{L}\p{N}]/u);
  });
});

describe("notifyError", () => {
  it.each(UNREADABLE)("never sends %s as the notification body", async (_label, raw) => {
    await notifyError(raw);

    expect(sendNotificationMock).toHaveBeenCalledTimes(1);
    const { body, title } = sendNotificationMock.mock.calls[0][0];
    expect(body).toBe(ERROR_WITHOUT_READABLE_TEXT);
    expect(body).toMatch(/[\p{L}\p{N}]/u);
    expect(title).toBe("JustSay");
  });

  it("sends a readable message through unchanged", async () => {
    await notifyError("  The model file is missing.\n");

    expect(sendNotificationMock.mock.calls[0][0].body).toBe("The model file is missing.");
  });
});

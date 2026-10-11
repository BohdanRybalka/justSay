// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelDownload } from "../../api";
import { buildDownload, buildLocalStatus } from "../local-status-stub.test-helper";

const apiMock = {
  sttLocalStatus: vi.fn(),
  sttLocalModel: vi.fn(),
  sttLocalModelStart: vi.fn(),
};
vi.mock("../../api", () => ({ api: apiMock }));

const notifyErrorMock = vi.fn();
vi.mock("../../notify", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../notify")>();
  return { ...actual, notifyError: notifyErrorMock };
});

const { watchLocalModel } = await import("./local-model");

const MISSING = { model_downloaded: false };
const downloading = (done: number) => buildDownload({ stage: "downloading", bytes_done: done, bytes_total: 100 });

function watch() {
  const onChange = vi.fn();
  const onStatus = vi.fn();
  return { onChange, onStatus, model: watchLocalModel({ onChange, onStatus }) };
}

beforeEach(() => {
  vi.useFakeTimers();
  apiMock.sttLocalStatus.mockResolvedValue(buildLocalStatus(MISSING));
  apiMock.sttLocalModel.mockResolvedValue(buildDownload());
});

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe("watchLocalModel — reading the engine", () => {
  it("draws what a download already under way looks like on the first read, as after reopening the window", async () => {
    apiMock.sttLocalModel.mockResolvedValue(downloading(30));
    const { model, onChange, onStatus } = watch();

    model.resume();
    await vi.advanceTimersByTimeAsync(0);

    expect(model.row(false)).toEqual({ state: "downloading", percent: 30 });
    expect(onStatus).toHaveBeenCalledWith(buildLocalStatus(MISSING));
    expect(onChange).toHaveBeenCalled();
    model.release();
  });

  it("does not ask for a download from an engine that reports none", async () => {
    apiMock.sttLocalStatus.mockResolvedValue(buildLocalStatus({ ...MISSING, model_download_reported: false }));
    const { model } = watch();

    model.resume();
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.sttLocalModel).not.toHaveBeenCalled();
    expect(model.row(false).state).toBe("not-installed");
    model.release();
  });

  it("reads the status every 3 s while resumed", async () => {
    const { model } = watch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    const first = apiMock.sttLocalStatus.mock.calls.length;

    await vi.advanceTimersByTimeAsync(6000);

    expect(apiMock.sttLocalStatus.mock.calls.length).toBe(first + 2);
    model.release();
  });

  it("keeps the last download when only the download read fails", async () => {
    apiMock.sttLocalModel.mockResolvedValueOnce(downloading(30)).mockRejectedValue(new Error("down"));
    const { model } = watch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);

    await vi.advanceTimersByTimeAsync(3000);

    expect(model.row(false).state).toBe("downloading");
    model.release();
  });

  it("draws a status read that failed as a failed row", async () => {
    apiMock.sttLocalStatus.mockRejectedValue(new Error("down"));
    const { model, onStatus } = watch();

    model.resume();
    await vi.advanceTimersByTimeAsync(0);

    expect(model.row(false)).toMatchObject({ state: "failed", cause: "engine" });
    expect(onStatus).not.toHaveBeenCalled();
    model.release();
  });
});

describe("watchLocalModel — following a download", () => {
  it("reads the download every second while it runs, and the status again as soon as it ends", async () => {
    const answers: ModelDownload[] = [downloading(10), buildDownload({ stage: "done" })];
    apiMock.sttLocalModel.mockResolvedValueOnce(downloading(0)).mockImplementation(async () => answers.shift());
    const { model } = watch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    apiMock.sttLocalStatus.mockResolvedValue(buildLocalStatus());

    await vi.advanceTimersByTimeAsync(1000);
    expect(model.row(false)).toEqual({ state: "downloading", percent: 10 });
    await vi.advanceTimersByTimeAsync(1000);

    expect(model.row(false)).toEqual({ state: "installed" });
    const calls = apiMock.sttLocalModel.mock.calls.length;
    await vi.advanceTimersByTimeAsync(500);
    expect(apiMock.sttLocalModel.mock.calls.length).toBe(calls);
    model.release();
  });

  it("starts the download on install, shows it at once and follows it", async () => {
    const { model } = watch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    apiMock.sttLocalModelStart.mockResolvedValue(downloading(0));
    apiMock.sttLocalModel.mockResolvedValue(downloading(60));

    await model.install();
    expect(model.row(false)).toEqual({ state: "downloading", percent: 0 });
    await vi.advanceTimersByTimeAsync(1000);

    expect(apiMock.sttLocalModelStart).toHaveBeenCalledTimes(1);
    expect(model.row(false)).toEqual({ state: "downloading", percent: 60 });
    model.release();
  });

  it("tells the user when the download could not be started and leaves the row as it was", async () => {
    const { model } = watch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    apiMock.sttLocalModelStart.mockRejectedValue(new Error("backend down"));

    await model.install();

    expect(notifyErrorMock).toHaveBeenCalledWith("backend down");
    expect(model.row(false).state).toBe("not-installed");
    model.release();
  });
});

describe("watchLocalModel — letting go", () => {
  it("stops every read on release, though the download goes on in the backend", async () => {
    apiMock.sttLocalModel.mockResolvedValue(downloading(30));
    const { model } = watch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    model.release();
    const statusCalls = apiMock.sttLocalStatus.mock.calls.length;
    const modelCalls = apiMock.sttLocalModel.mock.calls.length;

    await vi.advanceTimersByTimeAsync(20_000);

    expect(apiMock.sttLocalStatus.mock.calls.length).toBe(statusCalls);
    expect(apiMock.sttLocalModel.mock.calls.length).toBe(modelCalls);
  });

  it("drops an answer that arrives after release", async () => {
    let answer: (status: ReturnType<typeof buildLocalStatus>) => void = () => {};
    apiMock.sttLocalStatus.mockReturnValue(new Promise((resolve) => (answer = resolve)));
    const { model, onChange } = watch();
    model.resume();
    model.release();

    answer(buildLocalStatus());
    await vi.advanceTimersByTimeAsync(0);

    expect(onChange).not.toHaveBeenCalled();
    expect(model.read()).toBeNull();
  });
});

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

    expect(model.row(false)).toEqual({ state: "installing", step: "downloading", percent: 30 });
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

    expect(model.row(false).state).toBe("installing");
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

describe("watchLocalModel — following an install", () => {
  it("reads the download every second while it runs, and the status again as soon as it ends", async () => {
    const answers: ModelDownload[] = [downloading(10), buildDownload({ stage: "done" })];
    apiMock.sttLocalModel
      .mockResolvedValueOnce(downloading(0))
      .mockImplementation(async () => answers.shift() ?? buildDownload({ stage: "done" }));
    const { model } = watch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    apiMock.sttLocalStatus.mockResolvedValue(buildLocalStatus());

    await vi.advanceTimersByTimeAsync(1000);
    expect(model.row(false)).toEqual({ state: "installing", step: "downloading", percent: 10 });
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
    expect(model.row(false)).toEqual({ state: "installing", step: "downloading", percent: 0 });
    await vi.advanceTimersByTimeAsync(1000);

    expect(apiMock.sttLocalModelStart).toHaveBeenCalledTimes(1);
    expect(model.row(false)).toEqual({ state: "installing", step: "downloading", percent: 60 });
    model.release();
  });

  it("follows the steps after the download, one read a second", async () => {
    const answers: ModelDownload[] = [
      buildDownload({ stage: "checking", bytes_done: 100, bytes_total: 100 }),
      buildDownload({ stage: "starting", bytes_done: 100, bytes_total: 100 }),
    ];
    const { model } = watch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    apiMock.sttLocalModelStart.mockResolvedValue(downloading(0));
    apiMock.sttLocalModel.mockImplementation(async () => answers.shift());
    await model.install();

    await vi.advanceTimersByTimeAsync(1000);
    expect(model.row(false)).toEqual({ state: "installing", step: "checking", percent: 100 });
    await vi.advanceTimersByTimeAsync(1000);

    expect(model.row(false)).toEqual({ state: "installing", step: "starting", percent: 100 });
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

describe("watchLocalModel — the switch to Local after an install", () => {
  const DONE = buildDownload({ stage: "done", bytes_done: 100, bytes_total: 100 });

  function watchWithSwitch() {
    const onChange = vi.fn();
    const onInstalled = vi.fn();
    return { onChange, onInstalled, model: watchLocalModel({ onChange, onInstalled }) };
  }

  async function installUntilTheEngineRuns(model: ReturnType<typeof watchLocalModel>) {
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    apiMock.sttLocalModelStart.mockResolvedValue(downloading(0));
    apiMock.sttLocalModel.mockResolvedValue(DONE);
    apiMock.sttLocalStatus.mockResolvedValue(buildLocalStatus({ model_loaded: true }));
    await model.install();
    await vi.advanceTimersByTimeAsync(1000);
  }

  it("keeps the four finished steps on show for 900 ms, then hands over to Local", async () => {
    const { model, onInstalled } = watchWithSwitch();

    await installUntilTheEngineRuns(model);
    expect(model.row(false)).toEqual({ state: "ready" });
    await vi.advanceTimersByTimeAsync(899);
    expect(onInstalled).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);

    expect(onInstalled).toHaveBeenCalledOnce();
    expect(model.row(false)).toEqual({ state: "installed" });
    model.release();
  });

  it("redraws the row without the steps before it hands over", async () => {
    const { model, onChange, onInstalled } = watchWithSwitch();
    await installUntilTheEngineRuns(model);
    onChange.mockClear();
    onInstalled.mockImplementation(() => expect(onChange).toHaveBeenCalled());

    await vi.advanceTimersByTimeAsync(900);

    expect(onInstalled).toHaveBeenCalledOnce();
    model.release();
  });

  it("does not switch after an install it did not start, as when the window was reopened midway", async () => {
    apiMock.sttLocalModel.mockResolvedValueOnce(buildDownload({ stage: "starting" })).mockResolvedValue(DONE);
    apiMock.sttLocalStatus.mockResolvedValue(buildLocalStatus({ model_loaded: true }));
    const { model, onInstalled } = watchWithSwitch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);

    await vi.advanceTimersByTimeAsync(5000);

    expect(model.row(false)).toEqual({ state: "installed" });
    expect(onInstalled).not.toHaveBeenCalled();
    model.release();
  });

  it("does not switch when the install failed, and switches when the retry succeeds", async () => {
    const { model, onInstalled } = watchWithSwitch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    apiMock.sttLocalModelStart.mockResolvedValue(downloading(0));
    apiMock.sttLocalModel.mockResolvedValue(
      buildDownload({ stage: "failed", error: "The engine did not start.", failed_step: "starting" }),
    );
    await model.install();
    await vi.advanceTimersByTimeAsync(5000);
    expect(onInstalled).not.toHaveBeenCalled();

    apiMock.sttLocalModel.mockResolvedValue(DONE);
    apiMock.sttLocalStatus.mockResolvedValue(buildLocalStatus({ model_loaded: true }));
    await model.install();
    await vi.advanceTimersByTimeAsync(1000 + 900);

    expect(onInstalled).toHaveBeenCalledOnce();
    model.release();
  });

  it("does not switch when the user chose another engine while it installed", async () => {
    const { model, onInstalled } = watchWithSwitch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    apiMock.sttLocalModelStart.mockResolvedValue(downloading(0));
    await model.install();

    model.declineSwitch();
    apiMock.sttLocalModel.mockResolvedValue(DONE);
    apiMock.sttLocalStatus.mockResolvedValue(buildLocalStatus({ model_loaded: true }));
    await vi.advanceTimersByTimeAsync(1000 + 5000);

    expect(model.row(false)).toEqual({ state: "installed" });
    expect(onInstalled).not.toHaveBeenCalled();
    model.release();
  });

  it("does not switch when the user chose another engine while the install was being started", async () => {
    const { model, onInstalled } = watchWithSwitch();
    model.resume();
    await vi.advanceTimersByTimeAsync(0);
    let answerStart: (download: ModelDownload) => void = () => {};
    apiMock.sttLocalModelStart.mockReturnValue(new Promise((resolve) => (answerStart = resolve)));
    apiMock.sttLocalModel.mockResolvedValue(DONE);
    apiMock.sttLocalStatus.mockResolvedValue(buildLocalStatus({ model_loaded: true }));
    const starting = model.install();

    model.declineSwitch();
    answerStart(downloading(0));
    await starting;
    await vi.advanceTimersByTimeAsync(1000 + 5000);

    expect(model.row(false)).toEqual({ state: "installed" });
    expect(onInstalled).not.toHaveBeenCalled();
    model.release();
  });

  it("drops the switch, and the finished steps, when the user chooses another engine in the pause", async () => {
    const { model, onInstalled } = watchWithSwitch();
    await installUntilTheEngineRuns(model);
    expect(model.row(false)).toEqual({ state: "ready" });

    model.declineSwitch();
    await vi.advanceTimersByTimeAsync(5000);

    expect(model.row(false)).toEqual({ state: "installed" });
    expect(onInstalled).not.toHaveBeenCalled();
    model.release();
  });

  it("drops the switch when the window is hidden in the pause", async () => {
    const { model, onInstalled } = watchWithSwitch();
    await installUntilTheEngineRuns(model);

    model.release();
    await vi.advanceTimersByTimeAsync(5000);

    expect(onInstalled).not.toHaveBeenCalled();
  });

  it("ignores a press on the row in the pause before the switch", async () => {
    const { model } = watchWithSwitch();
    await installUntilTheEngineRuns(model);
    const actions = { choose: vi.fn(), retry: vi.fn() };

    model.press(false, actions);

    expect(actions.choose).not.toHaveBeenCalled();
    expect(actions.retry).not.toHaveBeenCalled();
    model.release();
  });
});

describe("watchLocalModel — a press on the row", () => {
  async function pressed(statusOverrides: object, download: ModelDownload, selected = false) {
    apiMock.sttLocalStatus.mockResolvedValue(buildLocalStatus(statusOverrides));
    apiMock.sttLocalModel.mockResolvedValue(download);
    apiMock.sttLocalModelStart.mockResolvedValue(downloading(0));
    const { model } = watch();
    const actions = { choose: vi.fn(), retry: vi.fn() };
    await model.refresh();
    model.press(selected, actions);
    await vi.advanceTimersByTimeAsync(0);
    return { actions, model };
  }

  it("hands an installed model over to choose", async () => {
    const { actions } = await pressed({}, buildDownload());

    expect(actions.choose).toHaveBeenCalledOnce();
    expect(apiMock.sttLocalModelStart).not.toHaveBeenCalled();
  });

  it("installs a missing model instead of choosing it", async () => {
    const { actions } = await pressed(MISSING, buildDownload());

    expect(apiMock.sttLocalModelStart).toHaveBeenCalledOnce();
    expect(actions.choose).not.toHaveBeenCalled();
  });

  it("resumes a failed download instead of retrying the engine", async () => {
    const { actions } = await pressed(MISSING, buildDownload({ stage: "failed", error: "Interrupted." }));

    expect(apiMock.sttLocalModelStart).toHaveBeenCalledOnce();
    expect(actions.retry).not.toHaveBeenCalled();
  });

  it("runs the install again when it failed after the download", async () => {
    const failed = buildDownload({ stage: "failed", error: "The engine did not start.", failed_step: "starting" });
    const { actions } = await pressed({}, failed);

    expect(apiMock.sttLocalModelStart).toHaveBeenCalledOnce();
    expect(actions.retry).not.toHaveBeenCalled();
  });

  it("retries the engine when the engine itself failed", async () => {
    const { actions } = await pressed({ last_error: "Out of memory." }, buildDownload(), true);

    expect(actions.retry).toHaveBeenCalledOnce();
    expect(apiMock.sttLocalModelStart).not.toHaveBeenCalled();
  });

  it("does nothing while the model installs", async () => {
    const { actions } = await pressed(MISSING, downloading(5));

    expect(actions.choose).not.toHaveBeenCalled();
    expect(actions.retry).not.toHaveBeenCalled();
    expect(apiMock.sttLocalModelStart).not.toHaveBeenCalled();
  });

  it("hands a missing model on an engine without download progress to choose", async () => {
    const { actions } = await pressed({ ...MISSING, model_download_reported: false }, buildDownload());

    expect(actions.choose).toHaveBeenCalledOnce();
    expect(apiMock.sttLocalModelStart).not.toHaveBeenCalled();
  });
});

describe("watchLocalModel — an answer older than the press", () => {
  it("does not take the row back to what it was before Install was pressed", async () => {
    let answerOld: (download: ModelDownload) => void = () => {};
    apiMock.sttLocalModel
      .mockResolvedValueOnce(buildDownload())
      .mockReturnValueOnce(new Promise((resolve) => (answerOld = resolve)));
    const { model } = watch();
    await model.refresh();
    const slowRead = model.refresh();
    await vi.advanceTimersByTimeAsync(0);
    apiMock.sttLocalModelStart.mockResolvedValue(downloading(0));

    await model.install();
    answerOld(buildDownload());
    await slowRead;

    expect(model.row(false)).toEqual({ state: "installing", step: "downloading", percent: 0 });
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

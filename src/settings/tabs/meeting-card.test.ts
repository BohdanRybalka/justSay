// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";
import { EVENT_MEETING_TOGGLE } from "../../contracts";

const calls: string[] = [];

const saveSettingsMock = vi.fn();
vi.mock("../settings", () => ({ saveSettings: saveSettingsMock }));

const emitSettingsChangedMock = vi.fn();
vi.mock("./dictation", () => ({ emitSettingsChanged: emitSettingsChangedMock }));

const notifyErrorMock = vi.fn();
vi.mock("../../notify", () => ({ notifyError: notifyErrorMock }));

const emitMock = vi.fn();
vi.mock("../../event-api", () => ({ loadEventApi: async () => ({ emit: emitMock }) }));

interface LevelStream {
  onLevel: (data: { mic_db: number | null; system_db: number | null }) => void;
  onDone: () => void;
  onError: (error: string) => void;
  controller: AbortController;
}

const getMeetingStatusMock = vi.fn();
const levelStreams: LevelStream[] = [];
const meetingLevelStreamMock = vi.fn(
  (
    onLevel: LevelStream["onLevel"],
    onDone: LevelStream["onDone"],
    onError: LevelStream["onError"],
  ) => {
    const controller = new AbortController();
    levelStreams.push({ onLevel, onDone, onError, controller });
    return controller;
  },
);
vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: { ...actual.api, getMeetingStatus: getMeetingStatusMock },
    meetingLevelStream: meetingLevelStreamMock,
  };
});

const { mountMeetingCard, meetingCardHint } = await import("./meeting-card");

function status(isRecording: boolean, durationSeconds = 0) {
  return { is_recording: isRecording, duration_seconds: durationSeconds };
}

function mount(overrides: Partial<UserSettings> = {}, windowHidden = false) {
  const card = document.createElement("div");
  const mounted = mountMeetingCard(
    card,
    {
      meetings_engine: "local",
      meeting_consent_acknowledged: true,
      ...overrides,
    } as UserSettings,
    windowHidden,
  );
  const button = (id: string) => card.querySelector<HTMLButtonElement>(`#${id}`)!;
  const title = () => card.querySelector("b")!.textContent;
  return { card, mounted, button, title };
}

async function settle() {
  await vi.advanceTimersByTimeAsync(0);
}

async function mountReady(overrides: Partial<UserSettings> = {}) {
  const view = mount(overrides);
  await settle();
  return view;
}

beforeEach(() => {
  vi.useFakeTimers();
  calls.length = 0;
  levelStreams.length = 0;
  vi.clearAllMocks();
  getMeetingStatusMock.mockResolvedValue(status(false));
  saveSettingsMock.mockImplementation(async () => void calls.push("save"));
  emitSettingsChangedMock.mockImplementation(() => void calls.push("settings-changed"));
  emitMock.mockImplementation(async (event: string) => void calls.push(`emit:${event}`));
});

afterEach(() => {
  vi.useRealTimers();
});

describe("the start card", () => {
  it("offers Start recording and says where the meeting becomes text", async () => {
    const { button, title, card } = await mountReady({ meetings_engine: "local" });

    expect(title()).toBe("Record a meeting");
    expect(card.querySelector("#meeting-hint")!.textContent).toBe(
      "Your microphone and what this computer plays · turned into text on this computer",
    );
    expect(button("meeting-start").textContent).toContain("Start recording");
  });

  it("names the cloud when meetings are heard there, and follows a change of engine", async () => {
    const { mounted, card } = await mountReady({ meetings_engine: "cloud" });

    expect(card.querySelector("#meeting-hint")!.textContent).toBe(meetingCardHint("cloud"));
    mounted.setEngine("local");

    expect(card.querySelector("#meeting-hint")!.textContent).toBe(meetingCardHint("local"));
  });

  it("cannot be pressed before the backend has said whether a meeting is running", async () => {
    const { button } = mount();

    expect(button("meeting-start").disabled).toBe(true);
    button("meeting-start").click();
    await settle();

    expect(emitMock).not.toHaveBeenCalled();
    expect(button("meeting-start").disabled).toBe(false);
  });

  it("hands the start to the widget once the disclosure was acknowledged", async () => {
    const { button } = await mountReady();

    button("meeting-start").click();
    await settle();

    expect(emitMock).toHaveBeenCalledExactlyOnceWith(EVENT_MEETING_TOGGLE);
    expect(saveSettingsMock).not.toHaveBeenCalled();
  });

  it("holds the button while the widget starts, then lets go when nothing came of it", async () => {
    const { button } = await mountReady();

    button("meeting-start").click();
    await settle();
    expect(button("meeting-start").disabled).toBe(true);
    await vi.advanceTimersByTimeAsync(4000);
    expect(button("meeting-start").disabled).toBe(true);
    await vi.advanceTimersByTimeAsync(1500);

    expect(button("meeting-start").disabled).toBe(false);
  });

  it("lets go at once when the widget could not be reached", async () => {
    emitMock.mockRejectedValue(new Error("no bridge"));
    const { button } = await mountReady();

    button("meeting-start").click();
    await settle();

    expect(notifyErrorMock).toHaveBeenCalledWith("no bridge");
    expect(button("meeting-start").disabled).toBe(false);
  });
});

describe("the first start (ADR 040 obligation 3)", () => {
  async function firstStart() {
    const view = await mountReady({ meeting_consent_acknowledged: false });
    view.button("meeting-start").click();
    return view;
  }

  it("shows the disclosure instead of starting, and sends and saves nothing", async () => {
    const { title, card } = await firstStart();

    expect(title()).toBe("Before your first meeting");
    expect(card.querySelector("#meeting-consent-responsibility")!.textContent).toMatch(/responsible.*consent/is);
    expect(card.querySelector("#meeting-consent-cloud")!.textContent).toMatch(/cloud.*participants.*provider/is);
    expect(emitMock).not.toHaveBeenCalled();
    expect(saveSettingsMock).not.toHaveBeenCalled();
  });

  it("puts the start card back on Cancel, still sending and saving nothing", async () => {
    const { button, title } = await firstStart();

    button("meeting-cancel").click();

    expect(title()).toBe("Record a meeting");
    expect(emitMock).not.toHaveBeenCalled();
    expect(saveSettingsMock).not.toHaveBeenCalled();
  });

  it("saves the acknowledgement, tells the widget, and only then starts", async () => {
    const { button } = await firstStart();

    button("meeting-accept").click();
    await settle();

    expect(saveSettingsMock).toHaveBeenCalledExactlyOnceWith({ meeting_consent_acknowledged: true });
    expect(calls).toEqual(["save", "settings-changed", `emit:${EVENT_MEETING_TOGGLE}`]);
  });

  it("starts nothing when the acknowledgement could not be saved, and keeps the disclosure", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    const { button, title } = await firstStart();

    button("meeting-accept").click();
    await settle();

    expect(notifyErrorMock).toHaveBeenCalledWith("backend down");
    expect(emitMock).not.toHaveBeenCalled();
    expect(title()).toBe("Before your first meeting");
    expect(button("meeting-accept").disabled).toBe(false);
  });

  it("starts straight away on the next press once it was acknowledged", async () => {
    const { button, title } = await firstStart();
    button("meeting-accept").click();
    await settle();
    await vi.advanceTimersByTimeAsync(5500);
    emitMock.mockClear();

    button("meeting-start").click();
    await settle();

    expect(title()).toBe("Record a meeting");
    expect(emitMock).toHaveBeenCalledExactlyOnceWith(EVENT_MEETING_TOGGLE);
  });
});

describe("a meeting that is recording", () => {
  async function recording(seconds = 754) {
    getMeetingStatusMock.mockResolvedValue(status(true, seconds));
    return mountReady();
  }

  it("turns the card into the running meeting, however it was started", async () => {
    const { title, card, button } = await recording();

    expect(title()).toBe("Recording · 12:34");
    expect(card.classList.contains("meeting-card--recording")).toBe(true);
    expect(card.textContent).toContain("You");
    expect(card.textContent).toContain("Others");
    expect(button("meeting-stop").textContent).toContain("Stop");
    expect(card.querySelector("#meeting-start")).toBeNull();
  });

  it("counts the time up between reads", async () => {
    const { title } = await recording(59);

    await vi.advanceTimersByTimeAsync(2000);

    expect(title()).toBe("Recording · 1:01");
  });

  it("hands the stop to the widget and holds the button until the meeting has ended", async () => {
    const { button, title } = await recording();

    button("meeting-stop").click();
    await settle();
    expect(emitMock).toHaveBeenCalledExactlyOnceWith(EVENT_MEETING_TOGGLE);
    expect(button("meeting-stop").disabled).toBe(true);

    getMeetingStatusMock.mockResolvedValue(status(false));
    await vi.advanceTimersByTimeAsync(1000);

    expect(title()).toBe("Record a meeting");
    expect(button("meeting-start").disabled).toBe(false);
    expect(levelStreams[0].controller.signal.aborted).toBe(true);
  });

  it("draws both voices from the level stream and empties them when it ends", async () => {
    const { card } = await recording();
    const fill = (id: string) => card.querySelector<HTMLElement>(`#${id}`)!.style.width;

    levelStreams[0].onLevel({ mic_db: -30, system_db: null });
    expect([fill("meeting-level-mic"), fill("meeting-level-system")]).toEqual(["50%", "0%"]);

    levelStreams[0].onDone();
    expect([fill("meeting-level-mic"), fill("meeting-level-system")]).toEqual(["0%", "0%"]);
  });

  it("opens one level stream however long it records", async () => {
    await recording();

    await vi.advanceTimersByTimeAsync(5000);

    expect(meetingLevelStreamMock).toHaveBeenCalledOnce();
  });

  it("keeps showing the meeting when a read fails", async () => {
    const { title } = await recording();
    getMeetingStatusMock.mockRejectedValue(new Error("backend busy"));

    await vi.advanceTimersByTimeAsync(3000);

    expect(title()).toMatch(/^Recording · /);
  });
});

describe("while the window is hidden", () => {
  it("reads nothing until it is shown", async () => {
    const { mounted } = mount({}, true);

    await vi.advanceTimersByTimeAsync(3000);
    expect(getMeetingStatusMock).not.toHaveBeenCalled();

    mounted.resumeResources!();
    await settle();
    expect(getMeetingStatusMock).toHaveBeenCalledOnce();
  });

  it("stops reading and closes the level stream, and starts again on show", async () => {
    getMeetingStatusMock.mockResolvedValue(status(true, 10));
    const { mounted } = await mountReady();
    expect(levelStreams).toHaveLength(1);

    mounted.releaseResources!();
    getMeetingStatusMock.mockClear();
    await vi.advanceTimersByTimeAsync(3000);

    expect(getMeetingStatusMock).not.toHaveBeenCalled();
    expect(levelStreams[0].controller.signal.aborted).toBe(true);

    mounted.resumeResources!();
    await settle();

    expect(getMeetingStatusMock).toHaveBeenCalledOnce();
    expect(levelStreams).toHaveLength(2);
  });

  it("ignores an answer that arrives after the window was hidden", async () => {
    const { mounted, title } = await mountReady();
    let answer: (value: unknown) => void = () => {};
    getMeetingStatusMock.mockReturnValueOnce(new Promise((resolve) => (answer = resolve)));
    await vi.advanceTimersByTimeAsync(1000);

    mounted.releaseResources!();
    answer(status(true, 5));
    await settle();

    expect(title()).toBe("Record a meeting");
    expect(levelStreams).toHaveLength(0);
  });

  it("reads nothing once the page is gone", async () => {
    const { mounted } = await mountReady();

    mounted.destroy();
    getMeetingStatusMock.mockClear();
    await vi.advanceTimersByTimeAsync(3000);

    expect(getMeetingStatusMock).not.toHaveBeenCalled();
  });
});

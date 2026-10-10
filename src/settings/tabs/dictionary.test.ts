// @vitest-environment jsdom
import { readFileSync } from "fs";
import { resolve } from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";

const saveSettingsMock = vi.fn();
vi.mock("../settings", () => ({
  saveSettings: saveSettingsMock,
}));

const notifyErrorMock = vi.fn();
vi.mock("../../notify", () => ({
  notifyError: notifyErrorMock,
}));

const {
  DICTIONARY_CHAR_BUDGET,
  DICTIONARY_SEPARATORS,
  addWords,
  canAdd,
  dictionaryFull,
  parseDictionary,
  renderDictionary,
  serializeDictionary,
} = await import("./dictionary");

const GLOSSARY_PY = readFileSync(resolve(__dirname, "../../../backend/app/stt/glossary.py"), "utf-8");

function buildSettings(overrides: Partial<UserSettings> = {}): UserSettings {
  return {
    language: "uk",
    shortcut: "Ctrl+Alt+KeyV",
    output_dir: "C:/fake",
    stt_mode: "cloud",
    whisper_model_size: "large-v3-turbo",
    whisper_device: "auto",
    ollama_host: "http://localhost:11434",
    initial_prompt: "",
    gemini_api_key: "",
    groq_api_key: "",
    meeting_consent_acknowledged: false,
    meetings_engine: "local",
    meetings_language: "uk",
    theme: "system",
    display_name: "",
    paste_at_cursor: true,
    previous_language: "",
    ...overrides,
  };
}

function render(prompt = "") {
  const container = document.createElement("div");
  const group = renderDictionary(container, buildSettings({ initial_prompt: prompt }));
  const input = container.querySelector<HTMLInputElement>("#dictionary-input")!;
  const chips = () => [...container.querySelectorAll(".word-chip")].map((chip) => chip.textContent);
  const count = () => container.querySelector("#dictionary-count")!.textContent;
  const hint = container.querySelector<HTMLElement>("#dictionary-hint")!;
  const type = (text: string) => {
    input.value = text;
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
  };
  const remove = (word: string) =>
    container.querySelector<HTMLButtonElement>(`[aria-label="Remove ${word}"]`)!.click();
  return { container, group, input, chips, count, hint, type, remove };
}

const lastSavedPrompt = () => saveSettingsMock.mock.lastCall![0].initial_prompt;

beforeEach(() => {
  vi.clearAllMocks();
  saveSettingsMock.mockResolvedValue({ settings: buildSettings(), warning: null });
});

describe("the backend's glossary contract", () => {
  it("keeps the budget the Whisper engines receive", () => {
    expect(GLOSSARY_PY).toMatch(new RegExp(`^WHISPER_PROMPT_CHAR_BUDGET = ${DICTIONARY_CHAR_BUDGET}$`, "m"));
  });

  it("splits on the backend's own separators", () => {
    const backend = GLOSSARY_PY.match(/^_TERM_SEPARATORS = re\.compile\(r"(.+)"\)$/m)![1];
    expect(DICTIONARY_SEPARATORS.source).toBe(backend);
  });
});

describe("parseDictionary / serializeDictionary", () => {
  it("reads every backend separator and writes comma-joined", () => {
    const words = parseDictionary("Tauri, n8n;Rybalka\nPlaywright、Київ，Львів；Odesa");

    expect(words).toEqual(["Tauri", "n8n", "Rybalka", "Playwright", "Київ", "Львів", "Odesa"]);
    expect(serializeDictionary(words)).toBe("Tauri, n8n, Rybalka, Playwright, Київ, Львів, Odesa");
    expect(parseDictionary(serializeDictionary(words))).toEqual(words);
  });

  it("drops empty pieces and case-insensitive repeats", () => {
    expect(parseDictionary("  ,Tauri,, tauri ;  ")).toEqual(["Tauri"]);
    expect(parseDictionary("")).toEqual([]);
  });
});

describe("the limit", () => {
  it("accepts a word that lands exactly on the budget and refuses one character more", () => {
    const words = ["a".repeat(100)];
    const room = DICTIONARY_CHAR_BUDGET - 100 - ", ".length;

    expect(canAdd(words, "b".repeat(room))).toBe(true);
    expect(canAdd(words, "b".repeat(room + 1))).toBe(false);
    expect(canAdd([], "c".repeat(DICTIONARY_CHAR_BUDGET))).toBe(true);
  });

  it("is full when not even one more letter fits", () => {
    expect(dictionaryFull(["a".repeat(DICTIONARY_CHAR_BUDGET - 2)])).toBe(true);
    expect(dictionaryFull(["a".repeat(DICTIONARY_CHAR_BUDGET - 3)])).toBe(false);
    expect(dictionaryFull([])).toBe(false);
  });

  it("refuses duplicates whatever their case", () => {
    expect(canAdd(["Tauri"], "TAURI")).toBe(false);
  });

  it("adds what fits and hands back what does not", () => {
    const base = ["a".repeat(DICTIONARY_CHAR_BUDGET - 12)];

    expect(addWords(base, "tauri, bb, TAURI, toolongword")).toEqual({
      words: [...base, "tauri", "bb"],
      rejected: ["toolongword"],
    });
  });
});

describe("renderDictionary", () => {
  it("shows the stored words as chips with their count", () => {
    const view = render("Tauri, Rybalka");

    expect(view.chips()).toEqual(["Tauri", "Rybalka"]);
    expect(view.count()).toBe("2 words");
  });

  it("says one word and nothing here yet", () => {
    expect(render("Tauri").count()).toBe("1 word");
    const empty = render();
    expect(empty.count()).toBe("0 words");
    expect(empty.container.querySelector(".word-grid")!.textContent).toBe("Nothing here yet.");
  });

  it("adds on Enter, clears the input and saves the whole list", async () => {
    const view = render("Tauri");

    view.type("  Rybalka ");
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalled());

    expect(view.chips()).toEqual(["Tauri", "Rybalka"]);
    expect(view.count()).toBe("2 words");
    expect(view.input.value).toBe("");
    expect(lastSavedPrompt()).toBe("Tauri, Rybalka");
  });

  it("adds on the Add button", async () => {
    const view = render();
    view.input.value = "n8n";

    view.container.querySelector<HTMLButtonElement>("#dictionary-add")!.click();
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalled());

    expect(lastSavedPrompt()).toBe("n8n");
  });

  it("ignores a duplicate without saving", () => {
    const view = render("Tauri");

    view.type("tauri");

    expect(view.chips()).toEqual(["Tauri"]);
    expect(saveSettingsMock).not.toHaveBeenCalled();
  });

  it("removes a word with its ×", async () => {
    const view = render("Tauri, Rybalka");

    view.remove("Tauri");
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalled());

    expect(view.chips()).toEqual(["Rybalka"]);
    expect(lastSavedPrompt()).toBe("Rybalka");
  });

  it("keeps a word that does not fit in the input and says the dictionary is full", () => {
    const view = render("a".repeat(DICTIONARY_CHAR_BUDGET - 10));

    view.type("toolongword");

    expect(view.input.value).toBe("toolongword");
    expect(view.hint.hidden).toBe(false);
    expect(saveSettingsMock).not.toHaveBeenCalled();
  });

  it("disables the input with the full message once nothing more fits, and frees it on remove", async () => {
    const words = ["a".repeat(200), "b".repeat(DICTIONARY_CHAR_BUDGET - 200 - 4)];
    const view = render(serializeDictionary(words));

    expect(view.input.disabled).toBe(true);
    expect(view.input.placeholder).toBe("Your dictionary is full");

    view.remove(words[0]);
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalled());

    expect(view.input.disabled).toBe(false);
    expect(view.input.placeholder).toBe("Add a name, brand or term…");
  });

  it("puts the word back when the save fails", async () => {
    saveSettingsMock.mockRejectedValueOnce(new Error("backend down"));
    const view = render("Tauri");

    view.type("Rybalka");
    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("backend down"));

    expect(view.chips()).toEqual(["Tauri"]);
  });

  it("saves one list at a time, the second after the first answers", async () => {
    let answerFirst!: () => void;
    saveSettingsMock.mockReturnValueOnce(new Promise<void>((resolve) => (answerFirst = resolve)));
    const view = render();

    view.type("Tauri");
    view.type("Rybalka");
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalledTimes(1));
    answerFirst();
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalledTimes(2));

    expect(lastSavedPrompt()).toBe("Tauri, Rybalka");
  });

  it("ignores the Enter that confirms an input-method composition", () => {
    const view = render();
    view.input.value = "東京";

    view.input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", isComposing: true }));

    expect(view.chips()).toEqual([]);
  });

  it("shows a typed word as text, never as markup", () => {
    const view = render("<img src=x onerror=alert(1)>");

    expect(view.container.querySelector("img")).toBeNull();
    expect(view.chips()).toEqual(["<img src=x onerror=alert(1)>"]);
  });
});

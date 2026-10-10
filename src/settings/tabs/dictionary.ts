/**
 * YOUR DICTIONARY: the words the engines keep getting wrong, shown as chips and
 * saved comma-joined as the glossary. The list never grows past what a
 * Whisper-family engine actually receives, so every chip on screen is sent.
 */
import type { UserSettings } from "../../api";
import { saveSettings, type TabLifecycle } from "../settings";
import { notifyError } from "../../notify";
import { icon } from "../../ui/icons";

export const DICTIONARY_CHAR_BUDGET = 487;
export const DICTIONARY_SEPARATORS = /[,;\n、，；]/;
const DICTIONARY_JOIN = ", ";
const FULL = "Your dictionary is full";

export function parseDictionary(prompt: string): string[] {
  return prompt
    .split(DICTIONARY_SEPARATORS)
    .map((piece) => piece.trim())
    .reduce<string[]>((words, word) => (word && !hasWord(words, word) ? [...words, word] : words), []);
}

export function serializeDictionary(words: readonly string[]): string {
  return words.join(DICTIONARY_JOIN);
}

function hasWord(words: readonly string[], word: string): boolean {
  const folded = word.toLocaleLowerCase();
  return words.some((existing) => existing.toLocaleLowerCase() === folded);
}

function fits(words: readonly string[], length: number): boolean {
  const joined = serializeDictionary(words).length + (words.length ? DICTIONARY_JOIN.length : 0);
  return joined + length <= DICTIONARY_CHAR_BUDGET;
}

export function canAdd(words: readonly string[], next: string): boolean {
  return next.length > 0 && !hasWord(words, next) && fits(words, next.length);
}

export function dictionaryFull(words: readonly string[]): boolean {
  return !fits(words, 1);
}

/** `words` with every new term typed in `text`; duplicates are dropped and the
 *  terms that no longer fit come back in `rejected`. */
export function addWords(words: readonly string[], text: string): { words: string[]; rejected: string[] } {
  const rejected: string[] = [];
  const added = parseDictionary(text).reduce<string[]>((list, term) => {
    if (hasWord(list, term)) return list;
    if (canAdd(list, term)) return [...list, term];
    rejected.push(term);
    return list;
  }, [...words]);
  return { words: added, rejected };
}

function countLabel(count: number): string {
  return `${count} ${count === 1 ? "word" : "words"}`;
}

/** Adds the YOUR DICTIONARY group to the end of `container`, the chips read
 *  from the stored glossary. */
export function renderDictionary(container: HTMLElement, settings: UserSettings): TabLifecycle {
  container.insertAdjacentHTML(
    "beforeend",
    `
    <div class="group-label">${icon("book")}YOUR DICTIONARY</div>
    <div class="card"><div class="word-box">
      <div class="word-box-top"><b>Words it keeps getting wrong</b><span class="chip" id="dictionary-count"></span></div>
      <div class="word-grid" id="dictionary-words"></div>
      <div class="word-add">${icon("plus", "small")}
        <input id="dictionary-input" aria-label="Add a word" maxlength="${DICTIONARY_CHAR_BUDGET}">
        <button type="button" class="btn btn-small" id="dictionary-add">Add</button>
      </div>
      <div class="setting-row-hint">Used for dictation, meetings and files</div>
      <div class="setting-row-hint setting-row-hint--result" id="dictionary-hint" hidden>${FULL}</div>
    </div></div>
  `,
  );

  const count = container.querySelector<HTMLElement>("#dictionary-count")!;
  const grid = container.querySelector<HTMLElement>("#dictionary-words")!;
  const input = container.querySelector<HTMLInputElement>("#dictionary-input")!;
  const addButton = container.querySelector<HTMLButtonElement>("#dictionary-add")!;
  const hint = container.querySelector<HTMLElement>("#dictionary-hint")!;

  let words = parseDictionary(settings.initial_prompt);
  let confirmed = words;
  let saving = Promise.resolve();
  let destroyed = false;

  function chip(word: string): HTMLElement {
    const chipElement = document.createElement("span");
    chipElement.className = "word-chip";
    const dot = document.createElement("span");
    dot.className = "word-chip-dot";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "word-chip-remove";
    remove.setAttribute("aria-label", `Remove ${word}`);
    remove.innerHTML = icon("x");
    remove.addEventListener("click", () => commit(words.filter((existing) => existing !== word)));
    chipElement.append(dot, word, remove);
    return chipElement;
  }

  function draw() {
    count.textContent = countLabel(words.length);
    if (words.length) {
      grid.replaceChildren(...words.map(chip));
    } else {
      const empty = document.createElement("span");
      empty.className = "word-grid-empty";
      empty.textContent = "Nothing here yet.";
      grid.replaceChildren(empty);
    }
    const full = dictionaryFull(words);
    input.disabled = full;
    addButton.disabled = full;
    input.placeholder = full ? FULL : "Add a name, brand or term…";
  }

  async function persist() {
    const sending = words;
    try {
      await saveSettings({ initial_prompt: serializeDictionary(sending) });
      confirmed = sending;
    } catch (e) {
      notifyError(e instanceof Error ? e.message : String(e));
      words = confirmed;
      if (!destroyed) draw();
    }
  }

  function commit(next: string[]) {
    words = next;
    draw();
    saving = saving.then(persist);
  }

  function add() {
    const { words: next, rejected } = addWords(words, input.value);
    input.value = rejected.join(DICTIONARY_JOIN);
    hint.hidden = rejected.length === 0;
    if (next.length !== words.length) commit(next);
  }

  addButton.addEventListener("click", add);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.isComposing) add();
  });
  input.addEventListener("input", () => {
    hint.hidden = true;
  });

  draw();

  return {
    destroy: () => {
      destroyed = true;
    },
  };
}

import { confirm } from "@tauri-apps/plugin-dialog";
import { api, type HistoryEntry } from "../../api";

export function renderMetrics(container: HTMLElement): () => void {
  container.innerHTML = `
    <h2 class="tab-title">Metrics</h2>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
      <span class="value" id="metrics-count">Loading...</span>
      <button class="btn btn-danger" id="btn-clear-metrics">Clear All</button>
    </div>
    <div id="metrics-table-wrap" style="overflow-x: auto;">
      <table id="metrics-table" style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <thead>
          <tr>
            <th style="text-align: left; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border);">Time</th>
            <th style="text-align: left; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border);">Model</th>
            <th style="text-align: right; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border);" title="Tokens reported when the provider supplies them. Groq Whisper bills per audio-second instead — see the Audio column.">Usage</th>
            <th style="text-align: right; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border);">Process</th>
            <th style="text-align: right; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border);">Audio</th>
            <th style="text-align: right; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border);">Words</th>
          </tr>
        </thead>
        <tbody id="metrics-body"></tbody>
      </table>
    </div>
    <div id="metrics-empty" style="display: none; color: var(--text-muted); padding: 24px; text-align: center;">No metrics yet</div>
    <div id="metrics-load-more" style="text-align: center; padding: 12px; display: none;">
      <button class="btn btn-secondary" id="btn-load-more-metrics">Load more</button>
    </div>
  `;

  const countEl = container.querySelector<HTMLElement>("#metrics-count")!;
  const tbody = container.querySelector<HTMLElement>("#metrics-body")!;
  const emptyEl = container.querySelector<HTMLElement>("#metrics-empty")!;
  const tableWrap = container.querySelector<HTMLElement>("#metrics-table-wrap")!;
  const loadMoreWrap = container.querySelector<HTMLElement>("#metrics-load-more")!;
  const btnLoadMore = container.querySelector<HTMLButtonElement>("#btn-load-more-metrics")!;
  const btnClear = container.querySelector<HTMLButtonElement>("#btn-clear-metrics")!;

  let offset = 0;
  let total = 0;
  const LIMIT = 50;

  async function loadEntries(append = false) {
    try {
      const resp = await api.getHistory(LIMIT, offset);
      total = resp.total;
      countEl.textContent = `${total} entr${total !== 1 ? "ies" : "y"}`;

      if (!append) {
        tbody.innerHTML = "";
      }

      if (resp.entries.length === 0 && !append) {
        emptyEl.style.display = "block";
        tableWrap.style.display = "none";
      } else {
        emptyEl.style.display = "none";
        tableWrap.style.display = "block";
      }

      for (const entry of resp.entries) {
        tbody.appendChild(createRowEl(entry));
      }

      offset += resp.entries.length;
      loadMoreWrap.style.display = offset < total ? "block" : "none";
    } catch (e) {
      countEl.textContent = "Failed to load";
      console.error(e);
    }
  }

  function createRowEl(entry: HistoryEntry): HTMLElement {
    const tr = document.createElement("tr");
    tr.style.borderBottom = "1px solid var(--border)";

    const date = new Date(entry.timestamp);
    const timeStr = date.toLocaleDateString("uk-UA", {
      day: "2-digit",
      month: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });

    const cells: [string, string][] = [
      [timeStr, "padding: 7px 10px; color: var(--text-dim); white-space: nowrap;"],
      [entry.model_name ?? "—", "padding: 7px 10px; font-family: monospace; font-size: 12px; color: var(--text);"],
      [entry.tokens_used != null ? entry.tokens_used.toLocaleString("uk-UA") : "—", "padding: 7px 10px; text-align: right; color: var(--text);"],
      [`${(entry.duration_ms / 1000).toFixed(2)}s`, "padding: 7px 10px; text-align: right; color: var(--text);"],
      [entry.audio_duration_seconds != null ? `${entry.audio_duration_seconds.toFixed(1)}s` : "—", "padding: 7px 10px; text-align: right; color: var(--text);"],
      [entry.word_count != null ? entry.word_count.toString() : "—", "padding: 7px 10px; text-align: right; color: var(--text);"],
    ];

    for (const [text, style] of cells) {
      const td = document.createElement("td");
      td.setAttribute("style", style);
      td.textContent = text;
      tr.appendChild(td);
    }

    return tr;
  }

  btnLoadMore.addEventListener("click", () => loadEntries(true));

  btnClear.addEventListener("click", async () => {
    if (total === 0) return;
    btnClear.disabled = true;
    const ok = await confirm(
      `Delete all ${total} entr${total !== 1 ? "ies" : "y"}? Metrics and History share the same data — both tabs will be cleared.`,
      { title: "Clear History", kind: "warning" }
    );
    if (!ok) {
      btnClear.disabled = false;
      return;
    }
    btnClear.textContent = "Clearing...";
    try {
      await api.clearHistory();
      offset = 0;
      total = 0;
      tbody.innerHTML = "";
      emptyEl.style.display = "block";
      tableWrap.style.display = "none";
      countEl.textContent = "0 entries";
      loadMoreWrap.style.display = "none";
    } catch (e) {
      console.error(e);
    } finally {
      btnClear.disabled = false;
      btnClear.textContent = "Clear All";
    }
  });

  loadEntries();

  return () => {};
}

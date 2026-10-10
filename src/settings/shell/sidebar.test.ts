// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { backendStateOf, renderPanelSelection } from "./sidebar";

function sidebarMarkup(): HTMLElement {
  const html = readFileSync(resolve(__dirname, "../../../index.html"), "utf-8");
  return new DOMParser().parseFromString(html, "text/html").getElementById("sidebar")!;
}

describe("the sidebar markup", () => {
  it("lists Insights, History, a RECORD group with Dictation, Meetings and Files, a line, then Settings", () => {
    const rows = [...sidebarMarkup().children].flatMap((row) => {
      if (row.classList.contains("nav-label")) return [`label:${row.textContent}`];
      if (row.classList.contains("nav-gap")) return ["gap"];
      if (row instanceof HTMLElement && row.classList.contains("nav-item")) return [row.dataset.panel!];
      return [];
    });

    expect(rows).toEqual(["insights", "history", "label:RECORD", "dictation", "meetings", "files", "gap", "settings"]);
  });

  it("draws Dictation with the microphone, Meetings with the people icon and Files with the file icon", () => {
    const iconOf = (panel: string) =>
      sidebarMarkup().querySelector(`[data-panel="${panel}"] use`)!.getAttribute("href");

    expect(iconOf("dictation")).toBe("#mic");
    expect(iconOf("meetings")).toBe("#users");
    expect(iconOf("files")).toBe("#file");
  });

  it("no longer carries the Transcribe a file box under the sections", () => {
    expect(sidebarMarkup().querySelector("#transcribe-file")).toBeNull();
    expect(sidebarMarkup().textContent).not.toContain("Transcribe a file");
  });

  it("marks Meetings, and only Meetings, as the open section", () => {
    const sidebar = sidebarMarkup();

    renderPanelSelection(sidebar, "meetings");

    const current = [...sidebar.querySelectorAll<HTMLElement>('[data-panel][aria-current="true"]')];
    expect(current.map((item) => item.dataset.panel)).toEqual(["meetings"]);
  });
});

describe("backendStateOf", () => {
  it.each([
    [{ reachable: true, refusedRequest: false, starting: true }, "ready"],
    [{ reachable: true, refusedRequest: false, starting: false }, "ready"],
    [{ reachable: true, refusedRequest: true, starting: false }, "unauthorized"],
    [{ reachable: false, refusedRequest: false, starting: true }, "starting"],
    [{ reachable: false, refusedRequest: false, starting: false }, "offline"],
    [{ reachable: false, refusedRequest: true, starting: false }, "offline"],
  ])("reads %j as %s", (backend, state) => {
    expect(backendStateOf(backend)).toBe(state);
  });
});

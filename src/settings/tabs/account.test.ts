// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";

const { notifyErrorMock } = vi.hoisted(() => ({ notifyErrorMock: vi.fn(async () => {}) }));

vi.mock("../../notify", () => ({ notifyError: notifyErrorMock }));

import { renderAccount, type AccountName } from "./account";

let container: HTMLElement;
let rename: ReturnType<typeof vi.fn<(chosen: string) => Promise<void>>>;

function render(name: AccountName) {
  renderAccount(container, name, rename);
}

const nameButton = () => container.querySelector<HTMLButtonElement>("button.account-card-name")!;
const nameInput = () => container.querySelector<HTMLInputElement>(".account-card-name-input")!;
const avatar = () => container.querySelector(".avatar--large")!.textContent;

function typeName(value: string, key = "Enter") {
  nameButton().click();
  nameInput().value = value;
  nameInput().dispatchEvent(new KeyboardEvent("keydown", { key }));
}

beforeEach(() => {
  vi.clearAllMocks();
  document.body.innerHTML = "";
  container = document.createElement("div");
  document.body.append(container);
  rename = vi.fn(async () => {});
});

describe("renderAccount", () => {
  it("shows the computer's name until one is chosen", () => {
    render({ chosen: "", osName: "Bohdan Rybalka" });

    expect(nameButton().textContent).toBe("Bohdan Rybalka");
    expect(avatar()).toBe("BR");
    expect(container.textContent).toContain("On this computer.");
  });

  it("shows the chosen name over the computer's", () => {
    render({ chosen: "Богдан", osName: "Bohdan Rybalka" });

    expect(nameButton().textContent).toBe("Богдан");
    expect(avatar()).toBe("Б");
  });

  it("asks for a name when neither is known", () => {
    render({ chosen: "", osName: "" });

    expect(nameButton().textContent).toBe("Add your name");
    expect(nameButton().classList).toContain("account-card-name--empty");
    expect(avatar()).toBe("");
  });

  it("edits the name in place, starting from the one shown", () => {
    render({ chosen: "", osName: "Bohdan Rybalka" });

    nameButton().click();

    expect(nameInput().value).toBe("Bohdan Rybalka");
    expect(document.activeElement).toBe(nameInput());
  });

  it("stores a new name, trimmed, and shows it", async () => {
    render({ chosen: "", osName: "Bohdan Rybalka" });

    typeName("  Ada Lovelace ");

    expect(rename).toHaveBeenCalledWith("Ada Lovelace");
    await vi.waitFor(() => expect(nameButton().textContent).toBe("Ada Lovelace"));
    expect(avatar()).toBe("AL");
  });

  it("stores the name when the field loses focus", async () => {
    render({ chosen: "", osName: "Bohdan Rybalka" });

    nameButton().click();
    nameInput().value = "Ada";
    nameInput().dispatchEvent(new FocusEvent("blur"));

    expect(rename).toHaveBeenCalledWith("Ada");
    await vi.waitFor(() => expect(nameButton().textContent).toBe("Ada"));
  });

  it("goes back to the computer's name when the field is emptied", async () => {
    render({ chosen: "Ada", osName: "Bohdan Rybalka" });

    typeName("");

    expect(rename).toHaveBeenCalledWith("");
    await vi.waitFor(() => expect(nameButton().textContent).toBe("Bohdan Rybalka"));
  });

  it("stores nothing when the name is left as it was", () => {
    render({ chosen: "", osName: "Bohdan Rybalka" });

    typeName("Bohdan Rybalka");

    expect(rename).not.toHaveBeenCalled();
    expect(nameButton().textContent).toBe("Bohdan Rybalka");
  });

  it("drops the edit on Escape", () => {
    render({ chosen: "", osName: "Bohdan Rybalka" });

    typeName("Ada", "Escape");

    expect(rename).not.toHaveBeenCalled();
    expect(nameButton().textContent).toBe("Bohdan Rybalka");
  });

  it("keeps the old name and says why when storing fails", async () => {
    rename.mockRejectedValueOnce(new Error("backend offline"));
    render({ chosen: "", osName: "Bohdan Rybalka" });

    typeName("Ada");

    await vi.waitFor(() =>
      expect(notifyErrorMock).toHaveBeenCalledWith("Could not save your name: backend offline"),
    );
    expect(nameButton().textContent).toBe("Bohdan Rybalka");
  });
});

import { describe, expect, it } from "vitest";
import { escapeHtml } from "./html";

describe("escapeHtml — the frontend's only XSS barrier", () => {
  it("neutralises a tag written into text content", () => {
    expect(escapeHtml("<script>alert(1)</script>")).toBe(
      "&lt;script&gt;alert(1)&lt;/script&gt;",
    );
  });

  it("neutralises a payload that would break out of a double-quoted attribute", () => {
    expect(escapeHtml('" onerror="alert(1)')).toBe("&quot; onerror=&quot;alert(1)");
  });

  it("escapes the ampersand exactly once", () => {
    expect(escapeHtml("Tom & Jerry")).toBe("Tom &amp; Jerry");
    expect(escapeHtml("&amp;")).toBe("&amp;amp;");
  });

  it("replaces the ampersand first, which is what keeps the others single-escaped", () => {
    expect(escapeHtml("<b>")).toBe("&lt;b&gt;");
    expect(escapeHtml('<a href="x">&</a>')).toBe("&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;");
    expect(escapeHtml("<i>'</i>")).toBe("&lt;i&gt;&#39;&lt;/i&gt;");
  });

  it("neutralises a payload that would break out of a single-quoted attribute", () => {
    expect(escapeHtml("'")).toBe("&#39;");
    expect(escapeHtml("' onerror='alert(1)")).toBe("&#39; onerror=&#39;alert(1)");
  });

  it("leaves text with nothing to escape byte-identical", () => {
    expect(escapeHtml("Привіт світ")).toBe("Привіт світ");
    expect(escapeHtml("")).toBe("");
    expect(escapeHtml("Tom & Jerry <b>")).toBe("Tom &amp; Jerry &lt;b&gt;");
  });
});

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

  it("replaces the ampersand first, which is what keeps the other three single-escaped", () => {
    /** The ordering pin. `.replace()` runs left to right over the whole string,
     *  so an `&` rule that ran after the others would find the ampersands they
     *  had just introduced and escape those too: `<b>` would come out as
     *  `&amp;lt;b&amp;gt;`, which renders as the literal text `&lt;b&gt;`
     *  instead of an inert tag. Every call site assigns the result to
     *  `innerHTML`, so a double-escaped value is a visible defect and a
     *  single-escaped one is the whole barrier. */
    expect(escapeHtml("<b>")).toBe("&lt;b&gt;");
    expect(escapeHtml('<a href="x">&</a>')).toBe("&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;");
  });

  it("leaves text with nothing to escape byte-identical", () => {
    expect(escapeHtml("Привіт світ")).toBe("Привіт світ");
    expect(escapeHtml("")).toBe("");
  });
});

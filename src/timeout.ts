/**
 * A budget for work that may never answer.
 *
 * The transports this app sits on can stop answering without ever failing. A
 * Tauri `invoke()` has no reject channel at all (ADR 028), and the dynamic
 * `import("@tauri-apps/api/core")` in front of it is unbounded too, so a promise
 * built from either is not slow but absent — and everything sequenced after it
 * is absent as well. The HTTP requests `api.ts` gives a budget to are bounded
 * there — fourteen of its endpoints are deliberately `UNRECONCILED` and are not —
 * so what this race still covers is the bridge underneath them and any
 * caller-level unit of work assembled from several awaits.
 *
 * The budget covers the whole unit of work the caller cannot proceed without,
 * not the first half of it. Racing only the fetch and then awaiting an unbounded
 * apply leaves exactly the wedge the race was added to remove.
 *
 * Each caller owns its own budget, because they are unrelated numbers: the two
 * settings loads happen to be equal today and the widget's shell command is a
 * different order of magnitude.
 */

/** Thrown whenever a budget in this app expires, so a caller can branch on "we
 *  gave up waiting" as distinctly as it already branches on a 401 or an HTTP
 *  status (ADR 049). It carries the budget that expired and, for an HTTP call,
 *  the path that went unanswered — `subject` is `null` when the work being
 *  bounded is not addressable, such as a Tauri `invoke()`.
 *
 *  The message is built here rather than at the throw sites so both mechanisms —
 *  this module's race and `api.ts`'s `AbortController` — produce one sentence
 *  with one shape. */
export class TimedOutError extends Error {
  readonly budgetMs: number;
  readonly subject: string | null;

  constructor(budgetMs: number, subject: string | null = null) {
    const what = subject === null ? "" : ` ${subject}`;
    super(`the backend did not answer${what} within ${budgetMs / 1000} seconds`);
    this.name = "TimedOutError";
    this.budgetMs = budgetMs;
    this.subject = subject;
  }
}

export function withTimeout<T>(work: Promise<T>, budgetMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const expiry = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new TimedOutError(budgetMs)), budgetMs);
  });
  return Promise.race([work, expiry]).finally(() => clearTimeout(timer)) as Promise<T>;
}

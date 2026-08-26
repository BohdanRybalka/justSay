/**
 * A budget for work that may never answer.
 *
 * Both places that read settings — the Settings window's first paint and the
 * widget's bounded retry — sit on transports that can stop answering without
 * ever failing: an HTTP request the backend accepts and abandons, and a Tauri
 * `invoke()` that has no reject channel at all (ADR 028). A promise like that
 * is not slow, it is absent, and everything sequenced after it is absent too.
 *
 * The budget covers the whole unit of work the caller cannot proceed without,
 * not the first half of it. Racing only the fetch and then awaiting an unbounded
 * apply leaves exactly the wedge the race was added to remove.
 *
 * Each caller owns its own budget, because the two are unrelated numbers that
 * happen to be equal today.
 */
export function withTimeout<T>(work: Promise<T>, budgetMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const expiry = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`the backend did not answer within ${budgetMs / 1000} seconds`)),
      budgetMs,
    );
  });
  return Promise.race([work, expiry]).finally(() => clearTimeout(timer)) as Promise<T>;
}

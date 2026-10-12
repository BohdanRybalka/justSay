/**
 * jsdom leaves focus on a button that gets disabled; a browser drops it. Makes
 * jsdom drop it too and returns the function that puts the real behaviour back.
 */
export function dropFocusWhenDisabled(): () => void {
  const native = Object.getOwnPropertyDescriptor(HTMLButtonElement.prototype, "disabled")!;
  Object.defineProperty(HTMLButtonElement.prototype, "disabled", {
    configurable: true,
    get: native.get,
    set(this: HTMLButtonElement, disabled: boolean) {
      native.set!.call(this, false);
      if (disabled) this.blur();
      native.set!.call(this, disabled);
    },
  });
  return () => Object.defineProperty(HTMLButtonElement.prototype, "disabled", native);
}

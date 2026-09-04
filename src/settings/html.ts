export function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/**
 * The meeting-recording disclosure panel.
 *
 * ADR 040 obligation 3 and ADR 043: the product gets no consent signalling for
 * free, so it has to say plainly that the obligation sits with the user, and
 * that Cloud mode sends the other participants' voices to a third party. Both
 * statements are required — the backend refuses to start a meeting recording
 * with `403` until this has been acknowledged.
 */
export function meetingDisclosureHtml(acknowledged: boolean): string {
  return `
    <div class="setting-label">Meeting recording</div>
    <div class="setting-hint" id="meeting-consent-responsibility">
      A meeting recording captures everyone on the call, including people who never
      installed JustSay. You are responsible for obtaining whatever consent your
      jurisdiction and your employer require before you start one.
    </div>
    <div class="setting-hint" id="meeting-consent-cloud">
      In Cloud mode the other participants' audio is sent to the transcription provider
      you configured, along with your own. Switch to Local mode if none of it may leave
      this machine.
    </div>
    <div class="setting-row">
      <span class="label" id="meeting-consent-state">${
        acknowledged
          ? "You have already acknowledged this."
          : "Acknowledge this to enable meeting recording."
      }</span>
      <button class="btn btn-primary" id="btn-meeting-consent"${acknowledged ? " disabled" : ""}>${
        acknowledged ? "Acknowledged" : "I understand"
      }</button>
    </div>
  `;
}

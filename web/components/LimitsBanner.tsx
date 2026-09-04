/**
 * Persistent, non-dismissible, on every screen.
 *
 * There is no close button and no `localStorage` key that remembers a dismissal,
 * because every one of these four clauses is a limit on what the product is allowed to
 * be read as claiming, and a limit a user can turn off is a limit the product does not
 * actually have. It is the cheapest honest thing in the repo and it stays on screen at
 * 04:50 when the person reading it is tired.
 */
export function LimitsBanner() {
  return (
    <div className="limits-banner" role="note">
      Daylight only - ranking aid, not dispatch - never suppresses a camera - imagery:{" "}
      <a href="https://hpwren.ucsd.edu" rel="noreferrer noopener" target="_blank">
        HPWREN, hpwren.ucsd.edu
      </a>
    </div>
  );
}

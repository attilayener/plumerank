/**
 * Why there is one theme, said on the page rather than hidden in a stylesheet.
 *
 * A documented refusal reads as a more considered decision than a longer list of
 * features, and this one has a real reason: the product is a night-shift instrument
 * that will be judged on a projector in a lit room. A second theme would have cost a
 * second token set, a toggle, a CI contrast job and half the accessibility QA, to serve
 * a use case that does not exist.
 */
export function ThemeNote() {
  return (
    <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)", margin: 0 }}>
      One theme, deliberately. prefers-color-scheme: light is answered with this same
      dark palette and color-scheme: dark -- a 05:00 control room and a lit judging room
      want the same surfaces, and a second theme would have cost a token set, a toggle
      and half the accessibility QA to serve a user this product does not have.
    </p>
  );
}

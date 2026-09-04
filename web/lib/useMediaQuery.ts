"use client";

import { useEffect, useState } from "react";

/**
 * One media query in JavaScript, and only one, for one reason.
 *
 * The layout itself is pure CSS -- grid-template-areas swapped at 1280 px and a
 * display:none rule at 768 px -- because a JS layout is a layout that flashes wrong on
 * first paint. What CSS cannot do is choose a different DATA SET: below 1280 px the
 * wall becomes a strip of the TOP 12 tiles by rank, and rank is not something a
 * stylesheet can sort by. So this hook decides which cameras to render, never where to
 * put them.
 *
 * Starts false and corrects in an effect: `next build` prerenders the shell with no
 * window at all, and reading matchMedia during that render is the classic hydration
 * mismatch.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia(query);
    setMatches(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

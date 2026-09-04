"use client";

/**
 * GET /api/benchmark, once per page load.
 *
 * There is no polling and no revalidation: the endpoint serves a committed file from
 * disk, so it cannot change while a judge is reading the page, and a refresh loop
 * would spend the box's CPU credits on re-fetching bytes that are identical.
 */

import { useEffect, useState } from "react";
import { fetchJson, ROUTES } from "./api";
import { readBenchmark } from "./read";
import type { Benchmark } from "./types";

export interface BenchmarkState {
  data: Benchmark | null;
  loading: boolean;
  /** Set when /api/benchmark did not answer. The page then renders the designed
   *  degraded state -- it does NOT fall back to a bundled copy of the results, which
   *  would let a stale build disagree with the repo about the headline. */
  error: string | null;
}

export function useBenchmark(): BenchmarkState {
  const [data, setData] = useState<Benchmark | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchJson<unknown>(ROUTES.benchmark, controller.signal)
      .then((doc) => {
        setData(readBenchmark(doc));
        setError(null);
      })
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  return { data, loading, error };
}

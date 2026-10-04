import type { Metadata } from "next";

export const metadata: Metadata = { title: "For judges" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}

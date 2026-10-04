// Copies the newest shipped-configuration DEV payload (results/arm_dev_<YYYY-MM-DD>.json,
// no cell suffix) into lib/generated/dev-run.json, so /results can show the development
// evidence -- labelled in-sample -- before the held-out run exists. The copy is committed
// so `tsc` and a clean-clone build work even without ../results; this refreshes it.
import { copyFileSync, existsSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const results = join(web, "..", "results");
const out = join(web, "lib", "generated", "dev-run.json");

if (!existsSync(results)) {
  console.log("sync-dev-run: ../results not present, keeping the committed copy");
  process.exit(0);
}
const shipped = readdirSync(results)
  .filter((n) => /^arm_dev_\d{4}-\d{2}-\d{2}\.json$/.test(n))
  .sort();
if (shipped.length === 0) {
  console.log("sync-dev-run: no shipped DEV payload in ../results, keeping the committed copy");
  process.exit(0);
}
copyFileSync(join(results, shipped.at(-1)), out);
console.log(`sync-dev-run: ${shipped.at(-1)} -> lib/generated/dev-run.json`);

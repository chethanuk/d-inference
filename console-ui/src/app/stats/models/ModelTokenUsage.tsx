import type { CatalogAliasSummary } from "@/lib/stats-model-filter";
import { formatCompactNumber } from "../format";
import type { PlatformStats } from "../platform-types";
import { aliasMemberBuilds } from "./model-inventory";

interface ModelTokenRow { id: string; total: number }

// Alias builds (desired, previous and retired) fold into the alias id: the
// window can include traffic served by a build that has since rotated out.
function tokenRows(stats: PlatformStats, aliases: CatalogAliasSummary[]): ModelTokenRow[] {
  const aliasByBuild = new Map<string, string>();
  for (const alias of aliases) {
    for (const build of aliasMemberBuilds(alias)) aliasByBuild.set(build, alias.id);
  }
  const totals = new Map<string, number>();
  for (const bucket of stats.tokens_by_model ?? []) {
    const id = aliasByBuild.get(bucket.model) ?? bucket.model;
    totals.set(id, (totals.get(id) ?? 0) + bucket.total_tokens);
  }
  return [...totals].map(([id, total]) => ({ id, total })).sort((a, b) => b.total - a.total || a.id.localeCompare(b.id));
}

export function ModelTokenUsage({ stats, aliases }: { stats: PlatformStats; aliases: CatalogAliasSummary[] }) {
  const available = stats.tokens_by_model_status === "available" && Array.isArray(stats.tokens_by_model);
  const rows = available ? tokenRows(stats, aliases) : [];
  const max = rows[0]?.total ?? 0;

  let content;
  if (!available) {
    content = <p className="px-4 py-6 text-sm text-text-secondary sm:px-6">Token usage by model is temporarily unavailable.</p>;
  } else if (rows.length === 0) {
    content = <p className="px-4 py-6 text-sm text-text-secondary sm:px-6">No token usage in the last 24 hours.</p>;
  } else {
    content = (
      <ol aria-labelledby="model-tokens-title" className="space-y-4 px-4 py-6 sm:px-6">
        {rows.map((row) => (
          <li key={row.id}>
            <div className="flex items-baseline justify-between gap-4 text-sm">
              <span className="min-w-0 truncate text-text-primary" title={row.id}>{row.id}</span>
              <span className="shrink-0 tabular-nums text-text-secondary">{formatCompactNumber(row.total)}<span className="sr-only"> tokens</span></span>
            </div>
            <div aria-hidden="true" className="mt-2 h-2 overflow-hidden rounded-full bg-bg-secondary">
              <div className="h-full rounded-full bg-accent-brand" style={{ width: `${max ? (row.total * 100) / max : 0}%` }} />
            </div>
          </li>
        ))}
      </ol>
    );
  }

  return (
    <section className="border-t border-border-dim pt-8" aria-labelledby="model-tokens-title">
      <div className="mb-6">
        <h2 id="model-tokens-title" className="text-lg font-medium text-text-primary">Tokens by model</h2>
        <p className="mt-1 text-sm text-text-secondary">Prompt and completion tokens served per model over the last 24 hours.</p>
      </div>
      <div className="overflow-hidden rounded-2xl border border-border-dim bg-bg-white">{content}</div>
    </section>
  );
}

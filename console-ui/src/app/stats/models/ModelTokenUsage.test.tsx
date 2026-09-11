import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { CatalogAliasSummary } from "@/lib/stats-model-filter";
import type { ModelTokenBucket, PlatformStats } from "../platform-types";
import { ModelTokenUsage } from "./ModelTokenUsage";

const base: PlatformStats = { total_requests: 0, total_prompt_tokens: 0, total_completion_tokens: 0, total_tokens: 0, avg_tokens_per_request: 0, active_providers: 0, total_gpu_cores: 0, total_cpu_cores: 0, total_memory_gb: 0, total_bandwidth_gbs: 0, network_capacity_tps: 0, providers: [], models: [], time_series: [] };
const row = (model: string, prompt: number, completion: number, requests = 1): ModelTokenBucket => ({ model, requests, prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion });
const gemma: CatalogAliasSummary = { id: "gemma-4-26b", desiredBuild: "gemma-4-26b-qat-4bit", previousBuild: "gemma-4-26b-8bit", retiredBuilds: ["gemma-4-26b-4bit"] };
const available = (rows: ModelTokenBucket[]): PlatformStats => ({ ...base, tokens_by_model_status: "available", tokens_by_model: rows });

function renderedRows() {
  return within(screen.getByRole("list", { name: "Tokens by model" })).getAllByRole("listitem").map((item) => ({
    id: item.querySelector("[title]")?.getAttribute("title"),
    text: item.textContent,
    bar: item.querySelector<HTMLElement>("[aria-hidden='true'] > div")?.style.width,
    barHidden: item.querySelector("[aria-hidden='true']") !== null,
  }));
}

describe("Tokens by model", () => {
  it.each([
    {
      name: "orders builds by total tokens with the largest bar full",
      stats: available([row("EigenLabs/Qwen3.8-27B-4bit-mtp", 900, 300), row("gemma-4-26b-qat-4bit", 1_500_000, 500_000), row("gemma-4-26b-8bit", 300_000, 200_000)]),
      aliases: [],
      want: [{ id: "gemma-4-26b-qat-4bit", total: "2M", bar: "100%" }, { id: "gemma-4-26b-8bit", total: "500K", bar: "25%" }, { id: "EigenLabs/Qwen3.8-27B-4bit-mtp", total: "1.2K", bar: "0.06%" }],
    },
    {
      name: "folds desired, previous and retired alias builds into one row",
      stats: available([row("gemma-4-26b-qat-4bit", 600, 0), row("gemma-4-26b-8bit", 300, 0), row("gemma-4-26b-4bit", 100, 0), row("EigenLabs/Qwen3.8-27B-4bit-mtp", 500, 0)]),
      aliases: [gemma],
      want: [{ id: "gemma-4-26b", total: "1K", bar: "100%" }, { id: "EigenLabs/Qwen3.8-27B-4bit-mtp", total: "500", bar: "50%" }],
    },
    {
      name: "keeps build rows when no alias metadata is loaded",
      stats: available([row("gemma-4-26b-qat-4bit", 600, 0), row("gemma-4-26b-8bit", 300, 0)]),
      aliases: [],
      want: [{ id: "gemma-4-26b-qat-4bit", total: "600", bar: "100%" }, { id: "gemma-4-26b-8bit", total: "300", bar: "50%" }],
    },
    {
      name: "draws empty bars when every model has zero tokens",
      stats: available([row("model-a", 0, 0, 2), row("model-b", 0, 0)]),
      aliases: [],
      want: [{ id: "model-a", total: "0", bar: "0%" }, { id: "model-b", total: "0", bar: "0%" }],
    },
  ])("$name", ({ stats, aliases, want }) => {
    render(<ModelTokenUsage stats={stats} aliases={aliases} />);
    const rows = renderedRows();
    expect(rows.map(({ id }) => id)).toEqual(want.map(({ id }) => id));
    rows.forEach((rendered, index) => {
      expect(rendered.text).toContain(`${want[index].total} tokens`);
      expect(rendered.bar).toBe(want[index].bar);
      expect(rendered.barHidden).toBe(true);
      expect(rendered.text).not.toContain("NaN");
    });
  });

  it.each([
    { name: "an empty window", stats: available([]), text: "No token usage in the last 24 hours." },
    { name: "an unavailable aggregate", stats: { ...base, tokens_by_model_status: "unavailable" as const, tokens_by_model: null }, text: "Token usage by model is temporarily unavailable." },
    { name: "a coordinator without the field", stats: base, text: "Token usage by model is temporarily unavailable." },
  ])("keeps the heading and explains $name", ({ stats, text }) => {
    render(<ModelTokenUsage stats={stats} aliases={[]} />);
    expect(screen.getByRole("heading", { name: "Tokens by model" })).toBeInTheDocument();
    expect(screen.getByText(text)).toBeInTheDocument();
    expect(screen.queryByRole("listitem")).not.toBeInTheDocument();
  });
});

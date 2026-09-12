package store

import (
	"fmt"
	"reflect"
	"testing"
	"time"
)

// TestUsageTokensByModel: the /v1/stats per-model token panel groups the
// last window's usage by the served build (usage.model), largest first,
// capped at the top 50, on both store implementations.
func TestUsageTokensByModel(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			record := func(model, publicModel string, prompt, completion int) {
				s.RecordUsageFullWithPublicModel("p", "c", "", model, publicModel, uniqueID("req"), prompt, completion, 0, nil)
			}
			since := time.Now().Add(-time.Hour)

			t.Run("sums by served build, aliases and legacy rows under model", func(t *testing.T) {
				record("gemma-4-26b-qat-4bit", "gemma-4-26b", 100, 50)
				record("gemma-4-26b-qat-4bit", "gemma-4-26b", 200, 25)
				// Alias traffic routed to the rollback build lands under that build.
				record("gemma-4-26b-8bit", "gemma-4-26b", 40, 10)
				// A zero-token row is still one request.
				record("gemma-4-26b-8bit", "gemma-4-26b", 0, 0)
				// Legacy rows written without a public model.
				s.RecordUsageFull("p", "c", "", "EigenLabs/Qwen3.8-27B-4bit-mtp", uniqueID("req"), 7, 3, 0, nil)
				// Rows without a served model are not attributable.
				record("", "gemma-4-26b", 999, 999)

				got, err := s.UsageTokensByModel(since)
				if err != nil {
					t.Fatal(err)
				}
				want := []UsageTokensByModelBucket{
					{Model: "gemma-4-26b-qat-4bit", Requests: 2, PromptTokens: 300, CompletionTokens: 75},
					{Model: "gemma-4-26b-8bit", Requests: 2, PromptTokens: 40, CompletionTokens: 10},
					{Model: "EigenLabs/Qwen3.8-27B-4bit-mtp", Requests: 1, PromptTokens: 7, CompletionTokens: 3},
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("UsageTokensByModel = %+v, want %+v", got, want)
				}
			})

			t.Run("cutoff after every row returns none", func(t *testing.T) {
				got, err := s.UsageTokensByModel(time.Now().Add(time.Hour))
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != 0 {
					t.Fatalf("future cutoff returned %d buckets, want 0", len(got))
				}
			})

			t.Run("keeps the 50 largest builds", func(t *testing.T) {
				for i := 1; i <= 51; i++ {
					record(fmt.Sprintf("cap-%02d", i), "", 1000+i, 0)
				}
				got, err := s.UsageTokensByModel(since)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != 50 {
					t.Fatalf("got %d buckets, want the top 50", len(got))
				}
				for _, b := range got {
					if b.Model == "cap-01" {
						t.Fatal("smallest build kept past the cap")
					}
				}
				if got[0].Model != "cap-51" {
					t.Fatalf("largest build = %q, want cap-51", got[0].Model)
				}
			})
		})
	}
}

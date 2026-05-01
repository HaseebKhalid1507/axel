# Research Fidelity Review — Axel Search Implementation
**Reviewer:** Zero  
**Date:** 2026-05-01  
**Scope:** `velocirag/src/search.rs`, `axel/src/consolidate/strengthen.rs`  
**Against:** `search-enhancement-papers-2026-05-01.md`

---

## Executive Summary

Seven papers were cited. Five were implemented as described features. Two were applied
only conceptually. The implementations range from *faithful and well-adapted* (MMR,
Ebbinghaus decay) to *plausible but divergent* (query expansion, recency boost) to
*partially invented* (spreading activation framing vs. actual Huang et al.). No
implementation is flatly wrong, but several carry calibration risks or misrepresent the
source paper's mechanism. The Ebbinghaus parameterization is the most defensible. The
query expansion is the highest-risk component.

---

## 1. MMR — Goldstein & Carbonell (1998)

### Formula Fidelity

The canonical Carbonell formula is:

```
MMR = arg max_{d_i ∈ R\S} [ λ · Sim₁(d_i, q) − (1−λ) · max_{d_j ∈ S} Sim₂(d_i, d_j) ]
```

The implementation (lines 526):

```rust
let mmr = LAMBDA * rel - (1.0 - LAMBDA) * max_sim;
```

**This is correct.** The greedy iterative selection loop, the seeding with the
highest-scoring candidate, and the max-similarity-to-selected computation all match the
original algorithm exactly. The cosine similarity function is correctly implemented
(dot product / (‖a‖·‖b‖)), including the zero-vector guard.

### Relevance Normalization

The implementation normalizes RRF scores to [0,1] before feeding them into MMR:

```rust
let rel = ((cand.rrf_score - min_rel) / span) as f32;
```

This is a *sound addition not in the original paper*. Carbonell's paper assumes
scores are already in a common range. Without normalization, the λ term and (1−λ) term
would operate on incommensurable scales (RRF scores vs. cosine similarities), making λ
meaningless as a balance parameter. The normalization is the right engineering call.

### Is λ=0.7 Reasonable?

**Yes, and it's the literature's standard default.** Carbonell's own experiments used
λ∈{0.5, 0.7, 1.0}. λ=0.7 is the canonical "prefer relevance, moderate diversity"
setting and appears as the default in virtually every downstream MMR implementation
(LangChain, Haystack, etc.). For a personal knowledge base where the user expects the
most relevant result first and diversity as a secondary benefit, 0.7 is exactly right.

A lower λ (e.g. 0.5) would make sense only if the corpus is known to have extreme
redundancy and the user benefits more from breadth. A higher λ (e.g. 0.9) would
effectively disable the diversity penalty. **λ=0.7 is correct.**

### Gap

MMR only activates when `fused.len() > opts.limit`. If RRF returns exactly `limit`
results, MMR is skipped entirely even though some of those results might be near-
duplicates. The guard should be `fused.len() >= opts.limit` — or better, MMR should
always run over the full fused candidate set regardless of count, trimming to `limit`
as its output. Low-traffic queries will silently bypass diversity reranking.

**Verdict: ✅ Formula correct. ✅ λ reasonable. ⚠️ Activation guard is off-by-one.**

---

## 2. Ebbinghaus Forgetting Curve — Murre & Dros (2015)

### Formula Fidelity

Murre & Dros replicated Ebbinghaus's 1885 data and confirmed the exponential forgetting
curve:

```
R(t) = e^(−t/S)
```

where `t` is elapsed time and `S` is the stability parameter.

**strengthen.rs** (lines 162–167):

```rust
const S_BASE: f64 = 30.0;
let stability = S_BASE * (1.0 + (1.0 + access_count as f64).ln());
let retention = (-days_inactive / stability).exp();
let new_excitability = (excitability * retention).max(EXCITABILITY_FLOOR);
```

**search.rs** (lines 322–323):

```rust
let decay_factor = (-days_since_access / 60.0).exp().max(0.7);
```

### strengthen.rs — Calibration

The exponential kernel `e^(−t/S)` is faithful. The stability scaling
`S = 30 * (1 + ln(1 + n))` is a reasonable adaptation: stability grows logarithmically
with access count, which mirrors the spacing effect literature (each retrieval adds
progressively less stability).

Concrete values:
| access_count | stability (days) | retention at 30d | retention at 90d |
|---|---|---|---|
| 0 | 30.0 | 0.368 | 0.050 |
| 1 | 51.6 | 0.560 | 0.174 |
| 5 | 82.7 | 0.696 | 0.337 |
| 20 | 120.2 | 0.778 | 0.472 |
| 100 | 168.3 | 0.836 | 0.584 |

**Is S_BASE=30 sensible?** For a personal knowledge base, a zero-access document
(S=30 days) reaches 37% retention at 30 days and effectively disappears (~5%) by 90
days. That is *aggressive* for a static notes corpus where documents don't expire
naturally. A user who imports 500 notes and doesn't access most of them for two months
will watch their excitability floor out across the board, potentially burying valid
cold-start knowledge.

A more conservative baseline — **S_BASE=60 to 90 days** — would be more appropriate
for a personal PKM where dormant-but-valid knowledge should persist. S_BASE=30 is
closer to what Ebbinghaus observed for *nonsense syllables* in memory experiments. Long-
form knowledge notes are not nonsense syllables; they have semantic hooks that aid
recall. The parameter is tunable but the default is too aggressive.

**search.rs decay_factor** uses a hardcoded half-life of 60 days with a floor of 0.7.
This is independent of the strengthen.rs formula and creates two separate, uncoordinated
decay systems. The real-time boost in search uses a 60-day characteristic time while the
consolidation phase uses a dynamic stability that starts at 30 days. These will drift
out of sync and produce non-monotonic excitability behavior for low-access documents.
Either one decay formula should govern both, or the search-time decay should read
`stability` from the database rather than hardcoding 60.

### Murre & Dros Attribution

The paper is correctly cited. One nuance: Murre & Dros (2015) is a *replication* paper;
the underlying model is Ebbinghaus (1885). The actual stability-vs-repetition scaling
used here is closer to **Wozniak & Gorzelańczyk (1994)** (the SM-2 algorithm, the basis
of SuperMemo/Anki), which explicitly models growing inter-repetition intervals. Murre &
Dros confirm the exponential kernel but do not define a stability-accumulation formula.
The implementation's attribution is partially inaccurate — it should credit Ebbinghaus
for the curve and SM-2/Anki literature for the stability growth model.

**Verdict: ✅ Kernel correct. ⚠️ S_BASE=30 too aggressive for PKM. ⚠️ Two uncoordinated decay systems. ⚠️ Attribution is partially misplaced.**

---

## 3. Graph Boost — Huang, Chen & Zeng (2004)

### What the Paper Actually Describes

Huang et al. describe *spreading activation* through an *associative network*: a node
activated by a query fires activation along all its edges, which propagates transitively
(multiple hops) with attenuation. The key properties are:

1. Multi-hop propagation (energy fans out across the graph)
2. Attenuation at each hop (typically multiplicative decay)
3. The final activation of a node is the *sum* of all activation paths reaching it

### What Was Implemented

The implementation does **one hop only** from top-5 scored documents:

```rust
let graph_score = parent_score * weight.clamp(0.0, 1.0) * GRAPH_BOOST_FACTOR;
*boost_accum.entry(neighbor_id).or_insert(0.0) += graph_score;
```

This is *direct neighbor boosting*, not spreading activation. It:
- ✅ Propagates relevance from high-scoring nodes to neighbors (correct concept)
- ✅ Accumulates boosts from multiple parents (correct additive model)
- ❌ Does not propagate beyond depth-1 neighbors
- ❌ Does not attenuate across hops (the paper attenuates per hop)
- ❌ Only fires from top-5 parents, not all activated nodes

This is a sound *approximation* of spreading activation that trades fidelity for
O(1-hop) query time. It's a defensible engineering tradeoff. But calling it "spreading
activation (Huang et al.)" in comments and docs overstates fidelity. It is better
described as *neighbor-based co-retrieval boosting*.

### Is GRAPH_BOOST_FACTOR=0.3 Too High or Low?

The boost formula is:

```
graph_score = parent_rrf_score × edge_weight × 0.3
```

`parent_rrf_score` is a raw RRF score. RRF scores with k=60 for a top-ranked document
over 3 lists are roughly:
```
1/(60+1) + 1/(60+1) + 1/(60+1) ≈ 0.049
```

So graph_score for a weight=1.0 neighbor of the top-ranked doc ≈ `0.049 × 1.0 × 0.3 ≈
0.015`. Since the RRF scores for documents ranked 5th–10th are in the
`0.020–0.035` range, a weight=1.0 neighbor of the top-ranked doc can jump past documents
ranked 5–10 in the fused list. This is **plausible but potentially over-powered** for
weak edges (e.g., a `co_retrieved` edge from a single coincidental co-access).

A safer parametrization: 0.15–0.20 for GRAPH_BOOST_FACTOR, combined with a minimum
edge weight threshold (e.g., `weight >= 0.3`) before the boost fires. This prevents
a single noisy co-retrieval from significantly reranking results.

**Verdict: ⚠️ Mechanism is neighbor boosting, not true spreading activation. ⚠️ GRAPH_BOOST_FACTOR=0.3 is on the high side for low-confidence edges. ✅ Additive accumulation is conceptually correct.**

---

## 4. Query Expansion — Gao et al. (2023) / Pseudo-Relevance Feedback

### What HyDE Actually Does

HyDE (Hypothetical Document Embeddings) is an *embedding-level* technique: the LLM
generates a hypothetical document that answers the query, embeds *that* document, and
uses the hypothetical document's embedding as the query vector. It never touches BM25 or
appends text terms.

### What Was Implemented

The implementation is **Pseudo-Relevance Feedback (PRF)**, not HyDE:

```rust
let expansion_terms: Vec<String> = extract_top_terms(top_content, 3)
    .into_iter()
    .filter(|t| !query_lower.contains(t.as_str()))
    .collect();
// ...
format!("{} {}", query, expansion_terms.join(" "))
```

This is classic Rocchio/PRF: take the top result, extract its most frequent terms, and
append them to the query. PRF has been in information retrieval since the 1970s. HyDE
is an entirely different mechanism. The paper credit is **wrong** — this should cite
Rocchio (1971) or Manning et al.'s IR textbook, not Gao et al. (2023).

This is the most significant attribution error in the implementation.

### Does Query Expansion Risk Degrading Precision?

**Yes, and this is the highest-risk component in the system.**

PRF degrades precision in two well-documented failure modes:

**1. Topic drift** — If the top vector result is on the right topic but emphasizes a
sub-topic the user didn't intend, the expansion terms steer the BM25 layer toward that
sub-topic. Example: query "memory consolidation" → top result is about hippocampal
replay → expansion appends "hippocampal replay theta oscillations" → BM25 retrieves
neuroscience papers instead of software architecture notes.

**2. Vocabulary explosion** — `extract_top_terms` uses raw frequency counting with no
TF-IDF weighting. High-frequency terms in the top document are not necessarily the most
discriminative terms for the query. A document about "project planning" will surface
terms like "project", "tasks", "timeline" — all low-discrimination terms that pollute
the BM25 index match.

**Mitigations not present in the implementation:**
- No relevance score threshold before expansion fires (it fires on the top result
  regardless of how confident that result is)
- No IDF weighting in `extract_top_terms` — raw frequency only
- No expansion term count limit relative to query length (3 terms on a 1-word query
  triples the query mass)
- Expansion only filtered for exact query substring match (line 186) — not semantic
  overlap

A minimum threshold check (e.g., only expand if `results_lists[0][0].score > 0.75`)
and IDF-weighted term selection would substantially reduce the precision risk.

**Verdict: ❌ Attribution to Gao et al. is incorrect — this is classic PRF, not HyDE. ⚠️ No confidence gate before expansion fires. ⚠️ Raw frequency extraction risks low-precision term selection. The mechanism works but misrepresents its lineage.**

---

## 5. Recency Boost — Covington et al. (2016)

### What the Paper Describes

The YouTube DNN paper (Section 4.1) notes that *age of the training example* is an
important feature, represented as the log of the video upload time relative to training
time. The insight is that recency at *training* time matters; the model learns to
up-rank newer content because freshness correlates with engagement.

### What Was Implemented

```rust
let recency = (1.0 + 1.0 / (days_since_indexed + 1.0)).ln().min(0.7);
let effective = stored_excitability * decay_factor + recency * 0.05;
```

The function shape is reasonable: logarithmic, today ≈ 0.69, 7 days ≈ 0.29, 30 days
≈ 0.10. The 0.05 weight keeps it mild. However:

- The Covington paper's recency feature is a *training-time* feature for a neural
  model, not a scalar score multiplier in a ranking function. The *concept* that recency
  matters is drawn from Covington; the *implementation* is an independent design choice.
- More precisely relevant citations would be: **Elsas & Dumais (2010)** "Leveraging
  temporal dynamics of document content for improved search" (WSDM) or **Li & Croft
  (2003)** "Time-based language models" (CIKM), which directly model recency in
  retrieval ranking.
- The log formula `ln(1 + 1/(d+1))` is a reasonable choice but the effective weight
  (0.05) is small enough that this signal barely moves scores. At 0 days: 0.69 × 0.05 =
  0.034 additive. For a document with excitability=0.5, this inflates effective
  excitability by ~6%. That's probably fine as a mild freshness nudge.

**Verdict: ✅ The implementation is sound. ⚠️ Attribution to Covington is a stretch — the formula is independent engineering. Better cites exist for retrieval-time recency.**

---

## 6. Prioritized Experience Replay — Schaul et al. (2015)

### Implementation

The notes say: *"Ebbinghaus stability already handles this implicitly"*. This is
intellectually defensible but architecturally lazy. The paper's actual mechanism
(priority queues weighted by TD-error, importance-sampling correction weights) has no
direct analogue in the current system. The mapping is:

| PER concept | Axel analogue |
|---|---|
| TD-error (surprise) | Implicit in access frequency |
| Priority queue | Ebbinghaus stability sort |
| IS correction weights | Not implemented |

The PER paper's *actual insight* — that the learning signal from rare high-surprise
experiences is disproportionately valuable — could inform which documents to inject into
the boot context. High-surprise documents (recently changed, low excitability but high
access) should be prioritized for context injection. This is not currently implemented.

**Verdict: ⚠️ Listed as implemented but is conceptual only. The analogy is coherent but the mechanism is not implemented.**

---

## 7. LLM Agent Survey — Wang et al. (2024)

The hot-document injection in `boot_context` is a reasonable operationalization of
the paper's context window management challenge. This is an engineering application of
a survey finding rather than an algorithm implementation, so strict formula-fidelity
isn't the right lens. The threshold of 0.6 excitability for "hot" is arbitrary but
reasonable.

**Verdict: ✅ Appropriate applied interpretation of a survey paper.**

---

## Calibration Summary Table

| Component | Paper | Formula Correct | Param Reasonable | Risk |
|---|---|---|---|---|
| MMR | Carbonell 1998 | ✅ Yes | ✅ λ=0.7 correct | Low — activation guard off-by-one |
| Ebbinghaus decay (consolidation) | Murre & Dros 2015 | ✅ Yes | ⚠️ S_BASE=30 too aggressive | Medium — cold docs decay too fast |
| Ebbinghaus decay (search) | Murre & Dros 2015 | ✅ Yes | ⚠️ 60d hardcoded, uncoordinated | Medium — diverges from consolidation |
| Graph boost | Huang et al. 2004 | ⚠️ 1-hop only | ⚠️ 0.3 high for weak edges | Medium — noisy edges over-rank |
| Query expansion | Gao et al. 2023 | ❌ Wrong paper (PRF not HyDE) | ⚠️ No confidence gate | High — precision degradation |
| Recency boost | Covington 2016 | ⚠️ Different mechanism | ✅ Weight 0.05 conservative | Low |
| PER | Schaul 2015 | ❌ Not implemented | N/A | None (not active) |

---

## Would Different Papers Have Been Better Choices?

### For Query Expansion

**Lavrenko & Croft (2001) — "Relevance-Based Language Models" (SIGIR, 1,800+ cites)**
is the correct citation for PRF in the dense/hybrid retrieval context. Their relevance
model explicitly computes P(w|R) — the probability of a term given the relevant set —
giving IDF-weighted expansion that doesn't suffer from raw-frequency bias.

If the intention was truly to use embedding-space expansion (closer to HyDE), the
correct lightweight implementation would be: embed the top result, average with the
query embedding, use the averaged vector for a second HNSW pass. No LLM required; no
text manipulation required.

### For Graph Spreading Activation

**Page et al. (1999) — PageRank** or **Balmin et al. (2004) — "A Framework for
Semantic Link Analysis"** would be cleaner citations for the one-hop neighbor boosting
actually implemented. Huang et al. specifically study *multi-hop* associative retrieval,
and citing them for a one-hop boost misrepresents the technique.

### For Ebbinghaus Stability Growth

**Wozniak & Gorzelańczyk (1994)** (SM-2 algorithm) or the **FSRS algorithm (Ye 2022)**
are the canonical citations for stability-scaling-with-repetition models. They provide
empirically validated constants. The current `S = 30 * (1 + ln(1 + n))` formula is
invented — reasonable in shape but without empirical grounding.

---

## Priority Fixes

**P0 — Query expansion confidence gate:**
```rust
// Only expand if top result has strong relevance
let should_expand = results_lists[0][0].score > 0.75;
```

**P1 — MMR activation guard:**
```rust
// Was: fused.len() > opts.limit
// Fix:
if fused.len() >= opts.limit {  // or: always run MMR, trim to limit
```

**P2 — Unify decay constants:**
The search-time decay hardcode of 60 days and the consolidation S_BASE=30 should be
unified under a single `STABILITY_BASE_DAYS` constant, ideally loaded from config. Two
independent decay systems operating on the same excitability field will produce
incoherent behavior.

**P3 — S_BASE recalibration:**
Raise S_BASE from 30 to 60–90 for a PKM corpus. Alternatively, add a
`min_excitability_age_days` threshold so new documents are protected from decay for
their first 30 days regardless of access.

**P4 — Graph boost edge threshold:**
```rust
// Only boost from edges with sufficient weight
if weight >= 0.3 {
    let graph_score = parent_score * weight * GRAPH_BOOST_FACTOR;
    // ...
}
```

---

*The architecture is coherent and the research selection is thematically appropriate.
The gaps are in calibration and attribution precision — not in the fundamental design.
Fix the query expansion confidence gate first: that is the only component with a clear
path to user-visible precision regression.*

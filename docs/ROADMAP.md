# Axel Consolidation — Roadmap

## What's Shipped (S183)
All 15 components from the original spec are live. See CONSOLIDATION.md for status table.

## What's Next (ranked by impact)

### 1. Retrieval-Triggered Re-embedding
**Impact: High | Effort: Medium**

When a document is retrieved but scores poorly across multiple searches (the "misaligned embedding" signal), automatically re-embed it with fresh context. The current prune phase flags these documents but doesn't act. The biological analog: reconsolidation doesn't just strengthen — it can also restructure the memory trace.

Implementation: In strengthen phase, when `extinction_signals` fires and the doc's embedding is >30 days old, re-embed the content. Requires calling `search.index_document()` during strengthen — currently it only updates excitability.

### 2. Context Prefix Embeddings (Pattern Separation)
**Impact: High | Effort: High (one-time re-embed of all 2,240 docs)**

Prepend `[source:mikoshi][path:3. Learning/...]` to document content before embedding. This pushes similar documents from different contexts apart in vector space — mimicking the grid cell orthogonalization from Moser et al. (2015).

Current problem: two session journal entries about debugging have nearly identical embeddings, causing retrieval confusion. Context prefixes would separate them.

Requires: one-time full re-embed (~6 minutes), modify `index_document()` to prepend context.

### 3. Temporal Decay Curve for Excitability ✅ SHIPPED
**Impact: Medium | Effort: Low**

~~Currently excitability only changes during consolidation runs (every 6 hours).~~ 
DONE: Search now applies a continuous 1%/day temporal decay factor to excitability
based on `last_accessed`. Documents untouched for weeks gradually lose their boost
even between consolidation runs. Capped at 80% of stored value.

### 4. Co-Retrieval Graph Visualization
**Impact: Medium | Effort: Medium**

`axel graph` command that shows the most connected document clusters based on co-retrieval edges. Outputs a textual graph or exports to DOT format for Graphviz rendering. Would make the Hebbian wiring visible.

### 5. Consolidation Health Alerts ✅ SHIPPED
**Impact: Medium | Effort: Low**

DONE: `axel stats` now checks last consolidation time and warns if >12 hours stale.
Also warns if no runs recorded or if excitability distribution is flat.

### 6. Session-Aware Access Logging
**Impact: Medium | Effort: Low**

Currently `document_access.session_id` is always NULL. Wire the actual session ID through so consolidation can track which sessions generated which access patterns. This enables "session replay" — understanding what the agent was thinking about during each session.

### 7. Forgetting Curve Integration
**Impact: Low | Effort: Medium**

Implement Ebbinghaus-style forgetting curves for memories. Each memory's confidence/importance should decay according to a spaced-repetition schedule unless reinforced by retrieval. This is more sophisticated than the current linear decay and better matches actual memory science.

### 8. Multi-Brain Consolidation
**Impact: Low | Effort: High**

Support consolidating across multiple `.r8` files — e.g., sharing high-excitability documents between Jawz's brain and a project-specific brain. The biological analog: system consolidation transfers memories between hippocampus and neocortex. Different brains could serve as different "memory systems."

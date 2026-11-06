---
id: 20261106-openalex-anonymous-search-is-rate-limite-94eb
type: failure
title: OpenAlex anonymous search is rate limited
tags: [literature, openalex, tooling]
citations: []
confidence: 0.45
support: 1
contradictions: 1
uses: 3
created: 2026-11-06T16:45:28+00:00
updated: 2026-11-06T16:45:28+00:00
---

# OpenAlex anonymous search is rate limited

Anonymous OpenAlex queries return HTTP 429 under load. Use Europe PMC or Crossref instead, or set OPENALEX_API_KEY.

- 2026-11-06 **contradicted**: OpenAlex worked fine today without a key for 20 queries.

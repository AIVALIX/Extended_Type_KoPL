# Pipeline Evaluation Summary

## Test Configuration
- **Sample Size**: n=100 (random sampling)
- **Pipelines Tested**: Extended Type-KoPL, SAFE, KGT
- **Knowledge Graphs**: PrimeKGQA, MetaQA
- **Note**: Extended Type-KoPL was tested without relation information (type info only)

---

## PrimeKGQA Results

| Pipeline | 1-hop | 2-hop | 2-intersection | 3-intersection | Average |
|----------|-------|-------|----------------|----------------|---------|
| Extended Type-KoPL | **90.0%** | **38.0%** | 41.0% | 36.0% | 51.2% |
| SAFE | 87.0% | 25.0% | - | - | 56.0% |
| KGT | 84.0% | 0.0% | **75.0%** | **100.0%** | **64.8%** |

### Observations - PrimeKGQA
- **Extended Type-KoPL**: Best performance on 1-hop (90%) and 2-hop (38%) queries
- **SAFE**: Only supports 1-hop and 2-hop queries; competitive on 1-hop (87%)
- **KGT**: Excels at intersection queries (75% on 2-intersection, 100% on 3-intersection), but fails on 2-hop (0%)

---

## MetaQA Results

| Pipeline | 1-hop | 2-hop | 3-hop | Average |
|----------|-------|-------|-------|---------|
| Extended Type-KoPL | 87.0% | **92.0%** | **96.0%** | **91.7%** |
| SAFE | **100.0%** | 86.0% | - | 93.0% |
| KGT | 93.0% | 53.0% | 5.0% | 50.3% |

### Observations - MetaQA
- **Extended Type-KoPL**: Strong multi-hop performance (2-hop: 92%, 3-hop: 96%)
- **SAFE**: Perfect 1-hop accuracy (100%), but only supports up to 2-hop
- **KGT**: Good 1-hop (93%), but degrades significantly on multi-hop queries (3-hop: 5%)

---

## Summary

### Best Pipeline by Query Type

| Query Type | PrimeKGQA | MetaQA |
|------------|-----------|--------|
| 1-hop | Extended Type-KoPL (90%) | SAFE (100%) |
| 2-hop | Extended Type-KoPL (38%) | Extended Type-KoPL (92%) |
| 3-hop | - | Extended Type-KoPL (96%) |
| 2-intersection | KGT (75%) | - |
| 3-intersection | KGT (100%) | - |

### Key Findings

1. **Extended Type-KoPL** is the most versatile pipeline:
   - Handles all query types
   - Best for multi-hop sequential queries (especially on MetaQA)
   - Works well without relation information when type info is available

2. **SAFE** is excellent for simple queries:
   - Perfect 1-hop accuracy on MetaQA
   - Limited to 1-hop and 2-hop queries only

3. **KGT** specializes in intersection queries:
   - Exceptional performance on intersection queries (PrimeKGQA)
   - Poor performance on sequential multi-hop queries (especially 3-hop)

### Recommendations

- For **general-purpose KGQA**: Use Extended Type-KoPL
- For **simple 1-hop queries**: SAFE provides highest accuracy
- For **intersection queries**: KGT is the best choice
- For **multi-hop reasoning** (2-hop, 3-hop): Extended Type-KoPL significantly outperforms others

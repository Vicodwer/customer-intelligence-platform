# Monitoring Report

## ML drift monitoring

- Reference rows: 1000
- Current rows: 1000
- Drift threshold: 0.2
- Drift detected: True
- Drifted feature count: 3

### Drifted features

- age (numeric): 0.8998
- campaign (numeric): 6.2047
- month (categorical): 9.0759

### Recommended action

Review feature shift and consider retraining trigger.


## RAG monitoring

- Request count: 4
- Retrieval hit rate: 75.00%
- Empty retrieval count: 1
- Refusal count: 1
- Refusal rate: 25.00%
- Average top-k score: 0.3319
- Average latency ms: 101.56
- Average estimated answer tokens: 68.25

### RAG monitoring interpretation

The RAG monitor checks whether retrieval is returning evidence, whether the refusal rule is active, and whether latency remains acceptable for a small local MVP.

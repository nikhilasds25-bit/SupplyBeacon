# Graph-Based Supply Chain Risk Prediction

This module computes a deterministic risk baseline from the existing ElectroGraph-AI supply chain graph.

## Inputs

- `data/graph/supply_chain_graph.graphml`
- `data/processed/business_impacts/business_impact_events.csv`

The graph contains suppliers, brands, products, warehouses, categories, locations, and event nodes.
Risk features are derived from graph structure and event exposure.

## Output

- `data/risk_prediction/risk_scores.csv`
- `data/risk_prediction/risk_scores.json`
- `data/risk_prediction/risk_summary.json`

Each scored node includes a risk score, risk level, event exposure metrics, connected supply chain counts, and a human-readable reason.

## Feature engineering

Features are computed from the graph using:

- direct and indirect event exposure
- graph connectivity (degree)
- connected product, supplier, warehouse, and location counts
- maximum connected event severity
- average connected event confidence
- connected business impact risk score from event nodes

Event proximity is measured by shortest path distance in the undirected graph.
Nodes within 1 to 4 hops of an event accumulate exposure.

## Risk scoring formula

The baseline risk score is calculated using fixed weights:

- severity: 30%
- confidence: 20%
- business impact risk: 20%
- event exposure: 15%
- degree connectivity: 10%
- product connectivity: 5%

Scores are clamped to the range [0.0, 1.0].

## Risk thresholds

- Critical: `score >= 0.85`
- High: `score >= 0.60`
- Medium: `score >= 0.30`
- Low: otherwise

## Unknown-data handling

If an event has `Unknown` location or `Unknown` affected entity, the graph builder does not create derived relationships.
Those events are recorded as unresolved and counted in the summary.
Nodes with no event exposure receive low baseline risk.

## Limitations

- This baseline is deterministic and rule-based.
- It does not use machine learning or graph neural networks.
- It depends on the existing graph relationships and the available event nodes.
- It does not invent missing locations, entities, or relationships.

## Future extension

This module establishes a stable foundation that can later be replaced with a graph neural network or graph transformer.
A future model could use the same node and edge features while learning more complex patterns from the dynamic supply chain graph.

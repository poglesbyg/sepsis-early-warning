### What each feature block buys

With all 345 features the model scores AUROC 0.8206. **No single block costs more than 0.0080 AUROC to remove** — the most expensive is `recency` (hours since each channel was last measured). Yet every block scores between 0.720 and 0.782 on its own. That combination has one explanation: the matrix is enormously redundant, and the same physiology is reachable through several different encodings of it.

The sharpest number here is the ordering-only row. Using **109 features that contain no measured value whatsoever** — only which channel was sampled, how recently, and how often — the model reaches AUROC **0.7943**, or 97% of what the full matrix achieves. Nothing about the patient's physiology is in that subset. It is a record of what the care team chose to look at, and it is nearly as predictive as the measurements themselves.

Two readings follow. The optimistic one: missingness is signal, and the recency and intensity blocks earn their place rather than padding the matrix. The uncomfortable one: a model this dependent on ordering behaviour is partly learning clinical suspicion rather than physiology, so it would degrade wherever ordering habits differ — which is exactly what the drop from hospital A to hospital B looks like.

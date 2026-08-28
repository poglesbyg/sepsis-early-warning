### MICU to SICU: a shift this cohort can actually support

Trained on 2,727 medical ICU admissions and evaluated at a threshold frozen on held-out MICU patients (0.445), the model scores AUROC **0.8046** (95% CI 0.7794 to 0.8348) on MICU admissions it has not seen, and **0.7810** (0.7477 to 0.8183) on surgical ICU admissions: a drop of +0.0236, and the two intervals overlap, so discrimination is not measurably worse across the unit boundary.

Clinical utility tells a harsher story: 0.4579 at home against **0.0419** in the SICU. Most of that is not the model getting worse at ranking patients. The septic rate is 10.8% in the MICU and 4.0% in the SICU, and an alert threshold tuned where sepsis is common fires far too often where it is rare. Retuning the threshold on the SICU alone recovers +0.2573 to 0.2992 — that recovery is the price of shipping one operating point across a boundary the units do not share, and it is available in production by re-picking a threshold, without retraining.

The third bucket is the one it would be convenient to omit: 8,090 admissions where neither unit indicator was recorded — more than either named unit — scoring AUROC 0.7849 at 10.4% prevalence. Dropping them would have made the transfer claim cleaner and the cohort unrepresentative of the hospital it came from.

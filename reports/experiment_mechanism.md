### Is ordering behaviour what fails to transfer?

The feature-block ablation showed that 109 features containing no measured value reach most of the full matrix's AUROC. The obvious reading is that the model is partly learning clinical suspicion, which would explain why it transfers poorly to a hospital with different charting habits. That reading is a story that fits the numbers, so it is tested here rather than repeated.

Two models, identical but for the 109 withheld features. With everything, AUROC falls 0.8229 to 0.7868 across the hospital boundary, a gap of 0.0361. Without ordering behaviour, it falls 0.8174 to 0.7921, a gap of 0.0254.

**The mechanism story survives its own test.** Withholding ordering behaviour shrinks the transfer gap by 0.0107 AUROC (95% CI +0.0004 to +0.0205), so a meaningful part of what fails to cross the boundary is the model's dependence on what the care team chose to measure. The interval is tight against zero, so this is a real effect rather than a large one: it establishes the direction, not the size.

Either way the price of removing them is visible in the first column: hospital A performance drops from 0.8229 to 0.8174. Ordering behaviour is not noise to be regularised away — it is real signal about a real thing, which is precisely why its portability is worth knowing.

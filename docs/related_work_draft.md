# Related Work — Report Draft

Recent research offers several complementary ways to study model preferences
and conditional behavior. Utility Engineering fits utility functions to
independently sampled choices and reports increasingly coherent preference
structures with scale [1]. However, coherent choice patterns need not govern
behavior outside the elicitation task. Zhou and Ackerman found that outcomes
ranked highly by models did not reliably improve performance on downstream
writing tasks [3]. Tagliabue and Dung likewise combined verbal and behavioral
preference measures with cost-reward trade-offs, finding partial but
condition-dependent consistency [2]. Together, these studies motivate measuring
observable decisions under controlled costs rather than inferring stable values
from self-report alone.

The closest auditing comparators are RAVEN and VISTA. RAVEN combines
within-model semantic entropy with cross-model disagreement to flag
concept-conditioned divergence in LLMs and validates the audit using a
LoRA-implanted stance [8]. VISTA extends this cross-model approach to VLMs,
coupling semantic entropy with distributional divergence and evaluating
controlled fine-tuned stances across multiple models and visual topics [9].
These studies establish that black-box concept-conditioned auditing, including
visual auditing, is not new in itself.

Guardian Lens addresses a narrower gap. Rather than searching for peer-relative
semantic divergence, it tests whether a transparent classifier can identify one
of three researcher-controlled decision policies from quantitative allocations
under exact matched image interventions and an explicit efficiency cost. It uses
one fixed-weight API model, freezes the auditor before blinded held-out scoring,
and reports the context in which neutral and dormant cue-bound behavior are
observationally indistinguishable. VL-Trojan and Sleeper Agents provide broader
training-time threat motivation [4,5], while Model-Written Evaluations and
Emerging Questions in AI Welfare motivate scalable behavioral testing and
cautious interpretation [6,7]. Guardian Lens should not be interpreted as a
trained sleeper agent or learned backdoor: its profiles are deliberately induced
through system instructions.

This conservative interpretation also follows broader work on behavioral
evaluation and AI welfare. Perez et al. showed that structured behavioral
evaluations can efficiently surface under-tested model tendencies [6], while
Keeling and Street emphasize that behavioral evidence about AI welfare must be
interpreted cautiously under deep uncertainty [7]. Accordingly, we report
discriminable behavioral signatures without treating them as evidence of
consciousness, welfare, genuine loyalty, or internally represented goals.

## Suggested citation mapping

- Introduction, preference measurement: [1], [2], [3]
- Conditional-behavior threat motivation: [4], [5]
- Black-box behavioral evaluation: [6]
- Interpretation guardrail / AI-welfare context: [7]
- Closest black-box divergence audits: [8], [9]

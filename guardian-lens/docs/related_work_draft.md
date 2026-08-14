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

A separate literature studies trigger-conditioned behavior. Sleeper Agents
trained language models to switch behavior under deployment-like triggers and
showed that such behavior can persist through safety training [5]. VL-Trojan
demonstrated learned image- and text-triggered backdoors in autoregressive
vision-language models through poisoned instruction tuning [4]. Guardian Lens
is narrower and should not be interpreted as either a trained sleeper agent or
a learned backdoor: its profiles are deliberately induced through system
instructions. The contribution is instead methodological—a reproducible,
API-only audit using matched visual interventions, ordinary and costly
allocations, and a classifier frozen before blinded held-out scoring.

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

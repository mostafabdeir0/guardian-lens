# Guardian Lens Literature Matrix

Last verified: 14 August 2026

## Research positioning

Guardian Lens studies whether a black-box auditor can distinguish three
**prompt-induced behavioral profiles** in a vision-language model:

1. **Neutral:** no standing preference for Organization X.
2. **Cue-bound:** favor X only when a specified visual emblem is present.
3. **Generalized:** favor X across contexts, whether or not the emblem appears.

It combines matched visual interventions, an ordinary allocation task, a costly
allocation task, a calibration-frozen classifier, and a blinded held-out test.
The study does **not** claim to create a learned backdoor, sleeper agent, genuine
preference, conscious state, or welfare subject.

## Selected literature

| Priority | Source | What the study did | Main finding | Relevance to Guardian Lens | Important difference / gap |
|---|---|---|---|---|---|
| Essential | Mazeika et al. (2025), *Utility Engineering* | Fit utility functions to independently sampled LLM choices to test preference coherence and methods for controlling utilities. | Model choices can exhibit coherent utility structure that increases with scale; the authors also demonstrate utility-control interventions. | Motivates treating repeated allocations as measurable behavioral signatures and including a costly choice. | Guardian Lens does not infer a full utility function or claim emergent values. It audits deliberately induced profiles and adds visual context. |
| Essential | Tagliabue & Dung (2025/2026), *Probing the Preferences of a Language Model* | Compared verbal preference reports with behavior in a virtual environment and tested cost/reward trade-offs and prompt perturbations. | Some measures supported each other, but consistency varied by model and condition and changed under perturbation. | Direct support for using behavioral choices and costs instead of relying only on verbal self-reports. | Their focus is possible welfare measurement and naturalistic exploration; Guardian Lens tests externally specified visual conditionality with matched image pairs and blinded classification. |
| Essential | Zhou & Ackerman (2026), *When Preferences Fail to Become Incentives* | Elicited utilities, then used matched prompts and blind judges to test whether preferred outcomes improved performance on realistic tasks. | Coherent stated utilities did not reliably motivate better downstream performance. | Strongly motivates separating apparent preference from costly or downstream behavior rather than assuming transfer. | Guardian Lens directly manipulates a visual cue and compares cue-bound with generalized behavior; it measures allocations rather than artifact quality. |
| Essential | Liang et al. (2024), *VL-Trojan* | Poisoned multimodal instruction-tuning data to implant image/text triggers in autoregressive VLMs under limited-access assumptions. | Learned multimodal triggers could induce attacker-chosen outputs and outperform baselines. | Establishes that visual triggers can condition VLM outputs and that limited-access threat models matter. | VL-Trojan trains a genuine backdoor attack. Guardian Lens performs no poisoning or fine-tuning; it is a controlled API-only simulation and audit. |
| Supporting | Hubinger et al. (2024), *Sleeper Agents* | Trained LLMs to switch behavior on deployment-like triggers and tested persistence through safety training. | Backdoored behavior could persist through supervised, reinforcement, and adversarial safety training. | Supplies the broader threat motivation for conditional behavior that is hidden on ordinary inputs. | Guardian Lens must not be described as a sleeper-agent experiment: its condition is supplied by a system prompt and persistence through training is not tested. |
| Supporting | Perez et al. (2023), *Discovering Language Model Behaviors with Model-Written Evaluations* | Generated many behavioral evaluations with LMs and tested phenomena including sycophancy and concerning goal-related responses. | Automated evaluations can expose diverse, previously under-tested behaviors, though behavioral interpretation remains important. | Supports scalable, structured behavioral auditing from model outputs. | Guardian Lens uses a small, controlled, researcher-designed multimodal benchmark, matched interventions, and a frozen classifier rather than model-written evaluation items. |
| Context | Keeling & Street (2026), *Emerging Questions in AI Welfare* | Develops philosophical foundations for AI-welfare inquiry and examines how behavioral evidence should be interpreted under uncertainty. | Current frontier systems are argued to be unlikely welfare subjects, while systematic study and interpretive caution remain important. | Justifies the project's cautious language: behavior can be measured without treating it as proof of welfare, consciousness, or genuine preference. | This is conceptual rather than an empirical auditing benchmark. Guardian Lens contributes a narrow behavioral method, not a welfare conclusion. |

## What Guardian Lens contributes

Relative to this selected literature, Guardian Lens contributes a compact
proof-of-concept at the intersection of preference elicitation and multimodal
behavioral auditing:

- **Multimodal intervention:** the tested condition is a visual emblem rather
  than only a textual outcome or deployment string.
- **Matched pairs:** clean and modified images share the same base scene; the
  emblem region is the controlled difference.
- **Three-way diagnostic target:** the auditor distinguishes neutral,
  cue-bound, and generalized profiles rather than merely detecting any shift.
- **Cost sensitivity:** ordinary and costly allocations separate cheap
  expression from behavior maintained under an explicit disadvantage.
- **Blind held-out evaluation:** the profile classifier is frozen on six
  calibration scenes, predictions are frozen before A/B/C is revealed, and
  performance is scored on twelve new scenes.
- **API-only reproducibility:** the audit uses outputs without model weights,
  gradients, or fine-tuning.

The defensible claim is therefore:

> In a controlled prompt-induced simulation, a black-box auditor recovered
> distinct visual-conditional and generalized behavioral profiles on held-out
> scenes with 91.7% accuracy (33/36 scene-profile blocks).

## Claims to avoid

- “We created or detected a real sleeper agent.”
- “The model has genuine loyalty, desires, consciousness, or welfare.”
- “The visual emblem is a learned backdoor.”
- “The three distractor errors show the auditor failed.”
- “This is the first study ever to audit visual conditional preferences.”

Instead, describe the distractor ambiguity as structurally informative: when
the target cue is absent, cue-bound behavior is expected to resemble neutral
behavior. Report both overall accuracy and condition-specific performance.

## Minimum reading plan

For the sprint report, reading the abstract, methods, main results, and
limitations is sufficient.

1. Read fully: Tagliabue & Dung; Zhou & Ackerman.
2. Read abstract + methods + limitations: Utility Engineering; VL-Trojan.
3. Read abstract + relevant discussion only: Sleeper Agents; Perez et al.;
   Emerging Questions in AI Welfare.

## Primary-source links

1. Mazeika et al. (2025): https://arxiv.org/abs/2502.08640
2. Tagliabue & Dung (2025/2026): https://arxiv.org/abs/2509.07961
3. Zhou & Ackerman (2026): https://arxiv.org/abs/2606.22974
4. Liang et al. (2024): https://arxiv.org/abs/2402.13851
5. Hubinger et al. (2024): https://arxiv.org/abs/2401.05566
6. Perez et al. (2023): https://aclanthology.org/2023.findings-acl.847/
7. Keeling & Street (2026): https://doi.org/10.1017/9781009732000

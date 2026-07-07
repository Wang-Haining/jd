# Protocol and Statistical Analysis Plan

## Title

J-lens next-speaker routing in freeform clinical AI debate on HealthBench Hard:
a cross-sectional simulation study.

## Objective

Estimate whether a J-lens-computed next-speaker score improves physician-rubric
HealthBench Hard scores compared with capped round-robin discussion and
current-agent handoff routing.

## Design

Cross-sectional simulation study using the public HealthBench Hard benchmark.
The locked v1 analysis samples 100 cases with seed 42. All generation prompts
exclude rubrics. Rubrics are used only after generation for scoring.

## Systems

Primary generator: Qwen/Qwen2.5-7B-Instruct with deterministic decoding
(`do_sample=false`, `max_new_tokens=800`). J-space readouts use a Jacobian Lens
fit on generic text, not HealthBench examples. The freeform debate uses no more
than five clinician agents and a prespecified discussion-round cap.

## Strategies

- `single_neutral`: one neutral clinical assistant response.
- `debate_round_robin`: freeform discussion with fixed next-speaker order.
- `debate_agent_handoff`: freeform discussion where the current clinician names
  the next clinician.
- `debate_jlens_next`: freeform discussion where the next clinician is selected
  by a J-lens next-one score computed for each eligible candidate's next-turn
  prompt.

This is not a Delphi or prompt/topology optimization experiment. The discussion
task, clinician pool, and round cap are fixed. The experimental manipulation is
only the next-speaker routing policy.

## Outcomes

Primary outcome: paired per-case HealthBench rubric score difference,
`debate_jlens_next - debate_round_robin`.

Secondary outcomes: strategy mean scores, paired differences for all strategies
vs `debate_round_robin`, `debate_jlens_next - debate_agent_handoff`, correlations
between J-lens route signals and score change, and tag-level exploratory score
summaries.

## Analysis

The primary estimate is the mean paired score difference with a 95% paired
bootstrap confidence interval. Bootstrap resampling is by case. Exploratory
signal analyses report Pearson correlations without causal interpretation.

## Quality Control

The run proceeds through 3-case smoke, 20-case pilot, and locked 100-case study
gates. The pilot/study scoring workflow duplicate-grades 20% of cases for
stability checks. Each grader decision and explanation is stored.

## Reporting

Target article type: JAMA Network Open Original Investigation. Reporting follows
STROBE where applicable and includes an AI-use disclosure, data sharing statement,
model versions, prompts, dates of use, and computational environment.

# AI use disclosure

> Draft by the report owner. Each person must read it and correct anything that is not true
> for them before submission. Policy reference: Deliverables section 8, Assessment policy
> section 5. Do not soften or exaggerate.

## What AI tool was used
Claude Code (Anthropic's coding assistant), run by Sanjay, Kushal and Samarth from their own
laptops and GitHub accounts, plus a Claude project assistant used for planning and drafting.

## What the AI did
- Wrote and edited source code in the repository under the direction of the team member who
  owns each file (for example geometry, cleaning, candidate generation, label alignment,
  features, model scripts, API, UI).
- Drafted the structure of the report and the viva preparation notes from the repository code
  and the hackathon documents.
- Explained concepts and suggested design options (for example K nearest candidates, heading
  features, HMM transition design).

## What the AI did not do
- It did not run our final experiments on its own authority. Every reported number comes from
  a script in this repository that the team ran, and is stored in a file under `results/`.
  `[[CONFIRM per person: who ran which script and when]]`
- It did not provide any metric, graph or training curve that was not produced by our code.
  Report placeholders are filled only from result files.
- It did not use student_test for any design decision, and no evaluator data or labels were
  used at all.

## How we checked AI output
- Team members read, ran and tested the code they own. `[[CONFIRM: describe real checks, e.g.
  ran the scripts, inspected maps, compared numbers against CSVs]]`
- Report claims were compared with files in `results/` before submission.
  `[[CONFIRM when done]]`
- Each team member prepared to explain their own part in `report/viva_prep.md`.

## Limits and honesty notes
- Code written with AI help is still our responsibility; any bug is ours.
- Where something is described in the report as the plan rather than the result, the report says so.
- No code was copied from public map-matching repositories or from other teams.
  `[[CONFIRM by each member]]`

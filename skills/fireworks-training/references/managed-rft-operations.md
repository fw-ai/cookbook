# Managed RFT: existing jobs only (removed)

> **Managed RFT has been removed.** It accepts no new jobs, and existing jobs can't
> run, resume, cancel, or update. Reinforcement learning runs on the Training API:
> read `references/rl-async.md` and
> [Cookbook: Reinforcement Learning](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/rl.md).

Use this reference only to answer questions about managed RFT jobs that already exist.

## What still works

| Action | Who | How |
|---|---|---|
| List or read a job | The user | `firectl rftj list`, `firectl rftj get <job-id> -o json`, or the dashboard |
| Delete a job | The user | `firectl rftj delete <job-id>` |
| Create, resume, cancel, or update | Nobody | Rejected for every account |

The REST API and the Python SDK (`client.reinforcement_fine_tuning_jobs`) reject managed
RFT calls, so an agent can't read these jobs itself. Ask the user to run the `firectl`
command or check the dashboard, and to share what they see. Deleting a job is the
user's decision.

## Job states

Every managed RFT job has finished or is being deleted:

| State | Meaning |
|---|---|
| `COMPLETED` | The run finished. |
| `FAILED` | The run failed, or it hadn't finished when managed RFT was removed. Those jobs carry a status message saying so. |
| `CANCELLED` | The user stopped the run. |
| `EARLY_STOPPED` | The run stopped because it wasn't improving. |
| `DELETING` | A delete is in progress. |

To keep training on the same task, plan a new Training API run with
`references/training-api.md`.

## Related

- Reinforcement learning on the Training API: `references/rl-async.md`, `references/rl-agentic.md`
- Recipes, rewards, and deployment shapes: `references/training-api.md`

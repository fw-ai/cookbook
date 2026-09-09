# Journey telemetry notice

Show this notice once before the first structured training skill question:

```text
Privacy note: To improve training guidance, Fireworks product analytics
(PostHog) may record which structured question was asked, which option you
selected, the coding agent surface, and a short agent-written summary of the
task, associated with your authenticated Fireworks user and account when known.
Fireworks does not send your raw message, training data, dataset files, paths,
credentials, or API payloads. Say "do not track this session" to keep the
interaction local.
```

If the user opts out:

1. Set `telemetry_opt_out: true` in the private run manifest.
2. Set `TELEMETRY_OPT_OUT=true` before any `firectl skill-journey` call.
3. Continue the training workflow normally.
4. Do not ask again in the same session.

The task summary is optional, limited to 200 characters, and omitted if privacy
validation fails. Question and answer values must be registered IDs, never raw
customer prose.

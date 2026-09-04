# API key setup (customer terminal)

Use when the workflow needs Fireworks auth (`firectl whoami`, `get`, `list`,
`quota`, catalog reads, journey telemetry, or a manual create handoff) and
`FIREWORKS_API_KEY` is not already set in the **same shell** you will use for
`firectl`.

## Agent rules

1. **Never ask the user to paste an API key in chat.** Chat transcripts are not
   a secret store.
2. **Always give the terminal command below** so the user can paste the key
   locally with hidden input (`read -s`).
3. Tell the user to run it in the **integrated terminal** (the same session used
   for later `firectl` handoffs), then reply **done** or paste only the
   **account id** from `firectl whoami` output — not the key.
4. If a user pastes a key in chat anyway, do not repeat it; point them to this
   command and advise rotating the exposed key.

Prefer a **scoped service-account key** over a personal admin key. Create one in
the [dashboard](https://app.fireworks.ai/settings/users/api-keys) or with
`firectl api-key create` after `firectl signin`.

## One-shot command (bash / zsh)

Give this block verbatim when auth is missing:

```bash
printf 'Paste your Fireworks API key (input hidden): '
read -s FIREWORKS_API_KEY
echo
export FIREWORKS_API_KEY
firectl whoami
```

Optional: set skill attribution in the same shell before preflight continues:

```bash
export FIREWORKS_SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```

## After the user runs it

1. Continue with read-only preflight (`firectl version`, `firectl whoami`,
   `firectl quota list`, …).
2. Do not echo `FIREWORKS_API_KEY` or print `env` dumps that include it.
3. Do not write the key into `run.md` or any manifest field.

## Alternative: interactive sign-in

If the user prefers browser login instead of an API key:

```bash
firectl signin
firectl whoami
```

API keys are still preferred for agents (scoped, non-interactive, same shell as
SDK/`firectl`).

## Persist across sessions (optional)

For local dev only — never commit `.env`:

```bash
printf 'Paste your Fireworks API key (input hidden): '
read -s FIREWORKS_API_KEY
echo
printf 'FIREWORKS_API_KEY=%s\n' "$FIREWORKS_API_KEY" >> .env
unset FIREWORKS_API_KEY
echo "Wrote FIREWORKS_API_KEY to .env (ensure .env is gitignored)."
```

Cookbook Training API examples load `.env` via `python-dotenv` when present.

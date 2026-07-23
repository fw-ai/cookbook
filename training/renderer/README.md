# Training renderers

## Backward compatibility

Managed Training jobs may persist and reuse concrete renderer names across
retries and resumes. Consider existing jobs before changing or removing a
renderer registration or altering its rendering behavior. When corrected
semantics would change emitted tokens, register a new concrete name and point
the capability registry at it; leave the old name's implementation intact.

Only Managed Training may mark ``renderer_name_is_resolved=true``. Direct
cookbook renderer overrides leave it false so a simultaneously supplied
semantic thinking-history mode is validated rather than silently ignored.

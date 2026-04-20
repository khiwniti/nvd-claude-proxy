# E2E checklist — nvd-claude-proxy with Claude Code

Run the proxy locally with a real `NVIDIA_API_KEY`, then execute the checks
below with the `claude` CLI pointed at the proxy:

```sh
export ANTHROPIC_BASE_URL=http://localhost:8787
export ANTHROPIC_API_KEY=any-non-empty-string
export ANTHROPIC_MODEL=claude-opus-4-7
export ANTHROPIC_SMALL_FAST_MODEL=claude-haiku-4-5
```

## Basic flows

- [ ] `claude` → simple prompt, verify streaming works end-to-end.
- [ ] `claude --model claude-opus-4-7 "edit README.md to add a section X"`
      exercises the Read/Edit tool loop end-to-end.
- [ ] Send a screenshot (drag into the TUI / `/image`) against
      `claude-opus-4-7-vision`; verify vision model is used and the response
      references the image.
- [ ] Turn on thinking (`/think`) → verify `<details>`-collapsed thinking
      rendering in Claude Code.

## Robustness

- [ ] Kill the proxy mid-stream → Claude Code should report a clean network
      error, not hang.
- [ ] Restart proxy, resume — `toolu_` ids in prior turns should still
      round-trip.
- [ ] Run 50 RPM for one minute → verify 429 is surfaced as
      `rate_limit_error` rather than a generic failure.
- [ ] `claude /cost` → token counts should be within ~15 % of reality.

## Known limitations to verify

- [ ] Thinking blocks produced by the proxy are NOT replayable against the
      real Anthropic API (signature is proxy-local).
- [ ] `cache_control` markers on input are silently dropped; usage reports
      zero cached tokens.
- [ ] GIF/WEBP images are transcoded to PNG before upstream.

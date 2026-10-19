# Local edits

2026-08-20 — L2→L3 split: SKILL.md body relocated to references/cut-craft.md, references/packed-transcript.md, references/editor-subagent-brief.md, references/color-grade.md, references/subtitles.md, references/animations.md, references/output-spec.md, references/edl-format.md, references/anti-patterns.md to meet the 5k-token budget (lint MUST 'tokens'). Pure relocation, no content changes. On upstream re-vendor, re-apply.

- 2026-08-22: description += explicit when-to-use sentence (Anthropic best-practices audit; local divergence from upstream).
- 2026-08-22: allowed-tools for bundled scripts via ${CLAUDE_SKILL_DIR} (feature-adoption batch 2)
- 2026-09-03: line 162 cost note rewritten as "USD 0.10" form; a dollar sign before a digit is an argument sigil in SKILL.md (BACKLOG R40).
- 2026-10-19: rebased onto upstream 43cfc56. Upstream's SKILL.md additions were ported into the split layout: self-eval audio + critic pass stay inline in SKILL.md; bold-overlay chunking -> references/subtitles.md; fonts + "show the edit" example -> references/animations.md; music/SFX section -> NEW references/sound.md (pointer in SKILL.md); two anti-patterns -> references/anti-patterns.md. Whisper fallback now honours --audio-track (transcript_path + -map).

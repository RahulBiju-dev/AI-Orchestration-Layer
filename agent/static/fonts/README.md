# Bundled webfonts

Selene ships as an offline-first desktop app, so its typefaces are vendored here
rather than fetched from the Google Fonts CDN at boot. Without this, the UI falls
back to system fonts whenever the machine is offline.

Each file is the **variable** font for its family, subset to `latin` and
`latin-ext`. One file therefore covers every weight the UI uses; the
`@font-face` declarations in `../css/01-tokens.css` expose the full axis range.

| File | Family | Weight axis | Role |
|---|---|---|---|
| `grotesk-latin.woff2`, `grotesk-latin-ext.woff2` | Space Grotesk | 300–700 | `--display` — wordmark and all title text |
| `inter-latin.woff2`, `inter-latin-ext.woff2` | Inter | 100–900 | `--sans` / `--ui-label` — body copy and UI labels |
| `jbmono-latin.woff2`, `jbmono-latin-ext.woff2` | JetBrains Mono | 100–800 | `--mono` — code, numerals, identifiers |

## Licensing

All three families are licensed under the **SIL Open Font License 1.1**, which
permits bundling and redistribution with an application.

- Space Grotesk — © Florian Karsten, https://github.com/floriankarsten/space-grotesk
- Inter — © The Inter Project Authors, https://github.com/rsms/inter
- JetBrains Mono — © JetBrains s.r.o., https://github.com/JetBrains/JetBrainsMono

## Updating

Re-download the `latin` and `latin-ext` subsets from the Google Fonts `css2` API
using a modern browser User-Agent (an older UA yields `ttf` instead of `woff2`),
then keep the filenames identical — they are referenced by
`../css/01-tokens.css` and are served with a one-year immutable cache header.

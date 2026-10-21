# HazeBusters Design System

Instructions for Claude: follow this file for every UI change in HazeBusters. If a request conflicts with it, say so and ask before deviating.

## 1. Product context

HazeBusters tells people the air quality (CPCB-style AQI) for a place, shows where the number comes from, and says what to do. Users are mostly on phones, often schoolteachers, outdoor workers, and the general public in India. Clarity beats decoration: the verdict must be readable in two seconds, in sunlight, on a cheap phone.

## 2. Principles

1. **Mobile first.** Design at 360px wide, then enhance with `sm` (640), `md` (768), `lg` (1024). Never write a desktop layout and shrink it.
2. **One static sky, content floats over it.** The background never scrolls. Cards slide over it.
3. **Glass, but legible.** Translucency is a surface style, never a reason for weak contrast. Text must meet WCAG AA (4.5:1) on every glass surface.
4. **shadcn/ui only.** Use shadcn primitives and extend them with variants. Do not hand-roll a component that shadcn already ships.
5. **Honest data.** Confidence, data source, and "this never changes the AQI" notes stay visible. Do not hide them to look cleaner.

## 3. Stack

- React + TypeScript, Tailwind CSS, shadcn/ui (Radix underneath), `lucide-react` icons, `sonner` for toasts.
- Dark theme only (the sky is the theme). Set `class="dark"` on `<html>`; do not ship a light mode.
- Fonts, matched to the WeMakeDevs hackathon site: **Inter** (400/500/600/700) for all UI and body text, **JetBrains Mono** (400/500/700) for eyebrow labels, tags, handles and numbers-as-metadata, set uppercase with `letter-spacing: .06em`. These were identified by eye from screenshots; confirm in the live site's DevTools and change the two families here if they differ. Hindi text falls back to Noto Sans Devanagari. Minimum body size 16px.

## 4. Background layer

Asset: `/public/bg/hazebusters-bg.webp` (blue sky, clouds, flower meadow; source `background_hazebusters.avif`, 1536x1024). Keep the AVIF as a `<source>` and the WebP as fallback; add a 900w portrait crop for phones if weight matters. The sky takes the top 60%, so cards read over clean blue, and the meadow sits at the bottom edge.

Rules:
- Render as a single `fixed inset-0 -z-10` element. **Do not** use `background-attachment: fixed` (broken on iOS Safari).
- `object-fit: cover`, `object-position: center bottom`.
- **Mobile (< md): 25% blur.** Implement as `blur(var(--bg-blur-mobile))` with `--bg-blur-mobile: 8px` (the agreed "25%" strength), plus `scale(1.06)` so blurred edges never show. From `md` up, blur is `0`.
- Add a navy scrim above the image for contrast over white clouds: `linear-gradient(to bottom, rgb(0 20 50 / 0.30), rgb(0 20 50 / 0.46))`. Do not use a plain black scrim, it turns the sky grey.
- Respect `prefers-reduced-transparency` and `prefers-reduced-data`: fall back to a solid `#1d6fd0` with the scrim.
- The page itself is transparent. `body` has no background colour.

```html
<div aria-hidden class="fixed inset-0 -z-10 overflow-hidden">
  <img src="/bg/hazebusters-bg.webp" alt="" class="h-full w-full scale-[1.06] object-cover object-bottom blur-[8px] md:scale-100 md:blur-0" />
  <div class="absolute inset-0 bg-gradient-to-b from-[rgb(0_20_50/0.30)] to-[rgb(0_20_50/0.46)]"></div>
</div>
```

## 5. Tokens

Define in `globals.css` as CSS variables and map in `tailwind.config`.

**Glass surfaces**

| Token | Value | Use |
|---|---|---|
| `--glass-bg` | `rgb(255 255 255 / 0.10)` | Default card |
| `--glass-bg-strong` | `rgb(255 255 255 / 0.16)` | Inputs, selected states, popovers |
| `--glass-border` | `rgb(255 255 255 / 0.22)` | 1px border on every glass surface |
| `--glass-highlight` | `inset 0 1px 0 rgb(255 255 255 / 0.25)` | Top edge sheen |
| `--glass-shadow` | `0 8px 32px rgb(0 0 0 / 0.25)` | Elevation |
| `--glass-blur` | `20px` (mobile `14px` for performance) | `backdrop-filter` |

Always apply: `background: var(--glass-bg); backdrop-filter: blur(var(--glass-blur)) saturate(140%); border: 1px solid var(--glass-border); box-shadow: var(--glass-shadow), var(--glass-highlight);` Include the `-webkit-backdrop-filter` prefix. Provide a `@supports not (backdrop-filter: blur(1px))` fallback of `rgb(15 25 30 / 0.85)`.

**Text**: primary `#FFFFFF`, secondary `rgb(255 255 255 / 0.78)`, muted `rgb(255 255 255 / 0.62)` (never below this, never on top of busy sky areas without a card).

**Brand palette** (sampled from the WeMakeDevs x AWS hackathon pages; teal is retired)

| Token | Value | Use |
|---|---|---|
| `--brand` | `#0B6935` | Primary buttons, selected toggles, key accents |
| `--brand-hover` | `#0F7D3F` | Hover on primary |
| `--brand-pressed` | `#08502A` | Pressed on primary, button bottom edge |
| `--brand-disabled` | `#8EB8A0` | Disabled primary |
| `--brand-bright` | `#1FA256` | Progress fills and dots on glass |
| `--mint` | `#E7F2E9` | Tag chips (text `#086733`, mono) |
| `--warm-white` | `#FBFAF7` | Secondary button fill |
| `--ink` | `#17130E` | Text on light fills (secondary buttons, mint chips) |
| `--orange` | `#FF9A33` | Highlight accent (prizes, deadlines); never for AQI meaning |
| `--burnt-orange` | `#BF3A00` | Text on peach chips |
| `--peach` | `#FFDFD2` | Date / deadline chip fill |
| `--sky` | `#77CDFC` | Sky reference only |
| `--link-on-glass` | `#CFF5DC` | Links and "Details" text on glass |

Orange and peach are sampled from a hackathon banner and are approximate. Brand green on translucent glass over the sky fails contrast as text, so green is used as a solid fill with white text, and link text on glass is the pale `--link-on-glass`.

**AQI scale (CPCB). Semantics are fixed; do not reuse these colours for anything else.**

| Band | AQI | Colour | Tint on glass |
|---|---|---|---|
| Good | 0–50 | `#22C55E` | `rgb(34 197 94 / 0.28)` |
| Satisfactory | 51–100 | `#84CC16` | `rgb(132 204 22 / 0.28)` |
| Moderate | 101–200 | `#F59E0B` | `rgb(245 158 11 / 0.30)` |
| Poor | 201–300 | `#F97316` | `rgb(249 115 22 / 0.30)` |
| Very Poor | 301–400 | `#EF4444` | `rgb(239 68 68 / 0.30)` |
| Severe | 401+ | `#B91C1C` | `rgb(185 28 28 / 0.35)` |

Never rely on colour alone. Always show the band name and the number. Forecast bars also carry value labels.

**Radius**: cards `20px` (`rounded-[20px]`), buttons/inputs `14px`, badges `9999px`. **Spacing**: 4px base; card padding `16` mobile, `24` from `sm`; gap between cards `12` mobile, `16` from `sm`.

## 6. Layout

- Single centred column: `mx-auto w-full max-w-[560px] px-4 pb-[calc(2rem+env(safe-area-inset-bottom))] pt-[calc(1rem+env(safe-area-inset-top))]`.
- From `lg`, allow `max-w-[960px]` two-column result view: left = verdict + Air now, right = forecast + video; "What to do" spans full width. Input form stays single column.
- Page scrolls; background does not. Cards enter with a subtle slide-up (`translateY(12px)` to `0`, 250ms ease-out, staggered 40ms). Disable under `prefers-reduced-motion`.
- Touch targets at least 48px tall. Primary action sits near the thumb: on mobile, "Check the air" is a sticky bottom bar (glass-strong, safe-area padded).

## 7. Component mapping (shadcn)

Install via `npx shadcn@latest add card button badge toggle-group select tabs accordion progress alert separator skeleton sonner tooltip`.

| Current element | shadcn component | Notes |
|---|---|---|
| All panels | `Card` | Add `glass` variant (section 5 rules). Never nest glass inside glass more than one level; the inner level uses `--glass-bg` at 0.06 with no blur. |
| "Who is this for?" School / Outdoor worker / General | `ToggleGroup` (type single) | 3 equal columns, selected = solid `--brand` fill, white text, weight 600; unselected = 14% white glass with a 35% white border; hover brightens. Must always have a value (default School). |
| Use my location | `Button` variant `secondary` | Leading `MapPin` icon. Show spinner while locating. |
| Or pick a city | `Select` | Glass trigger and content. Placeholder "Choose a city". |
| Record or choose a clip | `Button` variant `secondary` | Leading `Video` icon. After selection show filename, duration, and a remove (`X`) button. Accept `video/*`, `capture="environment"`. |
| Check the air | `Button` variant `primary` (solid brand green) | Full width, 56px tall. Disabled until location is set; explain why in helper text, not only by greying out. |
| Progress steps (Starting, Uploading, Analysing, Done) | `Progress` + step list | Vertical list, `CheckCircle2` done, `Loader2` spinning for current, empty circle for pending. Announce with `aria-live="polite"`. |
| Verdict ("Outdoors OK") | `Alert` styled as hero card | Tinted glass using the AQI band tint, 1px border in band colour, large title (28px, 700), subtitle, one-line reason. Icon + word, not colour only. |
| AQI number + band | `Badge` + 64px numeral | Badge uses band colour tint with white text. |
| Confidence line | `Badge` variant `outline` + text | Low = amber, not red (red is reserved for AQI bands). Include a `Tooltip` or `Accordion` explaining why. |
| Details (2) | `Accordion` | Default closed. |
| Forecast (3-hour slots) | Custom bar chart in a `Card` | Bars use AQI band colours. Value above each bar, time below. Today and Tomorrow are separate rows. Bars at least 28px wide on 360px screens; show "Now" marker. Highlight school-hours slots when audience = School. |
| What's in your video | `Card` with `Separator` rows | Label left, result right. Detected = band colour text plus icon. Frame preview with `rounded-xl`, caption below. Keep the note "This only shows possible sources. It never changes the AQI." |
| English / हिन्दी | `Tabs` | Both tabs always visible. `lang="hi"` on the Hindi panel. |
| Audio player | Native `<audio controls>` wrapped in a glass pill | Add a custom `Play` button only if native styling clashes. Keep a visible transcript. |
| New check | `Button` variant `secondary` | Full width. |
| Data attributions footer | Plain muted text | Keep CAMS / Open-Meteo / WAQI links, underlined, 13px minimum. Place in a glass footer card. |
| Errors and toasts | `sonner` | Glass toast, bottom on mobile, top-right on desktop. |
| Loading | `Skeleton` | Glass-tinted shimmer shaped like the final card. |

## 7b. Buttons

All buttons are **solid fills** (no gradients, no glass blur) so they stand out against the translucent cards, matching the hackathon site's green CTAs. Add these variants to `button.tsx` (replace the shadcn defaults):

| Variant | Fill | Text | Border | Edge shadow |
|---|---|---|---|---|
| `primary` | `#0B6935` | white | none | `0 2px 0 #08502A` + soft drop shadow |
| `secondary` | `#FBFAF7` | `#17130E` | `#E3DFD6` | `0 2px 0 #CFCABF` + soft drop shadow |
| `glass` (low emphasis only) | 20% white | white | 30% white | none |

Shared: height 52px (primary CTA 56px, compact 44px), radius 12px, weight 600, Inter 16px. Min touch target 48px except the compact variant in dense desktop bars.

States, all with a 150ms ease transition on background, transform and shadow:
- **Hover** (pointer devices only): primary fill to `#0F7D3F`; secondary fill to `#FFFFFF`; both lift `translateY(-1px)` and the drop shadow grows.
- **Active / pressed**: primary fill to `#08502A`; secondary to `#EFECE7`; both sink `translateY(1px)` and the edge shadow shrinks to 1px.
- **Focus-visible**: 2px white ring, 2px offset.
- **Disabled**: primary fill `#8EB8A0`, no shadow, `cursor: not-allowed`, and helper text says why.
- **Loading**: keep the label, swap the leading icon for a spinner, and set `aria-busy`.
- Respect `prefers-reduced-motion`: no transform, colour change only.

## 8. Typography scale

| Role | Mobile | From `md` | Weight |
|---|---|---|---|
| App title | 28 | 32 | 700 |
| Verdict title | 28 | 32 | 700 |
| AQI numeral | 64 | 72 | 700 |
| Card heading | 18 | 20 | 600 |
| Body | 16 | 16 | 400 |
| Caption/meta | 13 | 14 | 400 |
| Eyebrow / tag (JetBrains Mono, uppercase, +0.06em) | 12 | 12 | 500 |

Line height 1.4 for body, 1.15 for headings. Text shadow `0 1px 2px rgb(0 0 0 / 0.35)` allowed on text placed directly over the background (title and tagline only).

## 9. Accessibility and performance

- AA contrast on all text; verify against the **brightest** part of the sky (sun area) behind each card.
- Visible focus ring: `2px solid #FFFFFF`, offset 2px, on every interactive element (plus the dark scrim keeps it visible over clouds).
- Colour is never the only signal (icons, labels, numbers).
- Hindi parity: every English string has a Hindi counterpart and layouts must tolerate roughly 30% longer text.
- `backdrop-filter` is expensive: at most ~6 blurred layers on screen at once, reduce to `14px` on mobile, and never animate blur radius.
- Background image under 250KB on mobile (WebP, `srcset`), loaded with `fetchpriority="high"`; use a dominant-colour placeholder while loading.
- Support low-end Android and slow networks: no layout shift from the background, results render progressively.

## 10. Do and don't

**Do**
- Reuse the glass tokens; add variants instead of one-off classes.
- Keep AQI colours and wording exactly per section 5.
- Keep the data source, confidence, and disclaimer text visible.

**Don't**
- Don't add a second background, gradients behind cards, or opaque white cards.
- Don't use red for non-AQI meaning.
- Don't use `background-attachment: fixed`.
- Don't use glass for body text blocks without the scrim behind them.
- Don't give buttons gradients, glass blur or a teal/neon fill; they are solid brand green or warm white.
- Don't put brand-green text directly on glass; use solid green fills with white text, or `--link-on-glass`.
- Don't introduce new UI libraries beyond shadcn/Radix, Tailwind, and lucide.
- Don't make the AQI look more certain than the data allows.

## 11. Definition of done for any UI change

- [ ] Checked at 360, 390, 768, 1280px widths
- [ ] Background static, blurred on mobile, sharp from `md`
- [ ] Every surface uses glass tokens and has the 1px border
- [ ] Contrast checked over the sun area
- [ ] Only shadcn components used
- [ ] Buttons are solid, and hover, pressed, focus and disabled states are all implemented
- [ ] Reduced-motion and reduced-transparency fallbacks work
- [ ] English and Hindi both checked

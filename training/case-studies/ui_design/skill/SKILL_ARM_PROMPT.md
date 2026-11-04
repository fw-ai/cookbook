<!-- SKILL.md -->
---
name: design-taste-frontend
description: >-
  tasteskill v2, the anti-slop frontend skill. Art direction for websites, distilled from a study of
  247 curated gallery sites (One Page Love, Awwwards Sites of the Day). Deals researched directions
  (idea, composition, type, palette, photographs, section blueprints) from a library, enforces a strict
  no-slop house style with a real-browser inspector, and runs a look-and-fix loop on screenshots. Use
  this whenever you build, design, redesign, restyle, polish or review a landing page, marketing site,
  portfolio, product page, one-pager, hero or any public-facing web page, in any stack (HTML/CSS, React,
  Next.js, Vue, Svelte, Astro, Tailwind) — also when the user says a page looks generic, templated,
  "AI-made", sloppy, or asks to make it look designed, premium, award-level or less like every other
  site. Also for app screens: dashboards, admin and analytics views, settings, inboxes and other screens
  people work in, with their own run and rules.
---

# tasteskill v2

A page built by a model usually looks like the average of every page the model has seen: a centred
headline over three lines, a little label above it, a row of icon cards, a soft gradient, a few thin
rules and a monospace caption. Nothing in it is wrong and nothing in it was decided.

This skill replaces those defaults with decisions, in four ways:

1. **A script deals the directions, you choose among them.** Asked to invent a look, a model picks
   the category default every time. `scripts/draw.mjs` deals three of twenty researched directions;
   you choose the one that carries the subject's idea, and the script then deals its type system,
   palette, hero and section compositions from a library distilled from 247 real sites.
2. **The idea comes from the subject.** In the study, what separated the best sites from the forgettable
   ones was a concept taken from the thing itself, not the styling.
3. **A house style with no ornaments,** measured in a real browser by `scripts/inspect.mjs`, so the
   rules are checked rather than hoped for.
4. **You look at what you made.** Screenshots at three widths, a juror's rubric, fix, look again.

One warning learned in testing. Taking the ornaments away is only half the job; a page that has lost its
labels, rules and cards and gained nothing in their place reads as thin, and blind jurors rightly prefer
a richer page with tells over an empty page without them. The other half is substance: type set
larger, whole fields of colour, real pictures of this subject's world (found or drawn), and every
section carrying its full content. The skill is built to give you that; do not trade it for restraint.

In the commands below `<skill>` is this skill's folder. Run them from the project's root, so the
working files land in `.taste-skill/` there. The scripts need Node 22+ and an installed Chrome, Edge
or Chromium, and install nothing. `assets.mjs` also needs network access; it uses no API keys.

## Page or app screen

Decide first which of two kinds of work this is.

- **A page** is read and scrolled: a landing page, a marketing or product site, a portfolio, an
  event, a restaurant, a one-pager. It follows **the run** below.
- **An app screen** is worked in: a dashboard, an admin or analytics view, settings, billing, an inbox
  or queue to work through, a wallet, a list with filters, an editor or builder. It follows
  **`references/app-screens.md`**: its own run, its own draw (`draw.mjs --app`), its own house rules
  (quiet borders, small labels and repeated panels are normal there; gradients, glows, icons in tinted
  squares, rainbow charts and invented vanity numbers are not) and `inspect.mjs --app`. Read that file
  whole before you start; the page run below does not apply.

A product site that shows its app is a page; the interface fragments on it follow the app-screen
rules for their insides. Deal the product's app style once, in its own folder, with the page's colours
and type kept:

```bash
node <skill>/scripts/draw.mjs --app --kind <kind> --brief "<the product>" --keep palette,type --out .taste-skill/app
```

and build every fragment the page shows (the hero's live screen, a feature's crop, a chart) as a real
piece of that app: `assets/app.css` for its panels, tables and controls, `assets/charts.js` for its
charts, real data that adds up, one finding stated, cropped and scaled the way the page's blueprint
treats a picture. A fragment that looks like a working product is the strongest picture a software page
has; grey bars and lorem numbers are the weakest. When a request mixes both (a landing page and the
dashboard behind it), build each with its own run and one shared palette and typeface.

## The run

Do these in order. Steps 1–5 take a few minutes and decide whether the page is any good.

### 1. Read the brief and the project

Find out what is real: the subject, the audience, the one thing the page must get someone to do, the
stack, any existing brand (colours, fonts, logo), and above all **what imagery truly exists**: the
project's own photographs or product shots, a real interface that can be shown, an image generation
tool you can call, or nothing. Do not plan around pictures you do not have. Nothing of its own is
rarely nothing at all: the subject's city, street, materials and history are in open archives and
museums, free to use with a credit (step 6, `scripts/assets.mjs`).

If the request is vague, decide and state your reading in one line instead of asking. Ask only when a
missing fact blocks the work (for example, no product name).

Then **inventory the subject** in your notes: eight to twelve things that belong to this client and to
no competitor. Look for its name and what the name means; its address, door number, street and
neighbourhood, and what that place looks like (its tiles, signs, paint colours, typefaces on its shop
fronts, its light, its weather); the numbers in the brief (seats, rooms, minutes to the sea, the
price, the year); the constraint it lives with (only one batch a month, closed in winter, by
appointment); its materials and colours; its rituals and hours (the oven lit at five, the ferry at
seven); the printed things of its world (a timetable, a label, a ticket, a sign). This list is where
the idea comes from in step 4. In blind tests, the page that won each brief was built from an item on
such a list (the client's name and what it means, the street it stands in, the one constraint that
makes it what it is), and the pages that lost had a good composition with a category idea. The list
is also your picture search: each place, material and printed thing on it is a query for step 6.

### 2. Write the argument

Before any design, list the page as claims with their proof, one line per section, the way
`references/copy.md` §1 shows. The claims decide which sections exist and in what order. Delete any
section that has no claim. This is how you avoid the stock sequence of hero, logos, three features,
testimonial, pricing, call to action.

### 3. Draw the direction

```bash
node <skill>/scripts/draw.mjs --sector <sector> --brief "<one line about the subject>" \
  --sections <roles in the order of your argument> [--tone <tone>] [--have <assets>] [--keep palette,type] \
  --options 3
```

This deals three directions and prints a short summary of each. Hold each one against your inventory:
which way of composing can carry something only this client has? Choose that one, not the most
familiar, and write one line in your notes on why. In testing, a single random direction that did not
suit the subject was the commonest cause of a weak page. Then deal the full sheet:

```bash
node <skill>/scripts/draw.mjs <the same arguments, without --options> --direction <chosen id>
```

- `--sector`: `node <skill>/scripts/draw.mjs --list sectors`. Pick the closest.
- `--sections`: roles from `--list roles` (features, showcase, how-it-works, testimonial, proof, stats,
  pricing, faq, cta, footer, intro, about, work, gallery, list, contact). The hero is implied.
- `--tone` (optional): quiet, bold, warm, technical, playful, luxury, editorial. Only pass it when the
  brief clearly asks for one.
- `--have`: assets that really exist, comma-separated: `photography` (the project's own photographs of
  the place, people or work), `product-photo` (photographs of the specific product, or an image tool
  that can make them), `product-ui` (real screenshots), `illustration`, `video`. Leave it out if you
  have none: the draw then avoids compositions that depend on them and the sheet says how to fill
  picture slots honestly (software shows its own interface built in HTML; a product with no photographs
  is drawn and set among real pictures of its world, never a fake product shot). Pictures you will
  find in the open collections are not the project's own; leave them out of `--have`.
- `--keep palette,type`: when the brand already fixes colours or typefaces. The draw then deals only
  the composition, and you map the brand's values onto the roles.

The command prints a **direction sheet**. It is the art direction for this build:

- **Adopt it.** Do not blend it with the layout you would have made anyway. If it feels unfamiliar,
  good: the familiar layout is the one this skill exists to prevent.
- Redraw one slot (`--redraw hero | type | palette | photos | scenes | section:<role> | direction`) only for a
  reason you can state: it contradicts a stated brand constraint, or needs an asset that does not
  exist. Never redraw to get something more comfortable. Record every redraw in your report.
- When a dealt section cannot hold the real content (it compares three plans and there is one price;
  it wants five quotations and there are none), do not bend the content to fit. List the others with
  `--alternatives <role>`, pick the one whose assumptions match the content, and take it with
  `--redraw section:<role>=<id>`.
- **Check the palette against the subject.** A dealt palette is a set of proportions (one ground, one
  ink, one accent, fields) as much as a set of colours. If its colours fight the subject's own material
  (candy pastels for a peat whisky, neon for a mountain hut), redraw the palette once; if the subject
  has unmistakable colours of its own (its material, the paint of its street, the landscape outside
  its windows), keep the dealt proportions and take the colours from the subject instead, and say so
  in the report.
- Reprint it any time with `node <skill>/scripts/draw.mjs --show`. In a long session, re-read it before each
  section you build.

### 4. Find the idea

The direction is a way of composing. The idea is what makes the page belong to this subject and no
other. Write this sentence in your notes and do not start building until both blanks are filled from
the subject itself:

> On this page, **[something that belongs to the subject]** becomes **[the device the direction calls for]**.

Places to look for the first blank:

- **Its output.** What does the thing produce? Put the real output on the page as material. (A camera
  that prints poems delivered its copy as paper slips.)
- **Its object or shape.** What does it physically look like, in cross-section, in silhouette?
  Enlarge it, crop it, make it the frame or the one motif.
- **Its name.** Can the letters hold a photograph, stack into a wall, split to two corners?
- **Its place.** Is there one view that makes someone want to be there?
- **Its ritual.** Is there a sequence the page can perform as it scrolls?
- **Its artefact.** What printed thing already exists in its world: ticket, menu, label, receipt,
  invitation, folder, manual?
- **Its opposite.** Which other category owns the feeling this one lacks?

Fill the first blank from your inventory (step 1), not from the category. **The competitor test:** put
the nearest real competitor in the sentence (the bakery two streets away, another oak furniture maker,
another hut on the same mountain). If the sentence still works, the idea belongs to the category: a
loaf of bread is every bakery; the seven doors of the street this one is named after are only this one.
"Clean and modern" is not an idea. The sites the study's panel scored lowest were the ones whose idea
was recorded as "none beyond the standard formula".

**One live thing.** Give the page one thing the visitor can do, or that is different for each visit,
made from the idea: pick a wood and see the chair in it; type your name and see it cut in the
foundry's letters; set the loan and read the real monthly figure; see whether the hut's stove is lit
right now and how long until the last ferry; choose a size and see it against the room. In blind tests every winning page had one and the jurors named it first. Rules: it uses the subject's real content, it works on a phone with a thumb, it has a plain
fallback without script, it never blocks reading, and it is one thing, not a playground. It is
built in step 6 as part of the page, not added as motion in step 9.

**The scenes carry the idea too.** The scroll scenes on the sheet are devices waiting for a subject.
Decide what each one shows from your inventory before building: the fly-through passes through this
studio's own title cards, the route is the road this shipment really takes, the picture that opens to
the whole screen is the room the guest will wake up in. A scene filled with pictures that could belong
to anyone is a screensaver.

### 5. Write the words

Write the whole page as a copy deck (plain text) before markup, following `references/copy.md`:

- **Every headline fits in two lines or fewer at every width.** Do the arithmetic in copy.md §2 before
  committing to a sentence: about 32 characters is safe everywhere at a confident size. The studied
  gallery heroes have a median of 5 words and 32 characters.
- One supporting sentence, directly under the headline (32–48px below it) and shown with it, never
  docked at the foot of the first screen or held back by an animation. When the headline is a name or
  a figure of speech, this sentence says plainly what the thing is. One primary action, named for what
  it does, as a real button on the first screen.
- **The first screen answers the visitor's first question in large type.** For an event: what it is,
  when and where (the dates belong in the headline or at the second size, never only in small text).
  For a product: what it is and what it costs or does. For a person: who and what they do. Every
  section the visitor came for (the line-up, the menu, the rooms, the price) has its own heading and
  its own place in the navigation.
- **Write the full content.** Every section the brief asks for delivers its information completely:
  the whole menu with prices, each room with its own details, every speaker and talk, the price with
  what it includes, an answer for each real question. Thin sections are the commonest reason a clean
  page loses to a busier one.
- Sample details are expected where the brief leaves gaps (dish names, room names and rates, project
  titles) so the design is seen at full density; list them in the report for the client to confirm.
  Proof is never invented: no customers, quotes, reviews, press, logos or business statistics.

### 6. Build it, still

Build the page with motion switched off in your head; a page that needs animation to look finished is
not finished. Order of work:

1. **Tokens.** Colour and type roles from the sheet as CSS custom properties. Everything downstream
   references tokens only. `<skill>/assets/base.css` is a neutral foundation you may start from (reset, token
   names, fluid type steps, grid, focus and selection styles, the motion helpers); it has no look of
   its own.
2. **The hero, exactly as the blueprint describes it,** at desktop and at phone width. Get the big
   thing big and the small things small before anything else exists.
3. **Each section from its blueprint,** with this project's content. Keep the geometry, change the
   subject. Neighbouring sections must not share a layout. **Carry the idea into every section,**
   pricing, FAQ and footer included: the commonest failure in the study, even among good sites, is a
   distinctive first screen followed by stock blocks.
4. **Imagery,** in this order of preference:
   1. the project's own;
   2. generated, if you have an image tool;
   3. **real pictures of the subject's world from the open collections,** whenever an honest one
      exists. Search them with `node <skill>/scripts/assets.mjs "<query>"` (Openverse, Wikimedia
      Commons, the Art Institute of Chicago and the Met; only licences that allow commercial use, by
      default). Search the inventory, not the category: its street in old postcards, an old map of the
      neighbourhood, its material up close, its trade in engravings, a museum object of its kind, a
      botanical plate, a star chart. Use them large and with intent: one archival photograph as the
      hero, an engraving enlarged into the page's motif, the real map of the street as the contact
      section. Look at each before choosing; `--get <ref>` downloads it into `assets/` and records
      its source and licence, and `--credits` prints the credit line for the footer. A found picture
      shows the world around the client and is never presented as the client's own product, room,
      dish, person or team;
   4. **drawn in code from the subject itself,** for what no honest picture exists of (its objects,
      its tiles, its windows, its product's interface, made as one consistent flat graphic language in
      the palette's real colours);
   5. the photographs dealt on the sheet, only where an honestly generic picture fits (the sea for a
      sea house), never as a stand-in for the subject;
   6. type and colour alone.

   When the project has footage or a real picture of its own process or product, the first screen
   shows it, large (about 40% of the screen or more). Never write a sentence on the page about where
   the pictures come from or what they are not ("none of these shows our bar"): a caption says what
   the picture is, and the credit line sits small in the footer. A portfolio whose work is invented
   for the brief draws each piece as a finished still of that work (its title card with its own type,
   an end frame, a poster), with real detail, never as a pictogram of the client's product.

   **The floor, which the inspector checks:** at least four real pictures on the page, usually six to
   twelve (photographs, archival images, maps, engravings, footage, raster artwork; for software, its
   cropped interface fragments count too). SVG drawings do not count toward the floor: draw what no
   picture exists of, and still carry real pictures. Add one moving picture wherever an honest clip
   exists: search it with `assets.mjs --video "<query>"` and use it as a muted loop with a poster
   frame. Models skip pictures and footage because finding them is work; the page fails without them.

   `references/imagery.md` covers searching, licences and credits, video, drawing, crops and treatments.
5. **The live thing** from step 4, built into the section where it proves the claim (often the hero
   or the booking), with real content, a thumb-sized phone version and a plain fallback.
6. **The scroll scenes,** when the sheet deals them (section 9), each with the section it carries: the page's big
   moments, the camera that flies through the work, the picture that opens to the whole screen, the
   word that buckles under the scroll. Copy the shared stage and the scene's files from
   `<skill>/assets/scenes/`, fill the scene with this project's own pictures and words, and set the
   page's tokens so it takes the page's look. A scene is composition: it decides how long the page is
   and what each screen shows, so it is built now and seen in every look round, not added at the end.
   `references/scenes.md` says how to put one on a page, fill it, fit it and check it;
   `assets/scenes/demo/<id>.html` is a working example of each.
7. **Fullness.** Look at each desktop screen as a poster: one dominant thing and the content it
   carries. A screen that is mostly empty ground is an underbuilt section, not a rest; one deliberate
   rest per page is the limit, and the inspector fails a page with two or more near-empty screens.
   Running text is in the full ink at 16px or more; grey is for secondary notes.
8. **The frame.** The page has one pair of equal outer margins at every width, and every section uses
   the width the page has set up: when its text sits in the left half, the right half holds its
   picture, its second column, its list or its figure, never nothing. Asymmetry lives inside the grid,
   never in the frame: a void is placed only when something closes it on the far side. No stretch of
   the page taller than about half a screen holds nothing at all; sections are 120–240px apart on
   desktop. Every section the visitor came for opens with a heading that names it, and the first
   screen carries the page's title in an `h1`. These are what readers name first when a page looks
   wrong ("no clear title", "random empty space", "everything is pushed to one side"), and the
   inspector measures each of them.
9. **States and semantics.** Real links and buttons, visible focus, alt text, a phone layout that is
   its own composition.

Read `references/type.md`, `references/colour.md` and `references/layout.md` when you need the detail
behind a decision; the laws below are the short form.

### 7. Gate

```bash
node <skill>/scripts/lint.mjs <source files or folder>
node <skill>/scripts/inspect.mjs <url or path/to/index.html>
node <skill>/scripts/scenes.mjs <url or path/to/index.html>     # when the page has scroll scenes
```

`scenes.mjs` photographs every scroll scene at 0, 10 … 100% of its hold, on desktop and on a phone,
and the reduced-motion page; with scenes on the sheet, `inspect.mjs` also films the whole scroll
(`desktop-film-NN.jpg`). A scene that never started or rests on desktop is a `FAIL`.

`lint.mjs` scans source for banned patterns and stock phrases. `inspect.mjs` opens the page in a real
browser at 1440, 768 and 390 pixels, measures it (headline line counts, type sizes, fonts actually
loaded, colours, rules, shadows, gradients, contrast, overflow, and the frame: equal margins, sections
huddled on one side, blank stretches, the title, untitled sections, text against the screen edge) and
saves screenshots to `.taste-skill/inspect/`. Fix every `FAIL`. Treat each `WARN` as wrong unless you can defend it to a
designer in one sentence. Re-run until there are no `FAIL`s. Do not argue with the meter and do not
hide things from it. The inspector serves local files over a throwaway local server; if the page must
also open straight from the file system, open `index.html` that way once before you finish.

### 8. Look

Passing the gate means nothing banned was measured. It does not mean the page is good. Open the
screenshots the inspector lists (every viewport, with your image viewer or Read tool) and critique them
with `references/critique.md`: the fourteen passes, the Awwwards-weighted score, the caps. Write the critique
down in its format, fix the top three problems, run the inspector again, look again. Three rounds at
most.

Judge only the pictures. If a problem is visible in a screenshot, it is real, whatever the code says.

### 9. Move

Now build the motion layer. It is required, and it is more than one fade: the inspector counts the
elements that really move while the page loads and is scrolled, and fails a page under eight (it warns
under sixteen). Any scroll scenes are already built (step 6); around them the layer has four parts:

1. **The signature move** named on the sheet: one, done well (a dealt scene can be it).
2. **An entrance for the parts of every section:** the heading, then the text, then the picture,
   staggered by 60–90ms, rising 16–32px with opacity, triggered once as the section enters. Not one
   fade for the whole block, and never content that stays hidden if the script fails.
3. **Feedback on every control:** press, hover and focus on buttons, links, cards and the live thing.
4. **One moving picture:** a found clip as a muted loop, a slow scale or drift on a photograph, or a
   scroll-linked move on the hero picture.

Transform and opacity only; every move has a `prefers-reduced-motion` fallback that keeps the content
and drops the travel. A move that changes the colour or the layout of whole screens (colour by
chapter, a pinned story) is part of the composition, so build it in step 6 and let every look round see
it. `references/motion.md` has the tokens and tested recipes. Then run the inspector once more and check
every screenshot for content that failed to appear.

### 10. Report

Do not report until all of these are true; each one was skipped in a test run that came out badly:

- `inspect.mjs` shows **0 FAIL** at 1440, 768 and 390, on the final version of the page.
- You opened every screenshot it listed after the last change, not only the first screen.
- With scroll scenes: `scenes.mjs` ran on the final page with no problems, and you read every sheet
  (desktop, phone, reduced motion).
- A stranger reading the first screen for five seconds can say what this is and what to do.
- Every section has a heading; the margins match on both sides; no stretch of the page is empty.
- The critique of your last round is written down in the shape of `references/critique.md` §4.

Tell the user, briefly: the direction dealt (name and seed) and any redraws with reasons; the idea
sentence and the live thing; the final scores from your last critique round and what still falls short;
every found picture and clip with its title, creator, licence and source page (`assets.mjs --credits` prints
the list) and where its credit sits on the page; which bundled photographs and which drawings stand in
for photography (so they can be replaced); and any sample content that must be swapped before launch.
Do not claim a check you did not run.

## House rules

These hold on every build, whatever the direction. `inspect.mjs` measures them.

| Rule | Instead |
| --- | --- |
| **Headlines: two lines or fewer, at every width.** Every `h1`–`h6`, and the main line of the first screen whatever its tag. | Cut words. Then widen the measure. Size last. |
| **A clear title on the first screen**, in the page's one `h1`, and a heading on every section. | Say what this is: the name with what it does, the product with what it is for, the event with its date and place. |
| **Equal outer margins; no section huddled on one side** with the other half of the screen empty. | Close every void with something on its far side: the section's picture, a second column, a list, a figure. |
| **No blank stretches** taller than about half a screen. | The normal section gap. A deliberate rest is a placed picture or a field of colour, once per page. |
| **Statements are paragraphs.** An intro sentence or a quotation set large is marked up as `<p>` or `<blockquote>`, stays at statement size (36px or less on desktop, 24px on a phone) and runs to about 25 words, three lines. | Larger than that it is a headline and the two-line rule applies. In blind tests jurors read a 42px sentence over four lines as a four-line headline. |
| **No eyebrows.** No small label above a heading, no pill badge above the headline, no "New ·" chip. | Put the information in the heading or the first sentence, or drop it. |
| **No monospace** outside real code blocks. | The text face. Tabular figures where numbers align. |
| **No index numbering.** No 01 / 02 / 03, no (001), no "Step 2". | Order shown by position, size, or the words. A real `<ol>` if order matters. |
| **No thin rules.** No hairline dividers, no 1px outlined boxes, no grid lines, crosshairs or crop marks. | Space. A step in background tone. A change of scale. A filled shape. |
| **No gradients.** Not as backgrounds, not on buttons, not in text. | Flat colour, or a real photograph. A neutral scrim under text on a photo is the one exception. |
| **No shadows on things at rest.** No glows. | A tone step between surfaces. Shadows only on what truly floats: a menu, a dialog. |
| **No assistant palette.** No yellow-cream ground with terracotta or rust; no violet or indigo. | The dealt palette. |
| **No tiny uppercase letter-spaced labels.** | Sentence case at a readable size. |
| **No icon-topped feature cards, no rows of identical cards.** | The dealt section blueprint. |
| **No glass.** No backdrop blur panels, no floating pill navigation. | A plain bar: wordmark, a few links, one action. |
| **No emoji, no gradient blobs, no decorative shapes, no fake dashboards of grey bars.** | A real picture, a real interface fragment, or nothing. |
| **No styled accent word** in a headline (italic, serif, coloured, underlined). | One face, one colour. The words carry the emphasis. |
| **Two typefaces at most,** usually one. | The dealt type system. |

Three places keep their own conventions and are left alone by the inspector: real code in
`<pre>`/`<code>` (monospace is correct there), real tables and form fields (they may have lines), and
a real interface fragment whose root carries the attribute `data-ui` (the product's own borders and
menus; see `references/imagery.md` §7).

Why these, honestly. Some are what the study measured: among the 247 sites, those the panel scored
7.5 or higher used icon feature cards 1% of the time against 28% for sites under 6.5, pill badges 14%
against 34%, gradients 22% against 52%, shadows 24% against 48%, tiny tracked caps 16% against 57%.
Others are the owner's taste and go against current gallery fashion: monospace labels, index numbers
and hairline grids are actually *more* common on award sites than on weak ones, which is exactly why
generators imitate them and why they now read as costume. They are banned here without exception.
`references/tells.md` keeps the dated registry.

Removing ornament leaves space. Never fill it with another ornament. Fill it with scale (make the main
thing larger), with the full content, with a picture made for this subject, or with a field of flat
colour. Leave it empty on purpose once per page at most.

## Laws of the craft

Short form. Numbers are from the study unless marked as house style.

**Type**
- Two or three sizes do the visible work: one very large, one small, nothing in between. Median
  headline-to-body ratio on the studied sites is 4.3; the best go far beyond (largest type: median
  8.4vw on the top sites against 5.0vw on the weak ones).
- Display type: tracking -0.02 to -0.05em, leading 0.85–1.05. Text: 16–20px, leading 1.45–1.6,
  45–75 characters per line.
- Five sizes on a page is the ceiling. One family is normal (a third of the studied sites); two is the
  limit.
- Left-aligned by default. Centre at most one moment.

**Space**
- One grid, one pair of tight margins (16–32px), used everywhere; large things run to the edges. Text
  blocks share two or three left edges on the whole page.
- Asymmetry inside the grid, never in the frame: the two outer margins are equal, and an off-centre
  heading is answered by something at the far side of the grid.
- The gap between sections is clearly larger than any gap inside one: 120–240px on desktop (the study's median is about 240).
- Each section has a different composition from its neighbours; none is a card grid.

**Colour**
- One ground, one ink, one accent. The accent appears on the action and perhaps one other thing.
- Running text in the full ink. A page set in soft grey reads unfinished.
- Surfaces are separated by a small step in tone, never by a line or a shadow.
- A section may change the whole field colour. Nothing fades into anything.

**Imagery**
- One dominant image beats six small ones. On the top-scored sites the hero media covers at least half
  the first screen 68% of the time.
- Same crop ratios, same treatment, square corners unless the direction says otherwise (73% of the
  studied sites leave media square).
- No image is better than a placeholder, an icon, or an abstract shape. A picture drawn for this
  subject, in its colours and with care, is better than either. A real picture of its world, found
  and credited, is better still.
- Every found picture is downloaded into the project, credited on the page and listed in the report.
  None is presented as the client's own.
- Nothing ever lies on top of a headline or a statement (the inspector fails it).

**Components**
- Buttons: flat, one fill colour, either square-cornered or fully round, decided once (top sites: 52%
  square, 23% pill; the 9–16px "soft" radius is the weak-site default). Most strong heroes have no
  button at all, or one.
- Navigation: wordmark, three to five plain links, one action. No bar chrome.
- The footer is a designed close, not a sitemap: on most strong sites the name returns at full width.

## Special cases

**An existing brand or design system.** Pass `--keep palette,type`, map the brand's colours and faces
onto the roles, and take only composition from the draw. The house rules still apply; if the brand
itself requires a banned device (its logo sits in a gradient, its accent is violet), keep the brand
asset, say so in the report, and pass `--allow <rule>` to the inspector for that rule only.

**Redesign, restyle or review of an existing page or app screen.**

1. **Record the before.** Run the project as it is and inspect it into its own folder
   (`inspect.mjs <url> --out .taste-skill/before`, with `--app` for an app screen) and lint its
   source. Read the code: the framework, the styling method (Tailwind v3 or v4, CSS modules,
   styled-components, plain CSS), the components, and what must survive: routes, forms, data,
   behaviour, the brand's own assets.
2. **A review stops here.** Critique the before screenshots with `references/critique.md`, list the
   findings with the inspector's measurements, and name the three changes that would move the page
   most.
3. **Keep the stack, the content and what works.** No framework or styling-library migration; check the
   dependency file before adding anything. A restyle keeps the layout and takes type, colour and
   components from a dealt direction (`--keep` whatever the brand fixes); a redesign deals a full
   direction and rebuilds the sections from its blueprints inside the existing components. Either way
   proceed from step 2 of the run with the page's real content.
4. **Order of work, biggest effect for least risk:** type (faces and scale), colour (one ground, one
   ink, one accent), the frame (margins, section rhythm, a title on the first screen and a heading on
   every section), states (hover, focus, loading, empty, error), then the stock components (icon
   cards, pill badges, gradient blobs, card grids) replaced by the dealt blueprints.
5. **Prove it.** Inspect again into `.taste-skill/after`, compare before and after at every width,
   click through every link, form and menu that worked before, and report the before and after
   measurements side by side.

**A component or a single section, not a page.** Skip the draw unless there is no direction to follow.
Match the surrounding design, apply the house rules and the laws, and still inspect the result.

**No checkout, no form backend.** The primary button goes to the real destination when the brief names
one. Otherwise it opens an email with a written subject and body (`mailto:` with the visitor's choices
filled in), or, for a shop, it points at `#checkout` and the report says it must be wired before
launch. Never fake a payment form.

**No Chrome or Node 22.** Run `lint.mjs` (Node 18+), take screenshots with whatever browser tool you
have, and critique those. Say in the report that the inspector could not run.

**No network.** If `assets.mjs` reaches no source, draw the pictures and say in the report that the
open collections could not be searched.

## Files

| File | Read it when |
| --- | --- |
| `references/app-screens.md` | Any dashboard, admin, settings or other app screen: its own run, rules, tokens and checks. |
| `references/copy.md` | Writing the argument and the copy deck (steps 2 and 5). The two-line arithmetic. |
| `references/type.md` | Setting type: scales, tracking, measure, loading fonts, pairing. |
| `references/colour.md` | Applying the palette: roles, proportions, tone steps, dark pages, contrast. |
| `references/layout.md` | Grid, margins, rhythm, section variety, navigation, footer, the phone composition. |
| `references/imagery.md` | Choosing, finding, crediting, cropping and treating pictures; what to do with none. |
| `references/motion.md` | Step 9. Budget, tokens, recipes, reduced motion. |
| `references/scenes.md` | Step 6, when the sheet deals scroll scenes: putting one on a page, filling, fitting, the phone, checking, the catalogue. |
| `references/critique.md` | Step 8. How to look, how to score, what to fix first. |
| `references/tells.md` | The dated registry of generated-page tells and what each is replaced by. |
| `references/research.md` | The study: method, numbers, what the best and worst sites do. Cite it, do not guess. |
| `scripts/draw.mjs` | Deals and reprints the direction; `--photos`, `--site <id>` for more references; `--app` deals an app style instead. |
| `scripts/assets.mjs` | Step 6: searches open collections for real pictures of the subject's world; `--get` downloads with the credit, `--credits` prints them. |
| `scripts/lint.mjs` · `scripts/inspect.mjs` | The gate. |
| `scripts/scenes.mjs` | The gate for scroll scenes: each scene at 0-100% of its hold, desktop, phone, reduced motion. |
| `assets/base.css` · `assets/base.js` | An optional neutral foundation to build on. |
| `assets/scenes/` | The scroll scenes: the shared stage (`stage.js`, `stage.css`) and, for each scene in the library, its `.js`, `.css` and a working demo page in `demo/`. |
| `assets/app.css` · `assets/charts.js` | For app screens: a neutral shell, panels, tables and controls, and small SVG charts drawn to the app rules. |
| `library/` | The data the draw reads: directions, heroes, sections, type systems, palettes, photographs, studied sites, motions, scroll scenes (`scenes.json`), and the app styles, shells and screen patterns (`app.json`). Do not read it whole; use the script. |


<!-- layout.md -->
# Layout

How the page is put together: margins, the grid, where text sits, how sections differ, navigation,
endings and the phone. The direction sheet gives the specific compositions; this file gives the habits
behind them, taken from the panel's notes on the 114 studied sites that scored 7.3 or higher (counts
below are hand tallies from those notes, so read them as "about").

## Contents

1. What strong pages do
2. Margins and the grid
3. Where text sits
4. Empty space
5. The frame: margins, balance and titles
6. Sections: carry the idea, vary the arrangement
7. Navigation
8. Endings
9. The phone is a second composition
10. How pages fail

## 1. What strong pages do

- **Tight margins, content to the edges.** About 45 of 114 use outer margins of 32px or less; only
  about 8 use 80px or more. The centred 1100–1300px container is the minority among strong sites and
  the usual frame of the ones the panel called stock.
- **Text hangs from an off-centre line.** About 22 anchor text to one fixed line that is neither the
  centre nor the margin: a third across, 42%, the midpoint.
- **Two or three type sizes with nothing between** (about 30): one very large with one small, or
  everything small with the scale given to the image.
- **One thing per screen** (about 20): one statement, one object, one project.
- **Sections told apart by their ground.** About 20 change the whole background colour at each
  section; about 12 alternate light and dark; about 10 never change and separate by space alone.
- **Sections recomposed, not repeated.** About 14 rearrange every section; about 12 alternate a
  full-bleed image screen with a near-empty type screen.
- **The system broken once, on purpose** (about 14): one centred line on a split page, one paper card
  on a dark page, colour held back until the last screen.
- **Often no navigation bar** (about 45), and endings that are a designed scene rather than link
  columns.
- **Few words.** Median 575 on the whole study; some of the best carry under 200.

## 2. Margins and the grid

```css
:root {
  --margin: clamp(1rem, 2.2vw, 2rem);   /* 16–32px: tight, so content reaches the edges */
  --gutter: clamp(0.75rem, 1.4vw, 1.5rem);
}
.wrap { padding-inline: var(--margin); }
.grid { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); column-gap: var(--gutter); }
```

- One pair of margins and one grid for the whole page. Every left edge is a column line.
- Do not wrap the page in a `max-width: 1200px; margin: auto` container by reflex. Let large things
  (type, images, colour fields) run the full width, and constrain only the measure of running text
  (`max-width` in `ch` on the paragraph, not on the page).
- On very wide screens cap the layout at about 1760–1920px, or let type sized in `vw` stop growing with
  a `clamp()`.
- Directions that call for wide margins (uneven photographs, the quiet instrument) say so on the
  sheet; follow the sheet.
- Never draw the grid. Column lines, crosshairs and guides are banned ornaments here even though they
  are fashionable; the grid is felt through alignment.

## 3. Where text sits

- **Pick the anchor line and reuse it.** If the hero headline starts at column 5, the section
  headings, the pricing figure and the footer links start there too. Two or three anchor lines on a
  whole page; six means nothing was decided.
- **Off-centre beats centred.** A heading in the left third with its paragraph in the far-right
  column on the same baseline and nothing between; a headline in the bottom third under a deliberate
  void; text docked in corners with the middle given to the picture.
- **Opposite ends.** Two short texts at opposite ends of one baseline (name left, one-line description
  right) is the commonest alternative to the centred stack.
- **Centred is a decision.** When a strong page centres, it uses one axis with wide empty flanks and a
  single element per screen, not a centred stack of heading, paragraph and two buttons.
- **Rows for long copy:** heading in the left third, body in the right two-thirds, the same split for
  every row on the page.
- Stagger sparingly: a second line indented by a third so a two-line headline steps across the screen.

## 4. Empty space

Empty space is placed, not left over. In the study of 97 Sites of the Day captured in October 2026,
readers judged every large empty area either a rest, which something closes, or a hole, which nothing
closes. Strong sites have plenty of rests. About half of them still leave at least one hole, and the
holes are what visitors name ("big random blank spaces").

**A void is placed when something closes it.** In the readers' notes the closer is, most often:

- the picture itself (sky, ground, the object, the film) with the type docked at its edge;
- a picture across the void: the photograph at the far side of an empty band, the full-width row
  rising under a statement, the opposite picture in a stagger;
- the type: a headline large enough to span the space, or two short texts at the two ends of one
  baseline;
- on a one-screen page, the wordmark, the navigation and one fact in the corners, like a mount around
  a print.

About one closer in five in the study is a hairline or grid line. That device is banned here; close the
void with the picture, the type, a step in ground tone or a field of flat colour.

**Where holes come from.** In order of frequency in the study:

1. A gap between sections stretched too far: bands of 250–760px of plain ground.
2. Text in one half of a section and nothing in the other.
3. Content that arrives late: a reveal not yet fired, so the visitor scrolls onto bare ground while the
   words fade in.
4. A short block stretched to a full screen: a footer padded to the viewport height, a section title
   alone on a whole screen.
5. A pinned scene that holds almost nothing for several screens.
6. Pictures of different heights top-aligned in a row, leaving a hole under the short one.

**Rules**

- Decide where the void is and name what closes it. If nothing does, it is a hole: move the section's
  picture into it, make the main element bigger, or take the space out.
- Keep the void clean: no stray caption, no decorative shape, no faint watermark, no grid lines.
- Sections sit 120–240px apart on desktop, counted from the last line of one to the first line of the
  next (the median in the study is about 230–250px), and 72–112px on a phone. A change of ground
  colour lets the gap be a little smaller. More than about 40% of a screen of plain ground between two
  sections is a hole.
- The gap between sections is clearly larger than any gap inside one. Related things sit close and
  unrelated things far; a heading has at least twice as much space above it as below.
- No stretch of the page taller than about half a screen holds nothing at all. In the study, runs of a
  screen or more with nothing in the page were almost always scroll distance for a pinned scene; the
  pinned stage must then hold something at every scroll position.
- Content is in place on the first paint. An entrance moves what is already there (see `motion.md`);
  it never leaves a screen of bare ground while the words arrive.
- Size a section to its content. `min-height: 100vh` belongs to a section that fills the screen, not
  to a footer or a lone heading.
- In a row of pictures of different heights, align the bottoms or put something under the short one,
  or crop the row to one ratio.
- A rest is one thing on a screen: a single line, one object, one picture or a field of colour. Once
  per page at most. A screen of nothing is never a rest, and the inspector fails two near-empty screens.
- When a section looks empty, the answer is almost never "add something". Make the main element
  bigger, or cut the section.

## 5. The frame: margins, balance and titles

The frame is what readers judge before any detail: "no good margin", "everything is shifted to one
side", "no clear title", "random unclear sections". The inspector measures each of these. The numbers
are from the study of 97 Sites of the Day (October 2026).

**One pair of equal margins.**

- The left and right margins are the same, at every width. On the studied sites with scrolling
  sections the two sides match to a few pixels: 16 and 16, 20 and 21, 32 and 32, 140 and 137. No
  strong page in the study was pushed to one side.
- The usual margin is tight: median about 28px at 1440, half the sites between about 18 and 45px, a
  few at 80px or more. Here it is 16–32px (section 2), and never under 16px at any width, on the phone
  included. About one award site in seven sets text 8–15px from the edge; that studio habit is
  excluded here.
- The navigation sits on the same margins as the content. The commonest small fault in the study is a
  header on one inset and the page on another (40px against 20px, or 34px on the left against 10px on
  the right).
- A bleed is deliberate and starts from the margin: a row of pictures that runs off the right edge
  begins on the left margin; a display line that bleeds starts or ends on one. A bleed never moves the
  margin.
- Asymmetry belongs inside the frame: an off-centre anchor line (a third across, 42%, the centre line)
  on symmetric margins.

**Use the width the page has set up.**

- In the study about one text section in a hundred leaves half the width empty. Text that sits in one
  half is common, in roughly one site in three, but the other half then holds the section's own
  picture (most often), a full-bleed picture behind the text, a second column, a list or a figure.
- When the text takes one side, put the picture, the second column, the list or the large number on
  the other. Never leave the other half to plain ground.
- The middle counts too: a card at the left edge and a lone button at the right with the middle third
  empty is the same hole.
- The first screen counts too. When the sentence and the button take the left third above a name
  docked at the foot, the upper right holds something (the picture or film, one fact, the action) or
  the name grows up to meet them; in testing, 60% of a first screen left as plain ground under the
  navigation read as a hole, and the inspector now warns when one empty block takes 28% of the first
  screen or more.
- A display line fitted to the width is fitted by measurement (a small script that sets its size from
  the content box, or SVG text with `textLength`), never by a size guessed for one face: a name that
  starts on the margin and runs a few pixels off the other edge reads as a mistake.
- A centred composition has one axis, equal flanks and one thing on it, not a stack of heading,
  paragraph and two buttons.

**A title that says what this is, on the first screen.**

- Asked whether a stranger could say what the page is within five seconds, readers found that 40 of
  the 97 first screens say it in words, 42 say it only through pictures, small print or a bare name,
  and 15 do not say it at all. The last group lost usability and content points, not design points.
- Size alone does not do it: 66 of the 97 first screens carry a display-size line, but 26 of those are
  a name, a slogan or a place with nothing saying what is offered.
- The first screen has one `h1`, at display size (32px or more on desktop; on the clear first
  screens in the study mostly 50–150px), two lines at most, that says what this is: the name with
  what it does, the product with what it is for, the event with its date and place. If the `h1` is
  the name, the line under it, at 18px or more, says what the thing is.
- It is there on the first paint. About a third of the studied pages still showed a loader or nothing readable 4.5
  seconds.

**A heading on every section.**

- 82% of the text sections on the studied sites open with a heading: median 57px (40–96px for the
  middle half), about three and a half times the body size, one or two lines.
- Headings start on the outer margin or on the page's inner anchor line (in the study about a third
  each) and in the same place for every section. About a quarter of the headings in the study are
  centred; centre one moment per page here.
- Where a studied section has no heading, the usual stand-in is a small label: caps, mono or a number
  in brackets. All three are banned here. Put the label's words in a heading in the heading face at the
  section size ("The rooms", "How it works", "Prices"), set them on the anchor line, and let the first
  sentence carry the rest.
- A statement can open a section as its heading only when it names what the section holds.

## 6. Sections: carry the idea, vary the arrangement

The most common failure in every score band is a distinctive first screen followed by stock blocks:
icon cards, a three-up row, an accordion, a link-column footer. About 25 of the studied sites lose
points this way, including several above 7.5.

- **The idea survives to the last pixel.** Ask of each section, pricing and FAQ included: what does
  the page's idea do here? If the page is a menu, the prices are menu lines. If the name is a
  monument, the footer is the name again. If the object's shape is the motif, the button and the image
  masks carry it.
- **No two neighbours share a layout.** Use the shape of each dealt blueprint (list, split, statement,
  full-bleed, mosaic, table, strip, pinned …). Count them in the screenshot; a shape used three times
  is the template showing.
- **Alternate dense and empty, image and type, light and dark** so the scroll has rhythm.
- **A repeated module is for archives only** (a work index, a menu, a schedule). Anywhere else,
  repetition reads as a kit.
- **Break the system once.** After the page has established its rule, one planned exception (a single
  centred line, one inverted section, one image at a different scale) is what makes the rule visible.
- **Short pages are fine.** A quarter of the studied sites are one or two screens. Do not add sections
  to seem complete; each one must carry a claim (see `copy.md` §1).

## 7. Navigation

- The plain form: the wordmark at one end, three to five text links, one action, all at text size on
  the page ground with no bar, border or background. Median link count in the study is 3.
- It is also normal to have less: small links in the corners, a single action, or the navigation
  folded into the content (a list of section titles that is itself the index).
- If it stays fixed, it must never sit on top of content: check every scroll position in the
  screenshots, especially on the phone. Floating pill bars are where the panel most often recorded a
  nav covering text.
- On a phone: wordmark and one action in view; the links go in a simple full-screen sheet opened by a
  text button ("Menu"), or move to the footer when there are only three.
- Navigation is small furniture, however it is drawn. On desktop it takes no more than about a tenth
  of the first screen; on a phone it is one row. A blueprint that builds the navigation as a column of
  colour blocks or a stack of tabs gets that treatment on desktop only, and collapses to one row on
  the phone so the headline, the supporting sentence and the action stay in the first screen. In
  blind tests a nav that ate the hero cost a whole point of usability.

## 8. Endings

- The page should end on a designed scene. In the study: the name at full width (about 18 of 114), the
  hero repeated in inverted colours (about 12), a closing scene of its own such as the one contact
  fact set as a headline (about 14), or simply one quiet line (about 22). Link-column footers appear
  in the panel's notes only as criticism.
- The full-width name is strong but already a studio habit; use it when the direction is name-led,
  otherwise prefer the blueprint dealt for the footer.
- Put the practical facts (address, hours, email, legal line) in the text face at a readable size, on
  the same anchor lines as the rest of the page.

## 9. The phone is a second composition

Design the 390px screen as its own page, not the desktop page squeezed. Strong sites that neglect this
lose usability points; weak ones are often just broken.

- Each blueprint on the sheet has its own phone paragraph. Build it.
- Keep one dominant thing on the first screen. A wordmark that was 20vw wide on desktop might stack on
  two lines at 26vw on the phone and be even stronger.
- Re-author line breaks; re-crop images to portrait (`object-position` per breakpoint, or a different
  crop size in the image URL); keep unevenness with alternating widths rather than making everything
  full width.
- Margins 16–20px. Body text 16–18px. Touch targets at least 44px high.
- Check 390px in the inspector's screenshots for every round: sideways scroll and three-line headlines
  usually appear here first.

## 10. How pages fail

From the panel's notes on the 54 sites scoring under 6.8 (counts out of 54):

- The whole page is the category's stock template; swap the logo and it could be any competitor
  (about 20).
- Centred headline, pill badge, two buttons, screenshot or tile grid below (about 9).
- A distinctive hero, then stock modules in the usual order (about 8, and about 17 more in the middle
  band).
- A default typeface with no character (about 13).
- Text too small or too low in contrast (about 13); legibility lost over busy photographs (about 9).
- An undersized headline and no focal point (about 9).
- The page never says what the thing is, or shows no work (about 12 and 7).
- Thin content stretched over many screens, or a single gag (about 11).
- Gates, preloaders and entry pop-ups before any content (about 9).
- A phone layout that is the desktop shrunk, or broken (about 8).
- Fixed elements sitting on content; careless details such as colliding lines (about 7 and 5).

And the faults of the best sites, which are not to be copied: body text at 9–13px (about 24 of the 90
sites at 7.5 or more), information hidden behind the concept (about 20), the idea running out
mid-page (about 15), text over footage with no scrim (about 12), gates and blank first paints (about
12), floating elements colliding with content (about 11), and a first screen with no statement of what
this is and no action (about 9). Say what the thing is, and offer the action, on the first screen.


<!-- type.md -->
# Type

Typography is most of what a visitor sees and most of what makes a page look designed or assembled.
The direction sheet deals you a type system (faces, weights, settings). This file is how to use it.

## Contents

1. What the study measured
2. The scale: few sizes, far apart
3. Setting display type
4. Setting text
5. Loading the fonts properly
6. One family or two
7. Details that separate set type from typed text
8. Mistakes the inspector cannot see

## 1. What the study measured

From 202 measurable sites out of the 247 studied (Chrome, 1440×900 and 390×844):

| | Median | Middle half | Note |
| --- | --- | --- | --- |
| Hero headline size | 72px (5.0vw) | 43–109px | Top-scored sites: 6.0vw median |
| Largest type on the page | 6.4vw | up to 10.4vw | Top-scored sites 8.4vw, weak sites 5.0vw |
| Hero headline on a phone | 40px | 32–59px | |
| Hero tracking | -0.01em | -0.03 to 0 | Tighter as size grows |
| Hero leading | 1.0 | 0.95–1.1 | |
| Body size | 16px | 14–18px | |
| Headline ÷ body size | 4.3 | | Top-scored sites 4.7 |
| Typefaces in real use | 2 | 31% use one, 45% two, 22% three | |
| Hero weight | regular 48%, medium 18%, bold or heavier 22% | | Big type does not need to be bold |
| Hero in capitals | 24% | | |
| Hero centred | 39% | | 52% on the weak sites |

Two things to take from it. First, size contrast is the main lever and the best sites pull it harder.
Second, weight is not: half of the heroes are set in a regular weight. Scale and spacing make type
confident; bold is one option among several.

## 2. The scale: few sizes, far apart

The panel described about 30 of the 114 strongest sites as using two or three type sizes with nothing
in between, and none as using a graded five- or six-step scale. So: one display size, one text size
and a small size carry the page; a lead size and a heading size exist for when they are needed. Five
is the ceiling. The common failure is nine sizes that are all near each other (16, 18, 20, 24, 28,
32…), which reads as mush. Choose steps that are clearly different:

```css
:root {
  --step-small: clamp(0.8125rem, 0.79rem + 0.1vw, 0.875rem);   /* 13–14px  captions, legal */
  --step-body:  clamp(1.0625rem, 1rem + 0.25vw, 1.1875rem);    /* 17–19px  paragraphs */
  --step-lead:  clamp(1.25rem, 1.1rem + 0.7vw, 1.625rem);      /* 20–26px  supporting line, large body */
  --step-head:  clamp(2rem, 1.3rem + 3.2vw, 4rem);             /* 32–64px  section headings */
  --step-display: clamp(3rem, 1rem + 9vw, 9rem);               /* set by the hero blueprint */
}
```

- `--step-display` belongs to the direction. A monument wordmark is 16–24vw; a quiet instrument's
  headline is 28px. Use the size the hero blueprint gives, as a `clamp()` so it scales with the
  viewport but is bounded on phones and very wide screens.
- Keep the gap: the display size should be at least four times the body size on desktop, and nothing
  should sit between `--step-lead` and `--step-head` unless the design truly needs a sixth size.
- Resist a "subheading" size for every card and label. If two sizes look almost the same, make them
  the same.

## 3. Setting display type

Large type exposes spacing that is invisible at text size. Apply all of these to anything over 40px:

```css
.display {
  font-size: var(--step-display);
  line-height: 0.95;              /* 0.85–1.05; tighter for capitals and condensed faces */
  letter-spacing: -0.035em;       /* -0.02 to -0.05em; use the value on the direction sheet */
  font-kerning: normal;
  text-wrap: balance;
  max-width: 14ch;                /* choose the measure so lines break where you want */
  margin: 0;
}
```

- **Tracking scales with size.** The larger the type, the tighter it needs to be; a value right for
  120px will choke 32px. Set tracking per step, not once for all headings.
- **Leading below 1.0 needs checking.** Look for collisions between descenders and the ascenders or
  accents of the next line. Two-line headlines make this easy to verify by eye.
- **All capitals** want slightly open tracking (0 to +0.03em) even at display size, and leading around
  0.9.
- **Optical alignment.** Big letters with round or diagonal left edges (O, C, A, T, W) look indented.
  Pull the headline left by about 0.04–0.06em (`margin-left: -0.05em`) so it aligns optically with the
  text beneath it.
- **Viewport-wide type.** For a line that must span the screen, size it in `vw` and verify at 390 and
  1440; set `white-space: nowrap` and let the container clip rather than wrap. Calculate the size from
  the character count: `font-size ≈ container width ÷ (characters × k)` with k from `copy.md` §2.
- Never add a text shadow, an outline stroke, or a gradient fill to display type.

## 4. Setting text

```css
body { font-size: var(--step-body); line-height: 1.5; font-kerning: normal; text-rendering: optimizeLegibility; }
p { max-width: 62ch; text-wrap: pretty; margin: 0; }
p + p { margin-top: 0.9em; }
.lead { font-size: var(--step-lead); line-height: 1.35; max-width: 40ch; }
small, .caption { font-size: var(--step-small); line-height: 1.45; }
```

- Measure: 45–75 characters. Wider than that and the eye loses the next line; the inspector warns
  past about 88.
- Leading: 1.45–1.6 for body, tighter (1.3–1.4) for the larger lead size, looser for long reading.
- Secondary text is the same face in a softer ink (`--muted`), not a lighter weight and not smaller
  than 13px.
- Do not centre paragraphs longer than three lines. Do not justify.
- Light text on a dark ground looks heavier: drop one weight step or add 0.01em tracking, and raise
  leading slightly.

## 5. Loading the fonts properly

The sheet gives a `<link>` tag. Use it exactly; do not swap in a face you prefer.

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="…from the direction sheet…" rel="stylesheet">
```

```css
:root {
  --font-display: "Family From Sheet", "Helvetica Neue", Arial, sans-serif;
  --font-text: "Family From Sheet", Georgia, serif;
}
```

- Request only the weights you use (usually two or three in total).
- Always give a fallback stack of the same broad kind (sans for sans, serif for serif).
- In a framework, use its font loader with the same family and weights (`next/font/google`, or a
  self-hosted `@font-face` with `font-display: swap`). Fontshare faces load from the
  `api.fontshare.com` stylesheet on the sheet.
- After building, `inspect.mjs` reports the families that actually rendered. If it warns that a face
  did not load, the screenshots show a fallback and nothing you judge from them is real. Fix the link
  first.

## 6. One family or two

A third of the studied sites use a single family for everything, and it is the safest way to look
coherent. When the dealt system has two faces:

- Give each a job and never swap them: one sets headlines, the other sets everything else.
- Do not introduce a third face for labels, numbers or quotes.
- Do not mix two faces inside one headline.
- Italic is for titles of works and real emphasis inside paragraphs, never as decoration.

## 7. Details that separate set type from typed text

- Curly quotes and apostrophes, real dashes, a real ellipsis (see `copy.md` §7).
- `font-variant-numeric: tabular-nums` for prices, tables, dates in columns; `oldstyle-nums` suits
  serif body text.
- `text-underline-offset: 0.18em; text-decoration-thickness: 0.07em` on links, so underlines clear the
  descenders.
- `hyphens: none` on headings. `overflow-wrap: anywhere` only on long URLs or emails.
- A heading belongs to the text after it: space above a heading is at least twice the space below.
- `::selection { background: var(--accent); color: var(--accent-ink); }` and a visible
  `:focus-visible` outline in the accent, so even the browser's own states belong to the design.
- Do not change weight on hover; the line reflows.

## 8. Mistakes the inspector cannot see

- A headline that passes the two-line rule but breaks badly: one orphan word on line two, or a break
  that splits a phrase ("Stools for small / kitchens"). Author the break.
- Display type so tight that letters touch, or so loose it falls apart.
- A confident hero followed by section headings almost as large. Keep one size dominant per screen.
- Body text set in the display face when that face was drawn for large sizes only.
- Everything bold. If every line is emphasised, none is.


<!-- colour.md -->
# Colour

The direction sheet deals a palette with named roles. This file is how to apply it so the page looks
decided rather than tinted.

## Contents

1. What the study measured
2. The roles
3. Proportion
4. Surfaces without lines or shadows
5. Colour fields
6. Dark pages
7. Colour and photographs
8. Contrast
9. Banned families, and what "warm" means here

## 1. What the study measured

- 41% of the studied pages are dark, 59% light. A quarter sit on pure white; 12% on pure black.
- Schemes, as the panel classed them: monochrome plus one accent 23%, light neutral 22%, dark neutral
  14%, several flat colours 13%, strictly two-tone 12%, colour carried by photographs 9%, one saturated
  field 6%.
- Buttons are neutral (black, white, grey) on 52% of sites. A coloured button is a choice, not a
  requirement.
- Gradient fills appear on 22% of the top-scored sites and 52% of the weak ones; gradient text on 2%
  against 13%.
- One site in 246 has a terracotta primary button and none has a violet one. Those are colours of
  generated pages, not of the gallery. Warm off-white grounds do occur (32 sites), nearly always with
  a dark or green ink rather than an orange accent.
- Top-scored sites change the page's field colour between sections more often (a median of two section
  grounds against one).

## 2. The roles

```css
:root {
  --bg: …;          /* the ground: most of the surface */
  --surface: …;     /* one tone step from the ground: filled panels, form fields, tiles */
  --ink: …;         /* main text and the strongest marks */
  --muted: …;       /* secondary text; still passes contrast on --bg */
  --accent: …;      /* the one accent */
  --accent-ink: …;  /* text on the accent */
  /* --field-1 … --field-n and their inks, when the palette has section fields */
}
```

Use the tokens for every colour in the build. A raw hex in a component is a second palette starting.
Do not add greys between the roles: if you need a quieter text, it is `--muted`; if you need a quieter
surface, it is `--surface`.

## 3. Proportion

Decide the shares before colouring anything and check them in the screenshot:

- **Ground:** 70–90% of the page.
- **Ink:** text and perhaps one inverted section (ink as ground, ground as ink).
- **Accent:** under 5% of any screen. It marks the action and at most one other recurring thing. The
  moment it appears in five places it stops meaning "here".

Palettes with fields (flat-loud, pastel) work differently: each section is one field at full strength
and the proportion rule applies inside each section (field, its ink, nothing else).

The accent is not for headings, icons, bullets, borders or underlines unless the direction sheet says
so. A headline is ink.

## 4. Surfaces without lines or shadows

With rules and shadows banned, a surface is told apart from the ground by tone alone:

```css
.panel { background: var(--surface); }          /* 2–6 lightness points from --bg */
.field { background: var(--surface); border: 0; } /* inputs: filled, with a clear focus state */
.field:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
```

- One step is enough. If a panel does not read, increase padding or separate with space before
  reaching for a second tone.
- Do not nest a panel inside a panel.
- Images do not need a frame. If a light photograph melts into a light ground, change the crop or set
  it on `--surface`, do not outline it.
- A thick bar (4px or more) in ink or accent is a graphic element, allowed where the direction uses
  it. A 1px line is not.

## 5. Colour fields

A section that changes the whole ground colour is the strongest colour move available, and it replaces
dividers, cards and background decoration at once.

- Fields meet edge to edge with a hard line. Nothing fades.
- Every field has its own assigned ink from the palette. Never put `--muted` on a coloured field; use
  the field's ink at full strength, or at 70% opacity for secondary text if it still passes contrast.
- Neighbouring fields should differ in lightness, not only in hue, or the page vibrates.
- Order them like chapters: open strong, rest, surprise, close dark or very light.
- The page's meta theme colour and the navigation follow the first field.

## 6. Dark pages

- Put `color-scheme: dark` on the root so form controls and scrollbars follow.
- The ground is a deep tone with a slight tint from the palette, and the ink is a soft light, not
  `#fff` on `#000`, unless the palette deals exactly that for a stark two-tone direction.
- Secondary text is where dark pages fail: check `--muted` against the ground (4.5:1).
- Depth on dark comes from a lighter surface step, never from glows or coloured light.
- Use lighter font weights than you would on a light page.

## 7. Colour and photographs

- When photographs carry the colour, the interface stays neutral and takes its ink from the darkest
  tone in the imagery.
- Do not wash a photograph with the brand colour. Either leave it alone, or commit to a real treatment
  for every image on the page (all greyscale, or all untouched).
- Text over a photograph needs a calm region. If there is none, move the text off the image. A neutral
  scrim (black or the ink colour, 30–55% opacity, flat or fading over the text area only) is the one
  gradient the inspector allows.
- A flat field sampled from a photograph (its sky, its shadow) makes a section sit with that image.

## 8. Contrast

- Body text 4.5:1 or better; display type and large numerals 3:1 or better; the dealt palettes meet
  7:1 for ink on ground.
- `inspect.mjs` computes contrast for text on flat grounds and fails anything under 2.6:1. It cannot
  see text over photographs: check those in the screenshots yourself.
- Do not lower opacity on text to make it "soft" unless you have checked the resulting ratio.
- Focus states and hover states increase contrast, never reduce it.

## 9. Banned families, and what "warm" means here

The inspector fails these:

- **Cream with terracotta.** A yellow-cream ground (roughly hue 22–62, light and noticeably
  saturated) with a rust, clay or burnt-orange accent is the house look of AI assistants and of the
  pages they generate. Either half alone is suspect; together they are unmistakable.
- **Violet and indigo.** The first wave of generated pages was purple; it still reads that way.

If the brief wants warmth, get it from a low-chroma stone, bone or putty ground (saturation under
about 15%), from a deep green, brown or oxblood ink, from photographs, or from a full warm field
(mustard, tomato red, pink) used flat and with conviction. Warmth is not beige.

If the brand itself owns one of the banned colours, it is the brand's and you keep it: use
`--keep palette`, pass `--allow palette-purple` (or the relevant rule) to the inspector, and note it in
the report.


<!-- tells.md -->
# Tells — the dated registry

A tell is a device that makes a page read as generated. Tells are not bad design in themselves; most
began as good ideas on good sites. They became tells by being produced by default, everywhere, without
a reason. That is why this registry is dated, and why it never names a replacement font, colour or
component: every published "use this instead" list has become the next wave of tells within months.
Replacements here are procedures (draw from the library, derive from the subject), not products.

`scripts/inspect.mjs` and `scripts/lint.mjs` detect most entries. Status: **banned** means the
inspector fails the page; **warned** means it flags it and you need a reason.

## Contents

1. First wave (2023–2024)
2. Second wave (2025)
3. Third wave (2025–2026): the "fix" that became the look
4. Gallery fashion, banned as house style
5. Structural tells
6. Copy tells
7. Motion tells
8. Why removal is not enough

## 1. First wave (2023–2024)

| Tell | Status | Do instead |
| --- | --- | --- |
| Violet / indigo gradients on hero, buttons and glows | banned | The dealt palette; flat colour |
| Gradient-filled headline text | banned | Solid ink |
| Glass panels: backdrop blur, translucent cards | warned (2 or more) | Opaque surfaces one tone step from the ground |
| Glowing or gradient "Get started" button | banned | Flat button named for its action |
| Soft grey drop shadow under every card | banned | Tone step, or no box |
| Three identical feature cards, each with an icon in a rounded square | banned | The dealt section blueprint |
| Emoji as icons or bullets | banned | Words, or a real picture |
| Centred headline, subtitle, two buttons, screenshot below | no rule; the draw avoids it | The dealt hero |
| The same typeface on every generated page | warned (lint: default-font) | The dealt type system |

## 2. Second wave (2025)

| Tell | Status | Do instead |
| --- | --- | --- |
| Eyebrow / kicker: tiny tracked uppercase label above every heading | banned | Fold it into the heading, or delete |
| Pill badge above the hero headline ("New", "Beta", "Introducing") | banned | Say it in the supporting sentence |
| One word of the headline in italic serif, colour or underline | warned | One face, one colour |
| Bento grid of rounded tiles as the feature section | warned as card rows | A composed section; tiles only in the dense-tiles direction, filled, strict sizes |
| Near-black ground with one neon accent and a glow | banned via shadow, gradient | A dealt dark palette, no glow |
| Floating rounded pill navigation with blur | warned | A plain bar |
| Fake dashboard hero built from grey bars and donut charts | no rule; see imagery.md §7 | A real interface fragment, or no image |
| Decorative status dot ("Available for work", pulsing "live") | lint: bounce | Remove |
| Fade-and-rise on every section | no rule; see motion.md | One signature move |

## 3. Third wave (2025–2026): the "fix" that became the look

The remedy published for the first two waves was "warm it up and make it editorial". Models adopted
the remedy wholesale and it is now the most recognisable generated look of all.

| Tell | Status | Do instead |
| --- | --- | --- |
| Yellow-cream ground + terracotta / rust / clay accent | banned together; terracotta banned alone; cream warned | The dealt palette. Warmth from a low-chroma stone, a deep ink, a photograph, or a full warm field |
| High-contrast or soft serif display as the shortcut to "premium" | no rule | The dealt type system, whatever class it is |
| Oversized italic serif headline | no rule | As dealt |
| Broadsheet cosplay: hairline rules everywhere, zero radius, dense columns | banned via hairline | Structure from space and tone |
| "Field notes", "Quietly trusted by", "On our desks": performative labels | lint: copy | Plain section names |

## 4. Gallery fashion, banned as house style

These are common on award sites right now. In the study they appear *more* often on the top-scored
sites than on the weak ones (panel observations: monospace labels 44% against 24%, index numbering 38%
against 21%; measured thin rules 61% against 52%). They are banned here regardless, by the owner's
decision: they are the first things a generator copies from the gallery, they carry no information,
and a page built without them has to find real structure.

| Tell | Status | Do instead |
| --- | --- | --- |
| Monospace micro-labels, mono as a "technical" voice, mono captions | banned | The text face; tabular figures for data |
| Index numbering: 01 / 02 / 03, (001), "Step 02", numbered section markers | banned | Order by position, scale, or the words; `<ol>` when it is truly a list |
| Thin rules: hairline dividers, ruled grids, 1px outlined boxes and tiles | banned (boxes: 3 or more) | Space, tone steps, filled shapes, a thick bar if a bar is the idea |
| Design-tool chrome as ornament: crosshairs, crop marks, selection handles, dashed guides, rulers | banned via hairline, dashed | Remove |
| Agency micro-meta: local time, coordinates, weather, version stamps, "©2026 — Scroll", middle-dot strings | lint: copy; micro-label | Remove |
| Tiny uppercase letter-spaced captions scattered around the viewport | banned (3 or more) | Sentence case at readable size, placed with the content it describes |

Tiny tracked capitals are the one item in this group the data condemns as well: 16% of top sites
against 57% of weak ones.

## 5. Structural tells

| Tell | Status | Do instead |
| --- | --- | --- |
| Headline as a full sentence at display size, three or four lines | banned (over two lines) | A title of about five words; the rest goes below |
| Stock section order: hero, logos, features, testimonial, pricing, CTA | no rule; see copy.md §1 | The order of the argument |
| Every section the same layout family | critique.md pass 9 | Different composition per section, from the sheet |
| Everything centred | warned over 72% | Left-anchored, one centred moment |
| One medium border radius (12–16px) on everything | none | Square or fully round, chosen once |
| Nine nearly equal font sizes | warned over 9 | Five sizes, far apart |
| Invented social proof: logos, quotes, star rows, "10,000+ teams" | copy.md §5 | Real proof, or none |
| Text or a CSS box where a picture belongs | critique.md cap | A real picture, or a design that needs none |

## 6. Copy tells

Flagged by `lint.mjs`: the genre's verbs and adjectives (unlock, supercharge, seamless, elevate,
effortless, cutting-edge, all-in-one …); the correction ("not just X, it's Y"); tricolon headlines;
"Get started" and "Learn more" as labels; more than a handful of em dashes; lorem ipsum. See
`copy.md` §6.

## 7. Motion tells

Identical fade-up on every block; a dozen staggered hero entrances; hover lift on every card; bounce
easing; infinite loops that mean nothing; reveals that fail and leave a blank band. See `motion.md` §9.

## 8. Why removal is not enough

A page with every tell removed and nothing put in their place is an empty page with good manners, and
that is a tell too (the "avoidance aesthetic": anyone could guess it from the category plus the list of
things to avoid). The tells were filling a gap where an idea should have been. The order of work in
`SKILL.md` exists to fill that gap first: the idea from the subject, the dealt composition, real
pictures or committed type, and only then the checks.

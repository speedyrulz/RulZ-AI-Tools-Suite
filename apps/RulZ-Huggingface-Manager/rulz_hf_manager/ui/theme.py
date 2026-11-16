"""The RulZ look, shared with the other RulZ apps: black and red (dark, the default) or white and red (light).

The colors are CSS variables (--rz-*), defined for the dark look on :root and redefined for the light look on
body.body--light, the class Quasar puts on <body>. Everything else styles itself with those variables, so switching
the look is only Quasar's dark mode switching.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from nicegui import app, ui

from .. import config

STATIC = Path(__file__).resolve().parent.parent / 'static'
STATIC_URL = '/rulz-static'
LOGO_DARK = f'{STATIC_URL}/logo-dark.png'    # white lettering, for the dark look
LOGO_LIGHT = f'{STATIC_URL}/logo-light.png'  # black lettering, for the light look
FAVICON = STATIC / 'icon.png'
RED = '#ED000D'
RED_BRIGHT = '#FF2633'

# 'auto' (follow the system) is still understood for settings files written before the look was Dark or Light.
DARK_VALUE: dict[str, bool | None] = {'auto': None, 'light': False, 'dark': True}

CSS = r'''
:root {
  --rz-red: #ED000D; --rz-red-bright: #FF2633; --rz-red-soft: rgba(237, 0, 13, .13); --rz-red-line: rgba(237, 0, 13, .40);
  --rz-link: #ff5961;
  --rz-bg: #050505; --rz-surface: #0c0c0e; --rz-surface-2: #141417; --rz-hover: #1c1c21;
  --rz-border: #232328; --rz-border-strong: #34343b; --rz-hover-border: #34343b;
  --rz-text: #f2f2f3; --rz-text-2: #d4d4d8; --rz-heading: #f2f2f3; --rz-muted: #a3a3ab; --rz-faint: #82828c; --rz-label: #82828c;
  --rz-header-bg: #050505; --rz-field-bg: #0c0c0e; --rz-tooltip-bg: #141417; --rz-menu-bg: #141417;
  --rz-deep: #0a0a0c; --rz-code-bg: #141417; --rz-code: #ff6b72; --rz-scroll: #34343b; --rz-track: rgba(255, 255, 255, .08);
  --rz-ok: #4ade80; --rz-ok-text: #4ade80; --rz-warn: #fbbf24; --rz-warn-text: #fcd679; --rz-err-text: #ff8a8e;
  --rz-danger: #ff5a5f; --rz-danger-text: #ff8a8e; --rz-danger-bg: #3a0c0f; --rz-danger-border: #6b1418; --rz-danger-hover: #4a1014;
  --rz-file-icon: #82828c; --rz-weight-icon: #f2f2f3; --rz-shadow: 0 24px 64px rgba(0, 0, 0, .9);
  --q-primary: #ED000D; --q-positive: #4ade80; --q-warning: #fbbf24; --q-negative: #ff5a5f;
}
body.body--light {
  --rz-red-bright: #C8000B; --rz-red-soft: rgba(237, 0, 13, .08); --rz-red-line: rgba(237, 0, 13, .35);
  --rz-link: #d0000b;
  --rz-bg: #FFFFFF; --rz-surface: #F5F5F7; --rz-surface-2: #EBEBEF; --rz-hover: #e2e2e7;
  --rz-border: #D9D9DE; --rz-border-strong: #c4c4cb; --rz-hover-border: #c4c4cb;
  --rz-text: #111114; --rz-text-2: #2a2a30; --rz-heading: #111114; --rz-muted: #5F6068; --rz-faint: #68696f; --rz-label: #68696f;
  --rz-header-bg: #FFFFFF; --rz-field-bg: #F5F5F7; --rz-tooltip-bg: #F5F5F7; --rz-menu-bg: #EBEBEF;
  --rz-deep: #F5F5F7; --rz-code-bg: #EBEBEF; --rz-code: #C8000B; --rz-scroll: #c4c4cb; --rz-track: rgba(0, 0, 0, .08);
  --rz-ok: #15803d; --rz-ok-text: #15803d; --rz-warn: #a16207; --rz-warn-text: #8a5300; --rz-err-text: #b4000a;
  --rz-danger: #d4161c; --rz-danger-text: #b4000a; --rz-danger-bg: #fdecec; --rz-danger-border: #f5b8bb; --rz-danger-hover: #fbd8da;
  --rz-file-icon: #68696f; --rz-weight-icon: #2a2a30; --rz-shadow: 0 24px 64px rgba(0, 0, 0, .18);
  --q-positive: #15803d; --q-warning: #a16207; --q-negative: #d4161c;
}
html { background: #050505; scrollbar-width: thin; scrollbar-color: var(--rz-border-strong) transparent; }
html:has(body.body--light) { background: #FFFFFF; }
body, .q-layout, .q-page-container { background: var(--rz-bg); }
body {
  color: var(--rz-text); -webkit-font-smoothing: antialiased; font-size: 14px;
  font-family: "Segoe UI Variable Text", "Segoe UI", Inter, system-ui, sans-serif;
}
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--rz-border-strong); border-radius: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::selection { background: rgba(237, 0, 13, .40); }
:focus-visible { outline: 2px solid var(--rz-red); outline-offset: 2px; }

/* header: logo, page links, open box, activity, look switch, account */
.nicegui-header.rz-header {
  background: var(--rz-header-bg); color: var(--rz-text); border-bottom: 1px solid var(--rz-border);
  box-shadow: none; padding: 0;
}
.rz-header-row { width: 100%; max-width: 1400px; margin: 0 auto; padding: 10px 16px; min-height: 60px; }
.rz-brand { color: inherit; text-decoration: none; }
.rz-logo { height: 24px; width: auto; display: block; }
.rz-wordmark {
  font-family: "Segoe UI Variable Display", "Segoe UI", system-ui, sans-serif;
  letter-spacing: .2em; font-weight: 600; font-size: 13px; text-transform: uppercase;
  color: var(--rz-muted); padding-top: 1px; white-space: nowrap;
}
.q-btn.rz-nav-btn { color: var(--rz-muted); font-weight: 500; padding: 4px 12px; }
.q-btn.rz-nav-btn:hover { color: var(--rz-heading); }
.q-btn.rz-nav-btn.rz-on { color: var(--rz-heading); background: var(--rz-red-soft); }
.q-btn.rz-nav-btn.rz-on .q-icon { color: var(--rz-red-bright); }
.q-btn.rz-icon-btn { color: var(--rz-muted); }
.q-btn.rz-icon-btn:hover { color: var(--rz-heading); }
.rz-open-input .q-field__control { background: var(--rz-field-bg); }
.rz-open-input .q-field__prepend .q-icon { color: var(--rz-muted); }

/* cards, dialogs, drawer */
.q-card.rulz-card {
  background: var(--rz-surface); color: var(--rz-text); border: 1px solid var(--rz-border); border-radius: 12px;
  box-shadow: none;
}
.q-dialog .q-card.rulz-card { border-color: var(--rz-border-strong); box-shadow: var(--rz-shadow); }
.q-card.rulz-card.rz-welcome { background: var(--rz-red-soft); border-color: var(--rz-red-line); border-left: 4px solid var(--rz-red); }
.q-drawer { background: var(--rz-surface); color: var(--rz-text); }
.q-drawer--right.q-drawer--bordered { border-left: 1px solid var(--rz-border); }
.q-card.rulz-card.rz-job { background: var(--rz-surface-2); }
.q-separator { background: var(--rz-border); }
.rz-page-title, .rz-title, .q-dialog .text-lg.font-semibold {
  font-family: "Segoe UI Variable Display", "Segoe UI", system-ui, sans-serif;
}
.rz-page-title { font-size: 22px; font-weight: 600; color: var(--rz-heading); }
.rz-title { font-size: 16px; font-weight: 600; color: var(--rz-heading); }
.q-dialog .text-lg.font-semibold { color: var(--rz-heading); }
.rz-sec-icon {
  color: var(--rz-red-bright); background: var(--rz-red-soft); border-radius: 10px; font-size: 20px;
  width: 36px; height: 36px; flex-shrink: 0;
}
.rz-kind-tile { color: var(--rz-red-bright); background: var(--rz-red-soft); border-radius: 12px; width: 52px; height: 52px; flex-shrink: 0; }

/* stat tiles on My Hub (click one to show only that type) */
.rulz-stat {
  min-width: 160px; flex: 1 1 160px; background: var(--rz-surface-2); border: 1px solid var(--rz-border);
  border-radius: 12px; padding: 10px 14px; cursor: pointer; transition: border-color .15s, background .15s;
}
.rulz-stat:hover { border-color: var(--rz-hover-border); }
.rulz-stat.rz-on { border-color: var(--rz-red); background: var(--rz-red-soft); }
.rz-stat-value { font-size: 15px; font-weight: 600; color: var(--rz-heading); }
.rz-stat-sub { font-size: 12px; color: var(--rz-muted); }

/* tables */
.q-table__card { background: var(--rz-surface); color: var(--rz-text); border-radius: 12px; box-shadow: none; }
.q-table--bordered { border: 1px solid var(--rz-border); }
.q-table thead tr th {
  font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--rz-muted);
}
.q-table thead tr th, .q-table tbody td, .q-table__bottom, .q-table thead tr:last-child th { border-color: var(--rz-border); }
.q-table__bottom { color: var(--rz-muted); }
.q-table tbody tr:not(.q-tr--no-hover):hover > td:not(.q-td--no-hover):before { background: var(--rz-hover); }
.q-table tbody tr.selected > td:after, .q-table--dark tbody tr.selected > td:after { background: var(--rz-red-soft); }
.rulz-table tbody tr { cursor: default; }
/* a table that fills a flex column and scrolls inside it (the local file picker) */
.rz-fill-table { min-height: 0; }
.rz-fill-table .q-table__middle { flex: 1 1 0; min-height: 0; }
/* an icon name the font lacks renders as wide blank text that spills over the row's checkbox: keep it in its box */
.rz-ic-file, .rz-ic-dir, .rz-ic-weight { overflow: hidden; }
.rulz-name:hover span { text-decoration: underline; text-decoration-color: var(--rz-red-bright); text-underline-offset: 3px; }

/* public / private pill, storage badges, chips */
.rz-vis {
  display: inline-flex; align-items: center; gap: 4px; font-size: 12px; font-weight: 600; line-height: 20px;
  padding: 0 9px 0 7px; border-radius: 999px; border: 1px solid var(--rz-border-strong); color: var(--rz-text-2);
  white-space: nowrap;
}
.rz-vis .q-icon { font-size: 14px; color: var(--rz-muted); }
.rz-vis.private { border-color: var(--rz-red-line); color: var(--rz-red-bright); background: var(--rz-red-soft); }
.rz-vis.private .q-icon { color: var(--rz-red-bright); }
.q-badge.rz-badge { background: transparent; border: 1px solid var(--rz-border-strong); color: var(--rz-muted); }
.q-badge.rz-badge.rz-new { border-color: var(--rz-red-line); color: var(--rz-red-bright); }
.q-chip--outline { border-color: var(--rz-border-strong); color: var(--rz-text-2); }
.q-chip--outline .q-chip__icon { color: var(--rz-muted); }
.q-chip.rz-shortcut:hover { border-color: var(--rz-red); }

/* form fields, buttons, toggles, tabs, menus */
.q-field--outlined .q-field__control { border-radius: 8px; background: var(--rz-field-bg); }
.q-field--outlined .q-field__control:before { border-color: var(--rz-border); }
.q-field__native::placeholder, .q-field__input::placeholder { color: var(--rz-faint); opacity: 1; }
.q-field__label { color: var(--rz-label); }
.q-field__native, .q-field__input, .q-field__prefix, .q-field__suffix { color: var(--rz-text); }
.q-field--dark .q-field__native, .q-field--dark .q-field__input { color: #f2f2f3; }
.q-btn { text-transform: none; font-weight: 500; letter-spacing: 0; border-radius: 8px; }
.q-btn.q-btn--dense { border-radius: 6px; }
.q-btn.q-btn--round { border-radius: 50%; }
/* NiceGUI gives every button the primary color. Filled buttons stay red, with the launcher glow; flat ones get
   neutral text (in the layer block at the end), outline ones a red border and link text, danger ones the danger wash. */
.q-btn.bg-primary:not(.q-btn--flat):not(.q-btn--outline) {
  box-shadow: 0 6px 20px -8px rgba(237, 0, 13, .8);
}
.q-btn-toggle .q-btn.bg-primary { box-shadow: none; }
.q-btn.bg-primary.disabled, .q-btn.bg-primary:disabled { box-shadow: none; }
.q-btn.disabled, .q-btn:disabled { opacity: .45; }
.q-btn.bg-primary:not(.q-btn--flat):not(.q-btn--outline):hover { background: var(--rz-red-bright); }
.q-btn-toggle .q-btn.bg-primary:hover { background: var(--rz-red); }
.q-btn.bg-negative {
  background: var(--rz-danger-bg); color: var(--rz-danger-text); border: 1px solid var(--rz-danger-border);
}
.q-btn.bg-negative:hover { background: var(--rz-danger-hover); }
.q-btn--flat.text-primary .q-icon { color: var(--rz-red-bright); }
.q-btn--outline.text-primary:before { border-color: var(--rz-red); }
.q-btn--outline.text-primary:hover { background: var(--rz-red-soft); }
.q-btn--outline.text-primary:hover:before { border-color: var(--rz-red); }
.q-btn--outline.text-negative:before { border-color: var(--rz-danger-border); }
.q-btn--flat.q-btn--round.text-primary .q-icon { color: var(--rz-muted); }
.q-btn--flat.q-btn--round.text-primary:hover .q-icon { color: var(--rz-heading); }
.q-btn-toggle { border: 1px solid var(--rz-border-strong); }
.q-btn-toggle .q-btn:not(.bg-primary) { color: var(--rz-text-2); }
.q-toggle__label, .q-checkbox__label, .q-radio__label { color: var(--rz-text-2); }
.q-tooltip {
  background: var(--rz-tooltip-bg); border: 1px solid var(--rz-border-strong); color: var(--rz-text);
  font-size: 13px; max-width: 440px; line-height: 1.45;
}
.q-menu { background: var(--rz-menu-bg); border: 1px solid var(--rz-border); color: var(--rz-text); border-radius: 8px; }
.rz-tabs { border-bottom: 1px solid var(--rz-border); }
.rz-tabs .q-tab { color: var(--rz-muted); min-height: 42px; }
.rz-tabs .q-tab__label { font-weight: 500; }
.rz-tabs .q-tab--active { color: var(--rz-heading); }
.rz-tabs .q-tab--active .q-tab__icon { color: var(--rz-red-bright); }
.rz-tabs .q-tab__indicator { background: var(--rz-red); height: 3px; border-radius: 3px 3px 0 0; }
.rulz-crumb .q-btn { color: var(--rz-text-2); }
.rulz-crumb .q-btn:last-child { color: var(--rz-heading); }
.q-linear-progress { color: var(--rz-red); }
.q-linear-progress__track { background: var(--rz-track); opacity: 1; }
.q-uploader { background: var(--rz-surface-2); border: 1px dashed var(--rz-border-strong); border-radius: 12px; box-shadow: none; }
.rulz-drop .q-uploader { width: 100%; }

/* text helpers */
.rulz-muted { color: var(--rz-muted); }
.rz-warn { color: var(--rz-warn-text); }
.rz-ok { color: var(--rz-ok-text); }
.rz-err { color: var(--rz-err-text); }
.rulz-mono, .nicegui-markdown code, .nicegui-markdown pre {
  font-family: "Cascadia Code", "Cascadia Mono", Consolas, ui-monospace, monospace;
}
.rz-box { border: 1px solid var(--rz-border); border-radius: 8px; background: var(--rz-deep); }
.rz-banner { border-radius: 10px; padding: 10px 14px; font-size: 13px; }
.rz-banner.info { background: var(--rz-surface); border: 1px solid var(--rz-border-strong); color: var(--rz-text-2); }
.rz-banner.info .q-icon { color: var(--rz-red-bright); }
.nicegui-markdown a, .rz-link { color: var(--rz-link); }
.nicegui-markdown code { background: var(--rz-code-bg); color: var(--rz-code); padding: 1px 6px; border-radius: 5px; }
.nicegui-markdown pre { background: var(--rz-deep); border: 1px solid var(--rz-border); border-radius: 10px; }
.nicegui-markdown pre code { background: transparent; color: var(--rz-text-2); padding: 0; }

/* "Choose your look" (first start) */
.rz-look-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; width: 100%; }
.rz-look {
  border: 2px solid var(--rz-border-strong); border-radius: 12px; padding: 10px; cursor: pointer; gap: 10px;
  background: var(--rz-surface); transition: border-color .15s, box-shadow .15s;
}
.rz-look:hover { border-color: var(--rz-hover-border); }
.rz-look.selected { border-color: var(--rz-red); box-shadow: 0 0 0 3px rgba(237, 0, 13, .18); }
.rz-look-name { font-weight: 600; font-size: 15px; color: var(--rz-heading); }
.rz-look-note { font-size: 12px; color: var(--rz-muted); }
.rz-check { display: none; color: var(--rz-red-bright); font-size: 20px; }
.rz-look.selected .rz-check { display: inline-flex; }
.rz-mini {
  border-radius: 9px; height: 104px; padding: 10px; display: flex; flex-direction: column; gap: 7px;
  border: 1px solid #232328; overflow: hidden; align-self: stretch;
}
.rz-mini img { height: 13px; width: auto; align-self: flex-start; }
.rz-mini.dark { background: #050505; }
.rz-mini.light { background: #FFFFFF; border-color: #D9D9DE; }
.rz-mini .bar { height: 8px; border-radius: 4px; }
.rz-mini.dark .bar { background: #1c1c21; }
.rz-mini.light .bar { background: #EBEBEF; }
.rz-mini .bar.red { width: 34%; background: #ED000D; height: 14px; margin-top: auto; }
.rz-mini .panel { border-radius: 6px; flex: 1; padding: 6px; display: flex; flex-direction: column; gap: 5px; }
.rz-mini.dark .panel { background: #0c0c0e; border: 1px solid #232328; }
.rz-mini.light .panel { background: #F5F5F7; border: 1px solid #D9D9DE; }

/* Rules that must beat Quasar's own !important utilities (text-primary and the like). NiceGUI loads those into its
   last cascade layer, and among !important rules the earliest layer wins, so these go into the first one. */
@layer theme {
  .rz-light-only { display: none !important; }
  body.body--light .rz-light-only { display: inline-flex !important; }
  body.body--light .rz-dark-only { display: none !important; }
  .q-btn--flat.text-primary, .q-chip--outline.text-primary { color: var(--rz-text-2) !important; }
  .q-btn--outline.text-primary, .q-btn--outline.text-primary .q-icon { color: var(--rz-link) !important; }
  .q-btn.bg-primary:not(.q-btn--flat):not(.q-btn--outline):hover { background: var(--rz-red-bright) !important; }
  .q-btn-toggle .q-btn.bg-primary:hover { background: var(--rz-red) !important; }
  .q-btn.bg-negative { background: var(--rz-danger-bg) !important; color: var(--rz-danger-text) !important; border: 1px solid var(--rz-danger-border) !important; }
  .q-btn.bg-negative .q-icon { color: var(--rz-danger-text) !important; }
  .q-btn.bg-negative:hover { background: var(--rz-danger-hover) !important; }
  .q-btn--flat.text-negative, .q-btn--outline.text-negative,
  .q-btn--flat.text-negative .q-icon, .q-btn--outline.text-negative .q-icon { color: var(--rz-danger-text) !important; }
  .rz-kind-icon, .rz-ic-dir { color: var(--rz-red-bright) !important; }
  .rz-ic-weight { color: var(--rz-weight-icon) !important; }
  .rz-ic-file { color: var(--rz-file-icon) !important; }
  .rz-ic-muted { color: var(--rz-muted) !important; }
}
'''


def setup() -> None:
    """Once per process, before the first page: the logo files, the brand colors and the stylesheet."""
    app.add_static_files(STATIC_URL, STATIC)
    app.colors(primary=RED, secondary='#141417', accent=RED_BRIGHT, dark='#0c0c0e', dark_page='#050505',
               positive='#4ade80', negative='#ff5a5f', info='#3F3F46', warning='#fbbf24')
    ui.add_css(CSS, shared=True)


def saved_dark() -> bool | None:
    """Quasar's dark setting for the saved look (True for dark, the default)."""
    return DARK_VALUE.get(config.get().dark_mode, True)


def save_look(light: bool) -> None:
    settings = config.get()
    settings.dark_mode = 'light' if light else 'dark'
    settings.look_chosen = True
    config.save()
    app.config.dark = not light  # pages opened from now on start in this look (no flash of the other one)


class PageLook:
    """The look of one browser tab: Quasar's dark mode, plus whoever wants to hear when it changes."""

    def __init__(self) -> None:
        self.dark = ui.dark_mode(saved_dark())
        self._listeners: list[Callable[[bool], None]] = []

    @property
    def light(self) -> bool:
        return self.dark.value is False

    def on_change(self, fn: Callable[[bool], None]) -> None:
        self._listeners.append(fn)

    def set(self, light: bool, save: bool = True) -> None:
        self.dark.set_value(not light)
        if save:
            save_look(light)
        for fn in self._listeners:
            fn(light)

    def toggle(self) -> None:
        self.set(not self.light)


def choose_look(look: PageLook) -> None:
    """First start: pick Dark (the default) or Light. Picking previews it; closing the dialog keeps Dark."""
    picked = {'light': False, 'saved': False}
    with ui.dialog() as dialog, ui.card().classes('rulz-card gap-4 p-6').style('width: 540px; max-width: 94vw'):
        ui.label('Choose your look').classes('rz-page-title')
        ui.label('You can change it any time with the sun / moon button at the top right, or in Settings.') \
            .classes('text-sm rulz-muted')
        cards = {}
        with ui.element('div').classes('rz-look-grid'):
            for light, name, note in ((False, 'Dark', 'Black and red (default)'), (True, 'Light', 'White and red')):
                with ui.column().classes('rz-look') as card:
                    with ui.element('div').classes('rz-mini ' + ('light' if light else 'dark')):
                        ui.element('img').props(f'src="{LOGO_LIGHT if light else LOGO_DARK}" alt=""')
                        with ui.element('div').classes('panel'):
                            ui.element('div').classes('bar')
                            ui.element('div').classes('bar').style('width: 70%')
                        ui.element('div').classes('bar red')
                    with ui.row().classes('items-center w-full no-wrap gap-2'):
                        with ui.column().classes('gap-0'):
                            ui.label(name).classes('rz-look-name')
                            ui.label(note).classes('rz-look-note')
                        ui.space()
                        ui.icon('check_circle').classes('rz-check')
                cards[light] = card
                card.on('click', lambda _, lt=light: pick(lt))

        def pick(light: bool) -> None:
            picked['light'] = light
            for lt, c in cards.items():
                c.classes(add='selected') if lt == light else c.classes(remove='selected')
            look.set(light, save=False)  # a live preview; saved on Continue

        def done() -> None:
            picked['saved'] = True
            look.set(picked['light'])
            dialog.close()

        def closed() -> None:
            if not picked['saved']:  # dismissed without Continue: Dark, saved
                look.set(False)

        with ui.row().classes('w-full justify-end'):
            ui.button('Continue', on_click=done).props('unelevated color=primary')
        dialog.on('hide', closed)
    cards[False].classes(add='selected')
    look.set(False, save=False)
    dialog.open()

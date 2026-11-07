"""Repo / bucket level actions shared by the dashboard and the browser page."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import run, ui

from .. import hub
from .common import browse_url, confirm, notify_error, notify_ok, prompt


def _id_error(v: str) -> str | None:
    if not hub._ID_RE.match(v):
        return 'Use the form "namespace/name" (letters, digits, "-", "_" and ".").'
    return None


async def create_dialog(namespaces: list[str], default_namespace: str = '', default_kind: str = 'model') -> None:
    if not namespaces:
        notify_error('Add a token with write access in Settings first.')
        return
    with ui.dialog() as dialog, ui.card().classes('rulz-card w-[560px] max-w-[95vw] gap-3'):
        ui.label('Create new').classes('text-lg font-semibold')
        kind = ui.toggle({k: hub.KIND_LABEL[k] for k in hub.KINDS}, value=default_kind).props('no-caps spread')
        with ui.row().classes('w-full items-center no-wrap gap-2'):
            ns = ui.select(namespaces, value=default_namespace if default_namespace in namespaces else namespaces[0],
                           label='Owner').props('outlined dense').classes('w-48')
            ui.label('/').classes('text-xl rulz-muted')
            name = ui.input('Name').props('outlined dense autofocus').classes('flex-1')
        private = ui.switch('Private', value=True)
        sdk = ui.select(hub.SPACE_SDKS, value='gradio', label='Space SDK').props('outlined dense').classes('w-48')
        sdk.bind_visibility_from(kind, 'value', value='space')
        note = ui.label('Buckets are S3-like storage: no git history, fast server-side copies and sync.') \
            .classes('text-xs rulz-muted')
        note.bind_visibility_from(kind, 'value', value='bucket')
        error = ui.label().classes('rz-err text-sm')

        async def create() -> None:
            repo_id = f'{ns.value}/{(name.value or "").strip()}'
            if msg := _id_error(repo_id):
                error.text = msg
                return
            btn.props('loading')
            try:
                await run.io_bound(hub.create_repo, kind.value, repo_id, private.value, sdk.value)
            except Exception as e:  # noqa: BLE001
                error.text = hub.friendly_error(e)
                btn.props(remove='loading')
                return
            dialog.submit(True)
            notify_ok(f'Created {hub.KIND_LABEL[kind.value].lower()} {repo_id}')
            ui.navigate.to(browse_url(kind.value, repo_id))
        name.on('keydown.enter', create)
        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat')
            btn = ui.button('Create', icon='add', on_click=create)
    await dialog


async def rename_dialog(kind: str, repo_id: str, on_done: Callable[[str], None] | None = None) -> None:
    new_id = await prompt(f'Rename / transfer {hub.KIND_LABEL[kind].lower()}', 'New id (namespace/name)', repo_id,
                          ok_label='Rename', validation=_id_error,
                          hint='Change the name to rename, or change the namespace to transfer it to an organization. '
                               'The Hub keeps a redirect from the old URL for repos.')
    if not new_id or new_id == repo_id:
        return
    try:
        await run.io_bound(hub.rename_repo, kind, repo_id, new_id)
    except Exception as e:  # noqa: BLE001
        notify_error(e)
        return
    notify_ok(f'Renamed to {new_id}')
    if on_done:
        on_done(new_id)


async def visibility_dialog(kind: str, repo_id: str, currently_private: bool,
                            on_done: Callable[[bool], None] | None = None) -> None:
    target = not currently_private
    ok = await confirm(f'Make {repo_id} {"private" if target else "public"}?',
                       'Private: only you (and your organization) can see it.' if target else
                       'Public: anyone on the internet can see and download it.',
                       ok_label=f'Make {"private" if target else "public"}', ok_color='warning' if not target else 'primary')
    if not ok:
        return
    try:
        await run.io_bound(hub.set_private, kind, repo_id, target)
    except Exception as e:  # noqa: BLE001
        notify_error(e)
        return
    notify_ok(f'{repo_id} is now {"private" if target else "public"}')
    if on_done:
        on_done(target)


async def duplicate_dialog(kind: str, repo_id: str) -> None:
    ident = None
    try:
        ident = await run.io_bound(hub.whoami)
    except Exception:  # noqa: BLE001
        pass
    default = f'{ident.name if ident else repo_id.split("/")[0]}/{repo_id.split("/")[1]}-copy'
    new_id = await prompt(f'Duplicate {repo_id}', 'New id (namespace/name)', default, ok_label='Duplicate',
                          validation=_id_error, hint='Creates a full server-side copy (private).')
    if not new_id:
        return
    ui.notify(f'Duplicating to {new_id}...', type='ongoing')
    try:
        await run.io_bound(hub.duplicate_repo, kind, repo_id, new_id, True)
    except Exception as e:  # noqa: BLE001
        notify_error(e)
        return
    notify_ok(f'Duplicated to {new_id}')
    ui.navigate.to(browse_url(kind, new_id))


async def delete_dialog(kind: str, repo_id: str, on_done: Callable[[], None] | None = None) -> None:
    ok = await confirm(f'Delete {hub.KIND_LABEL[kind].lower()} {repo_id}?',
                       'This permanently deletes it and all of its files from the Hub. This cannot be undone.',
                       ok_label='Delete forever', ok_color='negative', require_text=repo_id)
    if not ok:
        return
    try:
        await run.io_bound(hub.delete_repo, kind, repo_id)
    except Exception as e:  # noqa: BLE001
        notify_error(e)
        return
    notify_ok(f'Deleted {repo_id}')
    if on_done:
        on_done()

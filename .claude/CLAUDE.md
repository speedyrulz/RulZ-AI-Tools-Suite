# Instructions for Claude

## Shipping: always merge, release and update the wiki

Read `.claude/ship.json` before finishing any change in this repository and follow it. It holds the owner's standing
approval and the exact steps (`approval.default: "auto"`):

1. Open a pull request against `main`, wait for CI, and **merge it yourself** with a merge commit once every check is
   green and there is no merge conflict. Do not ask first. Do not merge while CI is red or still running: fix the
   failure. Never skip, disable or weaken a test to get green.
2. **Release**: bump the changed app's version in the pull request (`version` in ship.json lists the files), then
   push its tag on the merge commit (`{id}-v{version}`, e.g. `rulz-trainer-v3.32.0`; the launcher uses `v{version}`).
   CI builds and publishes the release.
3. **Wiki**: update the app's pages after the release (`wiki` in ship.json: CRLF line endings kept, pages staged by
   name only, no `git add -A` / `commit -a`).

## Cloud sessions

A cloud session's GitHub access may only allow pushing its own branch: tag pushes and the wiki repository can be
refused (HTTP 403). Then still merge, and give the owner the exact tag commands and a wiki patch to apply.

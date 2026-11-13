# Axel

One local brain for Synaps. Axel runs as a
Synaps extension: it recalls what matters before each message, gives the model search and
remember tools, learns from your chats as they happen, and keeps the project forum. It reads
Synaps's archive and note files and never writes them. Nothing leaves this machine.

If Axel is absent, crashed or slow, Synaps loses recall and nothing else: every hook it
answers late is answered "continue".

Axel runs on Linux and macOS only, the systems on which Synaps writes its archives. A build
on any other system stops with one line: `axel builds on Linux and macOS only`.

## What it adds to Synaps

- **Recall before each message.** Before a message goes to the model, Axel looks for earlier
  chats, saved memories and notes that match it and adds the best few (at most 5 items,
  1,600 characters) to the message. It has 150 ms; after that the message goes on without
  recall.
- **Four tools** for the model: `axel_search`, `axel_remember`, `axel_recall`, `axel_forget`.
  A model's `axel_forget` hides a memory; it does not delete it. `axel_forget
  {"stop":"learning"}` (or `"recall"`, `"both"`) pauses Axel for that chat only.
- **Learning from chats.** Whole turns are captured as they happen, and Synaps's context
  archives and notes are indexed in the background.
- **The project forum**: `axel_forum_post`, `axel_forum_read`, `axel_forum_forget`, shared by
  the sessions of one project.

## Requirements

- **Linux or macOS.** The macOS build compiles but has not been run on a Mac yet.
- **What the ONNX Runtime library needs.** Axel links the prebuilt ONNX Runtime of the `ort`
  crate, both in the prebuilt binary and in a `cargo install` build, so these hold for both:
  - Linux: glibc 2.39 or newer and libstdc++ 13.2 or newer (Ubuntu 24.04, Debian 13 or
    newer), on x86_64 or aarch64.
  - x86_64: a CPU with AVX2 (x86-64-v3: Intel Haswell, AMD Zen or newer). Without it ort
    warns, and Axel will likely crash with an illegal instruction.
  - macOS: 13.4 or newer, Apple Silicon only.
- **Synaps 0.12.0 or newer.** Axel uses the hooks `on_context_archived` and
  `on_memory_forgotten`, the `cwd` Synaps sends with each hook and tool call, and the
  `caller` field of `tool.call`. All four are in Synaps's development branch and ship first
  in 0.12.0. Axel does not check Synaps's version; on an older Synaps see the notes below.
- **To build from source** (instead of the prebuilt binary, below): Rust with
  cargo, and a C and C++ compiler (SQLite and the vector index are
  compiled in). The build downloads the ONNX Runtime library once (the `ort` crate's
  prebuilt binaries).

## Install

If you have the old Axel (`axel.r8`, the Synaps memory backend), read
[Upgrading from the old Axel](#upgrading-from-the-old-axel-axelr8) first.

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/HaseebKhalid1507/axel/releases/download/v0.1.0/axel-installer.sh | sh
axel model fetch
axel init
axel plugin install --plugins-dir ~/.synaps-cli/plugins
synaps daemon reload
```

The installer has prebuilt binaries for Linux on x86_64 and aarch64 (glibc) and for macOS
on Apple Silicon. To build from source instead (same requirements), replace the first line
with:

```bash
cargo install --git https://github.com/HaseebKhalid1507/axel --tag v0.1.0 --locked axel
```

Intel Macs, musl Linux (Alpine) and Linux older than glibc 2.39 are not covered: `ort` has
no prebuilt ONNX Runtime for them, so neither the installer nor a plain `cargo install`
works there. A source build linked against an ONNX Runtime you built yourself
(`ORT_LIB_LOCATION`, see <https://ort.pyke.io/setup/linking>) may, but is untested.

Step by step:

1. The installer (or `cargo install`) puts the binary `axel` into `~/.cargo/bin`.
2. `axel model fetch` downloads the embedding model (about 90 MB) into
   `~/.cache/velocirag/models/all-MiniLM-L6-v2`. It is the only command that uses the
   network. Without it Axel still works, on words alone.
3. `axel init` makes an empty brain at `~/.local/share/axel/brain.r8`. If you have the old
   Axel brain, run `axel migrate init --from-axel <axel.r8>` instead: it makes the brain
   from the old one and leaves the old one unchanged.
4. `axel plugin install` writes the plugin manifest into Synaps's plugins folder.
5. `synaps daemon reload` (or restart Synaps) loads the extension.

This writes `~/.synaps-cli/plugins/axel/.synaps-plugin/plugin.json` with `command` set to the
absolute path of the binary you ran, with every link resolved: run it through a symlink and
the manifest names the file the link points to. `--plugins-dir` has no default. An existing manifest is
not replaced unless you add `--force`, which first keeps the old one as
`plugin.json.bak-<stamp>`. `--print` prints the manifest and writes nothing. `--command
<path>` names another binary instead (for packagers); it must be absolute and exist. If the
running binary's path cannot be found, install refuses and asks for `--command`.
[`plugin/plugin.json`](plugin/plugin.json) is a template: the same manifest with the
placeholder `/path/to/axel` as `command`. Do not copy it as is; `plugin install` writes the
real one.

The extension starts with Synaps, answers at once, and indexes in the background.
`axel status` shows what it is doing.

Note folders of your own (outside Synaps) are indexed only if you list them in
`~/.config/axel/sources.toml` (the config key `sources` moves it):

```toml
[[source]]
name = "notes"
path = "~/notes"
priority = "high"   # high, medium or low
```

**A Synaps older than 0.12.0** that does not know the hooks `on_context_archived` and
`on_memory_forgotten` refuses to load the plugin and runs without it. To use Axel there,
delete those two hook lines from `plugin.json` by hand; the background scan then does the
work, up to 10 minutes late.

**A Synaps that sends `caller`.** Every tool, the forum's included, needs a Synaps whose
`tool.call` names the calling run (`caller`, Synaps 0.12.0 and newer). Without it the
tool answers "this Synaps does not say who is calling (tool.call has no caller); Axel's
tools need a host that does" and reads and writes nothing. A tool call Synaps makes for
itself, with no chat behind it, carries no `caller` and gets the same answer. Recall before
each message and learning from chats do not need it.

**A sandbox Synaps.** Set `SYNAPS_BASE_DIR=<dir>`, run `axel plugin install --plugins-dir
<dir>/plugins`, and put `brain = <dir>/axel/brain.r8` in `<dir>/plugins/axel/config`, so the
sandbox never opens your real brain. If you forget the last step, Axel sees that the brain
belongs to another Synaps base and writes nothing.

## Upgrading from the old Axel (axel.r8)

The old Axel kept its brain in `~/.config/axel/axel.r8` (and, with Synaps 0.11.1 and
`memory.backend = axel`, Synaps kept its own archives and notes in a second Axel brain,
`~/.synaps-cli/memory/axel/brain.r8`). This Axel never changes either file.

**Keep the old binary as the rollback copy.** `cargo install` puts the new `axel` in
`~/.cargo/bin`, which may come before the old one on your `PATH`. To keep both, install the
new one beside the old under another name:

```bash
cargo install --git https://github.com/HaseebKhalid1507/axel --tag v0.1.0 --locked \
  --root ~/.local/axel-0.1.0 axel
install -m 0755 ~/.local/axel-0.1.0/bin/axel ~/.local/bin/axel-v2
axel-v2 migrate init --from-axel ~/.config/axel/axel.r8
axel-v2 plugin install --plugins-dir ~/.synaps-cli/plugins --force
```

Run this way, every hint the binary prints names it `axel-v2`. `migrate init` makes the
brain (`brain.r8`) from the old Axel brain; `migrate catch-up --from-axel <axel.r8>` later
brings over what the old binary saved after that. `--force` keeps the old manifest as
`plugin.json.bak-<stamp>`; putting it back returns Synaps to the old binary and the old
brain.

**If your Synaps 0.11.1 used `memory.backend = axel`**, Synaps 0.12.0 no longer reads the
archives and notes in its Axel brain. Before you upgrade Synaps, export them with

```bash
axel synaps-export --brain ~/.synaps-cli/memory/axel/brain.r8 --out <empty folder>
```

(the brain is the `memory.axel.brain` value when your config has one) and import them into
Synaps's plain-file stores as Synaps's `docs/upgrading/0.12.0.md` describes. The export
reads the brain and never writes it. `axel migrate synaps --from <brain.r8>` brings that
brain's forum posts into Axel's brain.

## Where the brain lives

One SQLite file, `~/.local/share/axel/brain.r8` (folder mode 0700, file 0600). The config key
`brain` moves it; on the command line `--brain <file>` or `AXEL_BRAIN` does. The extension's
log is `extension.log` beside the brain: ids, counts, timings and error kinds, never text;
it is renamed to `extension.log.1` at 4 MiB.

The brain records the Synaps base it mirrors. To move a brain to another base on purpose,
change that row:

```sql
UPDATE context_data SET value = '<the new base>' WHERE layer = 'axel' AND key = 'synaps_base';
```

The embedding model is `all-MiniLM-L6-v2`, read from `~/.cache/velocirag/models/all-MiniLM-L6-v2`
(or the config key `model_dir`). Axel never downloads it by itself: `axel model fetch`
does, when you run it. Without the model files, recall and search work on words alone and
`status` says so.

The old Axel brain `~/.config/axel/axel.r8` is not changed by this version; it is the
rollback copy. `axel migrate init --from-axel ~/.config/axel/axel.r8` makes the brain from
it. The `version` that `status` prints is the brain file's format number (1 in this
release), not Axel's version.

## Configuration

Synaps reads each key from, in this order: the environment variable
`SYNAPS_EXTENSION_AXEL_<KEY>`; the plugin's own config file `<plugins>/axel/config` (lines
`<key> = <value>`, `#` starts a comment line, the first line for a key wins); the key
`extension.axel.<key>` in the Synaps config, which still works but Synaps calls deprecated;
then the manifest's default. Switches take `on|off|true|false|1|0`. Paths must be absolute;
a leading `~/` means your home folder. A value Axel cannot read keeps its default and is
listed by `status`.

| key | default | meaning |
|---|---|---|
| `brain` | `~/.local/share/axel/brain.r8` | the brain file |
| `synaps_base` | derived from the plugin folder | the Synaps base to read archives and notes from |
| `recall` | `on` | recall before each message |
| `recall_max_items` | `5` | most items injected per message (1-8) |
| `recall_max_chars` | `1600` | most characters injected per message |
| `recall_budget_ms` | `150` | time allowed for one recall, at most 2000 |
| `recall_min_sem` | `0.30` | least similarity by meaning (0-1) |
| `recall_min_lex` | `0.50` | least share of the prompt's words, when meaning does not pass (0-1) |
| `recall_on_events` | `off` | also recall on host event messages |
| `recall_exclude_paths` | empty | comma-separated folders never recalled or searched |
| `capture` | `turns` | learning from chats as they happen: `turns` or `off` |
| `archives` | `on` | index Synaps's context archives |
| `forum` | `auto` | the forum tools: `auto` (on unless Synaps still uses its Axel backend), `on`, `off`; see below |
| `worker_scope` | `project` | what subagent workers can search: `project` or `all` |
| `forget_now_min_hidden_hours` | `24` | hours a memory must have been hidden before `forget --now` deletes it; `0`: at once. Read by the command only, from `<base>/config` (below) |
| `handoff_on_start` | `off` | inject the handoff written with `axel handoff set` at session start |
| `sources` | `~/.config/axel/sources.toml`; without it, no note folders | the note folders to index |
| `archive_events` | empty | comma-separated event sources to index from archives besides worker reports |
| `embedder` | `onnx` | `onnx` (the local model), `off` (words only), `hash` (tests only) |
| `model_dir` | `~/.cache/velocirag/models/all-MiniLM-L6-v2` | the model folder |
| `log_queries` | `on` | keep the first 120 characters of a prompt in the recall log, for turns that are captured anyway |

The command line reads neither `<plugins>/axel/config` nor the `SYNAPS_EXTENSION_AXEL_<KEY>`
variables. A brain moved with the key `brain` is named to the command line with `--brain` or
`AXEL_BRAIN`. `axel forget --now` takes its hours only from the line
`extension.axel.forget_now_min_hidden_hours = <hours>` in `<base>/config`, where `<base>` is
`SYNAPS_BASE_DIR` when it is set, else the base the brain records, else `~/.synaps-cli`; a
value in `<plugins>/axel/config` does not reach it.

Under `auto`, Axel checks `memory.backend` in `<base>/config` and in the `config` of every
Synaps profile folder (`<base>/<profile>/config`), because it cannot tell which profile is
active: if any of them says `axel`, the forum tools stay off. A base folder with more than
4,096 entries is not checked and the forum tools stay off; `status` names the file (or the
folder) that decided.

### What the extension cannot see

Synaps starts the extension with an empty environment, except `PATH`, `HOME`, `LANG`, `TERM`
and `XDG_RUNTIME_DIR`. So `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `AXEL_BRAIN` have no effect on
it, and the command line ignores the first two as well: the default brain and model folder are
under your home folder on every system. When Synaps itself runs with `SYNAPS_PROJECT_ROOT`
set, Axel cannot see the override: recall, learning from chats, the tools and the forum key
a chat by the folder it runs in, while its archives, notes and forgets are filed under the
override's key, as Synaps files them. Recall in that folder then does not find them.

## Commands

Installed under another name (such as `axel-v2`), the binary is run, and names itself in
its hints, by that name.

| command | what it does |
|---|---|
| `axel extension` | speaks the extension protocol on stdin and stdout; Synaps runs it |
| `axel plugin install --plugins-dir <dir> [--force] [--print] [--command <path>]` | writes the plugin manifest (above) |
| `axel status [--json] [--check]` | the one place to look: the brain, the running extension and its mode, what is indexed and waiting, the model, pauses, what a model hid. `--check` adds an integrity check. Exit 0 when nothing is wrong, 1 when something is |
| `axel why [--last N] [--session <id>] [--json]` | why the last recalls chose what they chose |
| `axel show <id>` | one memory in full, or one archive (`ctx-<id>`) |
| `axel search`, `recall`, `memories`, `remember` | search, recall, list and save memories from the command line |
| `axel forget <id>...` | hides memories, as the tool does |
| `axel forget --now <id>...`, `forget --now --hidden` | deletes for good, only rows hidden for at least `forget_now_min_hidden_hours` |
| `axel restore <id>...` | brings back hidden memories |
| `axel restore --by-model --since <date>` | brings back everything a model hid or superseded since that date |
| `axel restore --forget-turns ctx-<id>` | brings back the turns an archive left out because a forget tool was called in them |
| `axel pause <session id> <learning\|recall\|both> [--force]` | pauses one chat; `--force` pauses a chat the brain knows nothing of yet |
| `axel resume <session id> <learning\|recall\|both> [--catch-up]` | lifts a pause; `--catch-up` also indexes the chat's archives sealed during it |
| `axel migrate init --from-axel <axel.r8>` | makes the brain from the old Axel brain |
| `axel migrate synaps --from <brain.r8>` | brings over the forum posts of Synaps's own brain |
| `axel migrate verify --from-axel <axel.r8> --from-synaps <brain.r8>` | checks that the import is complete |
| `axel migrate catch-up --from-axel <axel.r8>` | brings over what was saved in the old brain after the import |
| `axel model fetch` | downloads the embedding model; the only command that does |

The commands of the first version (`init`, `index`, `index-sync`, `stats`, `consolidate`,
`handoff` and the rest) are still there; `axel --help` lists them all.

## Turning it off

Each key below goes in `<plugins>/axel/config` (or any place listed under Configuration).

- Recall only: `recall = off`.
- Learning from chats only: `capture = off`; archives only: `archives = off`.
- The forum tools only: `forum = off`.
- One chat: `axel pause <session id> both`, or the model's `axel_forget {"stop":"both"}`.
- The whole extension: add `axel` to `disabled_plugins` in the Synaps config, or remove the
  folder `<plugins>/axel`. Putting the old plugin folder back returns Synaps to the old
  binary and the old brain, which this version never changed.

## License

MIT

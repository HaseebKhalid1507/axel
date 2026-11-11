# Axel

One local brain for Synaps. Axel runs as a
Synaps extension: it recalls what matters before each message, gives the model search and
remember tools, learns from your chats as they happen, and keeps the project forum. It reads
Synaps's archive and note files and never writes them. Nothing leaves this machine.

If Axel is absent, crashed or slow, Synaps loses recall and nothing else: every hook it
answers late is answered "continue".

Axel runs on Linux and macOS only, the systems on which Synaps writes its archives. A build
on any other system stops with one line: `axel v2 builds on Linux and macOS only`.

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

## Install

Axel is the binary `axel` of this repository. Build it and install it as
`~/.local/bin/axel-v2`, beside the old `~/.local/bin/axel`, never over it:

```bash
cargo build --release --locked -p axel
install -m 0755 target/release/axel ~/.local/bin/axel-v2
```

Then write the plugin manifest into Synaps's plugins folder:

```bash
axel-v2 plugin install --plugins-dir ~/.synaps-cli/plugins
```

This writes `~/.synaps-cli/plugins/axel/.synaps-plugin/plugin.json` with `command` set to the
absolute path of the binary you ran, with every link resolved: run it through a symlink and
the manifest names the file the link points to. `--plugins-dir` has no default. An existing manifest is
not replaced unless you add `--force`, which first keeps the old one as
`plugin.json.bak-<stamp>`. `--print` prints the manifest and writes nothing. A copy of the
manifest is in [`plugin/plugin.json`](plugin/plugin.json).

Restart Synaps (or reload the extension). The extension starts with Synaps, answers at once,
and indexes in the background.

**An older Synaps** that does not know the hooks `on_context_archived` and
`on_memory_forgotten` refuses to load the plugin and runs without it. To use Axel there,
delete those two hook lines from `plugin.json` by hand; the background scan then does the
work, up to 10 minutes late.

**A Synaps that sends `caller`.** Every tool, the forum's included, needs a Synaps whose
`tool.call` names the calling run (`caller`, from core commit `6375d9af` on). Without it the
tool answers "this Synaps does not say who is calling (tool.call has no caller); Axel's
tools need a host that does" and reads and writes nothing. A tool call Synaps makes for
itself, with no chat behind it, carries no `caller` and gets the same answer. Recall before
each message and learning from chats do not need it.

**A sandbox Synaps.** Set `SYNAPS_BASE_DIR=<dir>`, run `axel-v2 plugin install --plugins-dir
<dir>/plugins`, and put `brain = <dir>/axel/brain.r8` in `<dir>/plugins/axel/config`, so the
sandbox never opens your real brain. If you forget the last step, Axel sees that the brain
belongs to another Synaps base and writes nothing.

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
(or the config key `model_dir`). Axel never downloads it by itself: `axel-v2 model fetch`
does, when you run it. Without the model files, recall and search work on words alone and
`status` says so.

The old brain `~/.config/axel/axel.r8` is not changed by this version; it is the rollback copy.
`axel-v2 migrate init --from-axel ~/.config/axel/axel.r8` makes a version 2 brain from it.

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
| `sources` | `~/.config/axel/sources.toml`, else the built-in list | the note folders to index |
| `archive_events` | empty | comma-separated event sources to index from archives besides worker reports |
| `embedder` | `onnx` | `onnx` (the local model), `off` (words only), `hash` (tests only) |
| `model_dir` | `~/.cache/velocirag/models/all-MiniLM-L6-v2` | the model folder |
| `log_queries` | `on` | keep the first 120 characters of a prompt in the recall log, for turns that are captured anyway |

The command line reads neither `<plugins>/axel/config` nor the `SYNAPS_EXTENSION_AXEL_<KEY>`
variables. A brain moved with the key `brain` is named to the command line with `--brain` or
`AXEL_BRAIN`. `axel-v2 forget --now` takes its hours only from the line
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

The binary's command name is `axel`; installed as `axel-v2`, it is run as below.

| command | what it does |
|---|---|
| `axel-v2 extension` | speaks the extension protocol on stdin and stdout; Synaps runs it |
| `axel-v2 plugin install --plugins-dir <dir> [--force] [--print]` | writes the plugin manifest (above) |
| `axel-v2 status [--json] [--check]` | the one place to look: the brain, the running extension and its mode, what is indexed and waiting, the model, pauses, what a model hid. `--check` adds an integrity check. Exit 0 when nothing is wrong, 1 when something is |
| `axel-v2 why [--last N] [--session <id>] [--json]` | why the last recalls chose what they chose |
| `axel-v2 show <id>` | one memory in full, or one archive (`ctx-<id>`) |
| `axel-v2 search`, `recall`, `memories`, `remember` | search, recall, list and save memories from the command line |
| `axel-v2 forget <id>...` | hides memories, as the tool does |
| `axel-v2 forget --now <id>...`, `forget --now --hidden` | deletes for good, only rows hidden for at least `forget_now_min_hidden_hours` |
| `axel-v2 restore <id>...` | brings back hidden memories |
| `axel-v2 restore --by-model --since <date>` | brings back everything a model hid or superseded since that date |
| `axel-v2 restore --forget-turns ctx-<id>` | brings back the turns an archive left out because a forget tool was called in them |
| `axel-v2 pause <session id> [learning\|recall\|both]` | pauses one chat |
| `axel-v2 resume <session id> [learning\|recall\|both] [--catch-up]` | lifts a pause; `--catch-up` also indexes the chat's archives sealed during it |
| `axel-v2 migrate init --from-axel <axel.r8>` | makes a version 2 brain from the old one |
| `axel-v2 migrate synaps --from <brain.r8>` | brings over the forum posts of Synaps's own brain |
| `axel-v2 migrate verify --from-axel <axel.r8> --from-synaps <brain.r8>` | checks that the import is complete |
| `axel-v2 migrate catch-up --from-axel <axel.r8>` | brings over what was saved in the old brain after the import |
| `axel-v2 model fetch` | downloads the embedding model; the only command that does |

The commands of the first version (`init`, `index`, `index-sync`, `stats`, `consolidate`,
`handoff` and the rest) are still there; `axel-v2 --help` lists them all.

## Turning it off

Each key below goes in `<plugins>/axel/config` (or any place listed under Configuration).

- Recall only: `recall = off`.
- Learning from chats only: `capture = off`; archives only: `archives = off`.
- The forum tools only: `forum = off`.
- One chat: `axel-v2 pause <session id>`, or the model's `axel_forget {"stop":"both"}`.
- The whole extension: add `axel` to `disabled_plugins` in the Synaps config, or remove the
  folder `<plugins>/axel`. Putting the old plugin folder back returns Synaps to the old
  binary and the old brain, which this version never changed.

## License

MIT

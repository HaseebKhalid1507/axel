# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2026-11-27

Axel for Synaps 0.13.0, which saves extension context in the conversation by default: recall
stays turn-only, and saved extension context in archives is never read as the person's words.
Synaps 0.12.0 keeps working exactly as with 0.1.0.

- **Recall is shown for one turn on a Synaps core with `inject.forget`.** When the core lists
  `inject.forget` in `initialize` `features` (it saves extension context by default), the
  `before_message` recall block says `forget: true`: the model sees it for the whole turn and
  the core leaves its marker in the conversation, not the block. On a core without the
  feature the hook result is exactly what 0.1.0 sends. The handoff at session start is
  unchanged.
- **Extension context in archives is not the person's words.** A saved extension block or a
  marker on a user message (`[Extension context from <id> — …]` … `[End extension context]`,
  `[<id> context was shown here for one turn and not kept]`) opens no turn, is never a prompt
  and is never indexed. The message's `_synaps_hooked` count decides when it is there; without
  it (an older core, and core's archive files, which keep only `role` and `content`), the
  trailing blocks that match the fence grammar, never the message's first block.

## [0.1.0] - 2026-11-14

The first release of Axel (0.1.0): one local brain for Synaps, run as a Synaps extension.

- **Recall before each message.** Earlier chats, saved memories and notes that match a
  message are added to it (at most 5 items, 1,600 characters, 150 ms).
- **Capture.** Whole turns are learned as they happen; Synaps's context archives and notes
  are indexed in the background.
- **Tools for the model:** `axel_search`, `axel_remember`, `axel_recall`, `axel_forget`
  (hides, never deletes; can pause one chat).
- **The project forum:** `axel_forum_post`, `axel_forum_read`, `axel_forum_forget`, shared by
  the sessions of one project.
- **One local brain**, a single SQLite file (`~/.local/share/axel/brain.r8`). Nothing leaves
  the machine; only `axel model fetch` uses the network.
- **Reads Synaps's archives and notes and never writes them.**
- **`axel synaps-export`** exports the archives and notes of a Synaps 0.11.1 Axel brain
  (`memory.backend = axel`) for import into Synaps 0.12.0's plain-file stores.
- **Migration from the old Axel brain (axel.r8):** `axel migrate init`, `verify`,
  `catch-up`, and `migrate synaps` for the forum posts of Synaps's own brain. The old brain is
  never changed.
- The command line: `status`, `why`, `show`, `search`, `recall`, `forget` (and `forget
  --now`), `restore`, `pause`, `resume`, `plugin install`. Hints name the binary as it was
  run.

### Requirements

- Linux or macOS.
- Synaps 0.12.0 or newer (the hooks `on_context_archived` and `on_memory_forgotten`, the hook
  and tool `cwd`, and the `tool.call` `caller`).
- To build: Rust with cargo, a C and C++ compiler; the build downloads the ONNX Runtime
  library once.
- The prebuilt ONNX Runtime (linked by the prebuilt binaries and by `cargo install` alike):
  Linux glibc >= 2.39 and libstdc++ >= 13.2 (Ubuntu 24.04+, Debian 13+); an AVX2 CPU on
  x86_64 (x86-64-v3); macOS 13.4 or newer.

### Known limits

- macOS is built but has not been run on a Mac.
- Prebuilt binaries (shell installer) for x86_64 and aarch64 Linux (glibc) and Apple Silicon
  macOS only. Intel Macs, musl Linux and glibc older than 2.39 have no prebuilt ONNX
  Runtime, so `cargo install` does not work there either without an ONNX Runtime built by
  hand (`ORT_LIB_LOCATION`), which is untested.
- On a Synaps without the two hooks, the plugin does not load until they are removed from
  `plugin.json` by hand; archives are then picked up by the background scan, up to 10
  minutes late. Without `caller`, every tool refuses; recall and capture still work.
- The extension sees only `PATH`, `HOME`, `LANG`, `TERM` and `XDG_RUNTIME_DIR`:
  `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `AXEL_BRAIN` do not reach it.
- With `SYNAPS_PROJECT_ROOT` set, chats are keyed by their folder while Synaps files archives,
  notes and forgets under the override, so recall in that folder does not find them.
- The forum tools stay off while any Synaps config or profile says `memory.backend = axel`,
  and when the base folder has more than 4,096 entries.
- `forget --now` reads its wait only from `<base>/config`, not from `<plugins>/axel/config`.
- Note folders outside Synaps are indexed only when listed in `~/.config/axel/sources.toml`;
  there is no built-in list.

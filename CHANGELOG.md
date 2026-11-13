# Changelog

## 0.1.0 (unreleased)

The first release of Axel: one local brain for Synaps, run as a Synaps extension.

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

### Known limits

- macOS is built but has not been run on a Mac.
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

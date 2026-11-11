# Context archive fixtures

Read only by `tests/archive_reader_tests.rs` (commit 6).

## `01a8f6fb3ec12e1d59d07bd6e3943c84c18415e3111dd5cba218b7ce9218b0c7/`

Synaps core's own fixtures, copied unchanged from
`docs/extensions/fixtures/context-archives/` of core at commit `babe8511`
(the page that describes them is `docs/extensions/context-archives.md` of the
same commit). Five files that the store wrote: a legacy slot file, a live
archive with a reduced message, one with a tool input stored as JSON text, a
tombstone, and a live archive with 6 of 7 messages. Do not edit them: the
tests assert their sizes.

## `c6c6…c6c6/` (`c6` 32 times)

Hand-made; core did not write these. The directory name is not the hash of
any project key.

- `s000001/b436b6196e6ca0758c70d5faed628f29.json`: core's live fixture of
  the same id with `scope` rewritten to this directory and the top-level
  field `"carried": 0` added. Core at `babe8511` does not write `carried`
  yet; the field is the one the spec's section 4, "A mark from core: the
  contract", describes.
- `s000001/28000000000000000000000000000001.json`: a successor window laid
  out as `successor_from_parts` builds one and stored as core stores it,
  without the continuation message at index 0 (section 2, test 28):
  `source_message_count` 10, stored messages 1 to 9. Its `logical` and
  `digest` are made up.
- `s000001/28000000000000000000000000000002.json`: the same messages with
  `"carried": 5`.

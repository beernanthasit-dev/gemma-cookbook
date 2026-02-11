## 2026-02-11 - Logging Bottleneck in Go
**Learning:** `log.Printf` inside a hot loop for streaming responses significantly impacts throughput (detected ~16% slowdown). Even "debug" logs left in production code can be costly.
**Action:** Always audit loops for logging statements. For high-frequency loops, use conditional logging or remove it entirely.

## 2026-02-11 - Slice Allocation in Writes
**Learning:** `pw.Write(append(bodyBytes, '\n'))` creates a new slice for every write. Writing `bodyBytes` then `newline` separately avoids this allocation.
**Action:** Use separate `Write` calls instead of `append` when writing to an `io.Writer`.

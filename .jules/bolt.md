## 2026-02-14 - Go SSE Processing Optimization
**Learning:** In Go, `bufio.Reader.ReadBytes` allocates a new slice for every line, and `append` inside a loop for `Write` calls also causes allocations. For high-throughput streams (like SSE), these allocations add up significantly.
**Action:** Use `bufio.Scanner` (which recycles its buffer) and separate `Write` calls to avoid unnecessary allocations in hot loops. Also, avoid logging in hot loops.

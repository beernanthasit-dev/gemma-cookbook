## 2026-02-10 - [Go Stream Optimization]
**Learning:** `append(slice, byte)` inside a loop allocates a new slice on every iteration. For `io.Writer`, writing the parts separately is allocation-free (if pre-allocated). Also, `log.Printf` inside a hot loop is extremely expensive due to I/O locking and formatting overhead.
**Action:** Use separate `Write` calls instead of concatenating slices for `io.Writer`. Remove logging from hot loops or use a conditional debug flag.

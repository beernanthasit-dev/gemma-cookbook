## 2026-02-12 - [Go Stream Processing Optimization]
**Learning:** Removing logging from hot loops in Go (e.g., inside a stream reader) can yield significant performance gains (~11% speedup here). Also, splitting `pw.Write(append(data, '\n'))` into two `pw.Write` calls avoids an allocation per chunk, which adds up in streaming scenarios.
**Action:** Always check for I/O (logging) and allocations (append/conversions) inside `for` loops processing streams.

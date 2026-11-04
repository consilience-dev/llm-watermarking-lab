# Reference implementation

Unmodified `watermark_processor.py` and Apache-2.0 license from
https://github.com/jwkirchenbauer/lm-watermarking/tree/82922516930c02f8aa322765defdb5863d07a00e

The reference test loads this source using a no-op replacement for the unrelated
`normalizers` import. All comparisons use token IDs, disable text normalizers,
and run the reference on CPU. No upstream algorithm code is replaced.

This is a pinned modern minimal implementation of `simple_1`, not a claim that
this repository revision is the exact historical ICML experiment environment.

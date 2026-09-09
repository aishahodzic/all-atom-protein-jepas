# CLJP checkpoint upload plan

Awaiting the new Hugging Face repository name. Previous repository links have been removed from this index. Nothing below claims upload to the new repository.

Folders: `high-lr/`, `middle-lr/`, `low-lr/`, `rotation/`, `per-residue/`. Files use `CLJP_step-XXXXXXXX.ckpt` and `CLJP_last.ckpt`.

| Folder | File | Saved step at inventory | Size (bytes) |
|---|---|---:|---:|
| middle-lr | CLJP_step-00010000.ckpt | 10000 | 2050630816 |
| middle-lr | CLJP_step-00016667.ckpt | 16667 | 2050630944 |
| middle-lr | CLJP_step-00020000.ckpt | 20000 | 2050630944 |
| middle-lr | CLJP_step-00030000.ckpt | 30000 | 2050630944 |
| middle-lr | CLJP_last.ckpt | 30000 | 2050630944 |
| high-lr | CLJP_step-00010000.ckpt | 10000 | 2050630944 |
| high-lr | CLJP_step-00016667.ckpt | 16667 | 2050631136 |
| high-lr | CLJP_step-00020000.ckpt | 20000 | 2050631136 |
| high-lr | CLJP_step-00030000.ckpt | 30000 | 2050631136 |
| high-lr | CLJP_last.ckpt | 30000 | 2050631136 |
| low-lr | CLJP_step-00010000.ckpt | 10000 | 2050630944 |
| low-lr | CLJP_step-00016667.ckpt | 16667 | 2050631136 |
| low-lr | CLJP_step-00020000.ckpt | 20000 | 2050631136 |
| low-lr | CLJP_step-00030000.ckpt | 30000 | 2050631136 |
| low-lr | CLJP_last.ckpt | 30000 | 2050631136 |
| rotation | CLJP_step-00010000.ckpt | 10000 | 2050630880 |
| rotation | CLJP_last.ckpt | 15640 | 2050630880 |
| per-residue | CLJP_step-00010000.ckpt | 10000 | 2050630816 |
| per-residue | CLJP_last.ckpt | 10201 | 2050630816 |

Rotation is still active; its rolling checkpoint will be snapshotted before upload. Smoke tests, failed attempts, recovery backups, and encoder-only diagnostic copies are excluded.

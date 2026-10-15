#!/usr/bin/env python3
"""render_rs3.py for the SMP Racing BR03 (smp_br03_v2024.kn5): same options, same views and output.

    python3 render_br03.py <skin_dir> <out_dir> [--views front34_left,side_left,rear34_right,top] [--jobs 2] ...

<skin_dir> holds body_paint.dds (4096, material «body») and optionally car_paint_rims.dds (rim colour).
--kn5 defaults to the BR03 kn5.  See br03_cfg.py for the material mapping; render_rs3.py is not modified.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import br03_cfg  # noqa: E402
import render_rs3 as rr  # noqa: E402

br03_cfg.patch_render_rs3(rr)
rr.DEFAULT_KN5 = br03_cfg.BR03_KN5

if __name__ == "__main__":
    rr.main()

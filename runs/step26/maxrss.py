"""Run a script in-process and print its wall time and peak resident memory (step 26 timing check).
    python runs/step26/maxrss.py scripts/run_step16_dynpinn.py <args...>"""
import resource
import runpy
import sys
import time

t0 = time.time()
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    print(f"[maxrss] {time.time() - t0:.0f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB",
          flush=True)

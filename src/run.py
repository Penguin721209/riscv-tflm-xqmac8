#!/usr/bin/env python3
"""Run the complete MLP correctness test without an instruction trace."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout',type=int,default=300)
    args=parser.parse_args()
    if args.timeout<=0: parser.error('timeout must be positive')
    app=Path(__file__).resolve().parent
    spike=Path(os.environ.get('SPIKE',str(Path.home()/'riscv/bin/spike'))).expanduser()
    plugin=Path(os.environ.get('XQMAC8_EXTLIB',str(Path.home()/'riscv-isa-sim/build/libxqmac8.so'))).expanduser()
    elf=app/'build/prepacked_check.elf'
    for path in (spike,plugin,elf):
        if not path.is_file(): raise FileNotFoundError(path)
    cmd=[str(spike),'--isa=rv64imc','-p1','-m16','--extlib='+str(plugin),'--extension=xqmac8',str(elf)]
    print('Checking all 197 tokens, 37824 outputs and 189120 accumulators; no trace.',flush=True)
    p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    started=time.monotonic()
    try:
        while True:
            try:
                output,_=p.communicate(timeout=min(10,args.timeout))
                break
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-started
                print('  MLP correctness running: %.0fs / %ds'%(elapsed,args.timeout),flush=True)
                if elapsed>=args.timeout: raise TimeoutError('MLP exceeded wall limit; this is not a pass')
    except BaseException:
        try: os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        output,_=p.communicate()
        (app/'build/correctness.run.log').write_text(output)
        raise
    (app/'build/correctness.run.log').write_text(output)
    print(output,end='',flush=True)
    required=('TFLM_MLP_XQMAC8_EXACT_MATCH_37824',
              'fc_invocations=2 xqmac8_calls=7262208 verified_accumulators=189120 accumulator_mismatches=0')
    if p.returncode!=0 or not all(x in output for x in required):
        raise RuntimeError('MLP failed, exit=%s; inspect build/correctness.run.log'%p.returncode)
    print('MLP_RUN_OK exit_code=0')

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        print('Interrupted; stopped Spike.',file=sys.stderr); sys.exit(130)
    except Exception as e:
        print('ERROR: '+str(e),file=sys.stderr); sys.exit(1)

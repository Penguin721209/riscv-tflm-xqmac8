#!/usr/bin/env python3
"""Stream Spike instruction logs; compare exact outputs before publishing counts."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

TRACE = re.compile(r"core\s+(\d+):\s+(?:[0-3]\s+)?0x([0-9a-fA-F]+)\s+\(0x([0-9a-fA-F]+)\)")
RESULT = re.compile(r"^PERF_RESULT variant=(reference|prepacked) inputs=(\d+) arena_reserved=(\d+) arena_used=(\d+)$")


class RegionCounter:
    def __init__(self, begin, end):
        if begin == end:
            raise ValueError("Measurement marker addresses are identical")
        self.begin, self.end = begin, end
        self.begins = self.ends = self.instructions = self.custom = self.records = 0
        self.active = False

    def feed(self, line):
        match = TRACE.search(line)
        if match:
            hart, pc, bits = int(match[1]), int(match[2], 16), int(match[3], 16)
            if hart != 0:
                raise ValueError("Unexpected extra hart in trace")
            self.records += 1
            if pc == self.begin:
                self.begins += 1
                if self.begins != 1 or self.ends:
                    raise ValueError("Repeated measurement begin")
                self.active = True
            elif pc == self.end:
                self.ends += 1
                if not self.active or self.ends != 1:
                    raise ValueError("Unexpected measurement end")
                self.active = False
            elif self.active:
                self.instructions += 1
                if bits & 0xFE00707F == 0x0000700B:
                    self.custom += 1
            return True
        if self.active and ("exception" in line.lower() or "Executed" in line):
            raise ValueError("Exception or unsupported compressed-repeat trace in measured region: " + line.strip())
        return False

    def validate(self):
        if self.begins != 1 or self.ends != 1 or self.active or self.instructions <= 0:
            raise ValueError("Incomplete measurement region: begin=%d end=%d instructions=%d" %
                             (self.begins, self.ends, self.instructions))


def marker_addresses(nm, elf, begin_name="perf_begin", end_name="perf_end"):
    text = subprocess.check_output([str(nm), "-n", str(elf)], text=True)
    symbols = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] in (begin_name, end_name):
            symbols[fields[2]] = int(fields[0], 16)
    if len(symbols) != 2:
        raise ValueError("Missing perf_begin/perf_end ELF symbols")
    return symbols[begin_name], symbols[end_name]


def section_sizes(size_tool, elf):
    lines = subprocess.check_output([str(size_tool), str(elf)], text=True).splitlines()
    values = lines[-1].split()
    text, data, bss = map(int, values[:3])
    return dict(text_bytes=text, data_bytes=data, bss_bytes=bss,
                section_bytes_sum=text + data + bss)


def run_variant(variant, app, tc, spike, plugin, seconds):
    elf = app / "build" / (variant + ".elf")
    begin, end = marker_addresses(tc / "riscv-none-elf-nm", elf)
    counter = RegionCounter(begin, end)
    prepare_counter = RegionCounter(*marker_addresses(tc / "riscv-none-elf-nm", elf, "prepare_begin", "prepare_end"))
    command = [str(spike), "--isa=rv64imc", "-p1", "-m16", "-l",
               "--extlib=" + str(plugin), "--extension=xqmac8", str(elf)]
    print("Measuring %s: one full 197-token MLP Invoke; streaming trace, wall limit %ss" %
          (variant, seconds), flush=True)
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace",
                            bufsize=1, start_new_session=True)
    timed_out = threading.Event()
    force_timers = []

    def signal_group(sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    def expire():
        timed_out.set()
        signal_group(signal.SIGTERM)
        forced = threading.Timer(2, signal_group, args=(signal.SIGKILL,))
        forced.daemon = True
        force_timers.append(forced)
        forced.start()

    timer = threading.Timer(seconds, expire)
    timer.daemon = True
    timer.start()
    result = None
    outputs = None
    passed = False
    last_progress = time.monotonic()
    nontrace_lines = 0
    try:
        with (app / "build" / (variant + ".run.log")).open("w", encoding="utf-8") as log:
            log.write("COMMAND " + repr(command) + "\n")
            for line in proc.stdout:
                parsed = counter.feed(line)
                prepare_counter.feed(line)
                if not parsed and not line.startswith("core"):
                    clean = line.strip()
                    if nontrace_lines < 200:
                        log.write(line)
                    nontrace_lines += 1
                    match = RESULT.fullmatch(clean)
                    if match:
                        if result is not None:
                            raise ValueError("Duplicate PERF_RESULT")
                        result = dict(variant=match[1], inputs=int(match[2]),
                                      arena_reserved=int(match[3]), arena_used=int(match[4]))
                    if clean.startswith("PERF_OUTPUTS "):
                        if outputs is not None:
                            raise ValueError("Duplicate PERF_OUTPUTS")
                        outputs = bytes.fromhex(clean.split(" ", 1)[1])
                    if clean == "PERF_CHECK_PASSED":
                        passed = True
                if counter.records % 8192 == 0 and time.monotonic() - last_progress >= 2:
                    print("  %s: trace_records=%d measured_instructions=%d" %
                          (variant, counter.records, counter.instructions), flush=True)
                    last_progress = time.monotonic()
            code = proc.wait()
            log.write("EXIT_CODE %d\n" % code)
        if timed_out.is_set():
            raise RuntimeError("%s exceeded wall limit; see build/%s.run.log" % (variant, variant))
        if code != 0:
            raise RuntimeError("%s exited with code %d; see build/%s.run.log" % (variant, code, variant))
        counter.validate()
        prepare_counter.validate()
        if not passed or result is None or result["variant"] != variant or result["inputs"] != 197:
            raise ValueError("Missing or incorrect program verification record for " + variant)
        if outputs is None or len(outputs) != 37824:
            raise ValueError("Expected 37824 output bytes for " + variant)
        expected_custom = 0 if variant == "reference" else 7262208
        if counter.custom != expected_custom:
            raise ValueError("%s: expected %d custom instructions, counted %d" %
                             (variant, expected_custom, counter.custom))
        row = dict(variant=variant, inputs=197, invocations=1, macs=58097664, instructions=counter.instructions,
                   instructions_per_token=counter.instructions / 197,
                   prepare_instructions=prepare_counter.instructions,
                   xqmac8_instructions=counter.custom,
                   arena_used=result["arena_used"], arena_reserved=result["arena_reserved"])
        row.update(section_sizes(tc / "riscv-none-elf-size", elf))
        row["elf_sha256"] = hashlib.sha256(elf.read_bytes()).hexdigest()
        (app / "build" / (variant + ".outputs.hex")).write_text(outputs.hex() + "\n")
        print("  %s finished: %d instructions; Xqmac8=%d" %
              (variant, counter.instructions, counter.custom), flush=True)
        return row, outputs
    finally:
        timer.cancel()
        for forced in force_timers:
            forced.cancel()
        if proc.poll() is None:
            signal_group(signal.SIGKILL)
        proc.wait()
        if proc.stdout:
            proc.stdout.close()


def check_prepacked(app, spike, plugin, seconds):
    elf = app / 'build/prepacked_check.elf'
    command = [str(spike), '--isa=rv64imc', '-p1', '-m16',
               '--extlib=' + str(plugin), '--extension=xqmac8', str(elf)]
    print('Checking both kernels on the complete 197-token MLP...', flush=True)
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=seconds)
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = proc.communicate()
        (app / 'build/correctness.run.log').write_text(output)
        raise
    (app / 'build/correctness.run.log').write_text(output)
    print(output, end='', flush=True)
    markers = ['TFLM_MLP_XQMAC8_EXACT_MATCH_37824',
               'verified_accumulators=189120 accumulator_mismatches=0',
               'fc_invocations=2 xqmac8_calls=7262208']
    if proc.returncode != 0 or not all(marker in output for marker in markers):
        raise RuntimeError('Prepacked correctness check failed, exit=%s; see build/correctness.run.log' % proc.returncode)
    print('MLP_CORRECTNESS_OK', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=3600, help='wall seconds per executable')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    app = Path(__file__).resolve().parent
    tc = Path(os.environ.get('TFLM_TOOLCHAIN_BIN', str(Path.home() / 'toolchains/xpack-15.2.0/bin'))).expanduser()
    spike = Path(os.environ.get('SPIKE', str(Path.home() / 'riscv/bin/spike'))).expanduser()
    plugin = Path(os.environ.get('XQMAC8_EXTLIB', str(Path.home() / 'riscv-isa-sim/build/libxqmac8.so'))).expanduser()
    variants = ('reference', 'prepacked')
    required = [spike, plugin, tc / 'riscv-none-elf-nm', tc / 'riscv-none-elf-size',
                app / 'build/prepacked_check.elf'] + [app / 'build' / (v + '.elf') for v in variants]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError('Required file missing: ' + str(path))
    for name in ('summary.csv', 'summary.json'):
        (app / 'build' / name).unlink(missing_ok=True)
    check_prepacked(app, spike, plugin, args.timeout)
    rows = []
    reference_outputs = None
    for variant in variants:
        row, outputs = run_variant(variant, app, tc, spike, plugin, args.timeout)
        if reference_outputs is None:
            reference_outputs = outputs
        elif outputs != reference_outputs:
            raise ValueError('%s: %d output bytes differ from reference' %
                             (variant, sum(a != b for a, b in zip(outputs, reference_outputs))))
        rows.append(row)
    reference = rows[0]
    comparisons = {}
    for row in rows[1:]:
        comparisons[row['variant']] = dict(
            instruction_ratio_reference_over_variant=reference['instructions'] / row['instructions'],
            instruction_reduction_percent=100 * (1 - row['instructions'] / reference['instructions']),
            allocation_instruction_difference=row['prepare_instructions'] - reference['prepare_instructions'],
            arena_byte_difference=row['arena_used'] - reference['arena_used'])
    report = dict(rows=rows, comparisons=comparisons, exact_output_match=True,
                  prepacked_accumulator_checks_passed=True,
                  interpretation='Dynamic instruction counts, not measured cycle or elapsed-time speedup',
                  allocation_scope='AllocateTensors plus marker-return/call overhead; includes FC Prepare and prepacking',
                  inference_scope='One first Invoke on all 197 tokens plus marker-return/call overhead; input copy, verification and I/O excluded',
                  preprocessing_excluded_from_inference=True,
                  performance_software_counters=False, performance_scalar_accumulator_checks=False)
    with (app / 'build/summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (app / 'build/summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('\nvariant,instructions,instructions_per_token,prepare_instructions,xqmac8_instructions,arena_used')
    for row in rows:
        print('{variant},{instructions},{instructions_per_token:.2f},{prepare_instructions},{xqmac8_instructions},{arena_used}'.format(**row))
    print('exact_output_match=37824/37824 for both full MLP performance executables')
    for variant, comparison in comparisons.items():
        print('%s_instruction_ratio_reference_over_variant=%.6f' %
              (variant, comparison['instruction_ratio_reference_over_variant']))
        print('%s_instruction_reduction_percent=%.3f' %
              (variant, comparison['instruction_reduction_percent']))
    print('MLP_MEASURE_OK (instruction counts, not a time-speedup claim)')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; stopped the measurement subprocess.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(1)

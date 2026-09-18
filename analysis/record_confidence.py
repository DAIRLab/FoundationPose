"""Record the masked object's published confidence for a fixed time window.

Run alongside run_live_demo.py:
    python analysis/record_confidence.py

By default, saves successive runs to analysis/confidence/confidence_1.json,
confidence_2.json, etc. Use --out to specify an explicit output file.

The window starts with the first valid confidence message. This records the
existing mask/depth-fit confidence; it does not run segmentation or pose fitting.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcm_systems.lcm_types.drake import lcmt_drake_signal


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def save_recording(result, output=None):
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
        return output

    directory = Path(__file__).resolve().parent / 'confidence'
    directory.mkdir(parents=True, exist_ok=True)
    versions = [int(match.group(1)) for path in directory.iterdir()
                if (match := re.fullmatch(r'confidence_(\d+)\.json', path.name))]
    version = max(versions, default=0) + 1
    while True:
        output = directory / f'confidence_{version}.json'
        try:
            with output.open('x') as stream:
                stream.write(payload)
            return output
        except FileExistsError:
            version += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=positive_seconds, default=10.0)
    parser.add_argument('--wait-timeout', type=positive_seconds, default=30.0,
                        help='seconds to wait for the first valid message')
    parser.add_argument('--channel', default='OBJECT_STATE_CONFIDENCE')
    parser.add_argument('--url', help='LCM URL; default uses LCM_DEFAULT_URL')
    parser.add_argument('--out', type=Path,
                        help='explicit output file; default saves a new numbered '
                             'JSON in analysis/confidence/')
    args = parser.parse_args()

    import lcm
    lc = lcm.LCM(args.url) if args.url else lcm.LCM()
    samples = []
    started = None
    started_utc = None
    skipped = 0

    def receive(channel, data):
        nonlocal started, started_utc, skipped
        now = time.monotonic()
        if started is not None and now >= started + args.duration:
            return
        try:
            msg = lcmt_drake_signal.decode(data)
            confidence = dict(zip(msg.coord, msg.val))['confidence']
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('invalid confidence')
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            skipped += 1
            print(f'Skipping message: {exc}', file=sys.stderr)
            return
        if started is None:
            started = now
            started_utc = datetime.now(timezone.utc).isoformat()
            print(f'Recording for {args.duration:g} seconds...', flush=True)
        samples.append({'elapsed_seconds': now - started,
                        'timestamp_ms': msg.timestamp,
                        'confidence': confidence})

    subscription = lc.subscribe('^' + re.escape(args.channel) + '$', receive)
    wait_deadline = time.monotonic() + args.wait_timeout
    print(f'Waiting for {args.channel}...', flush=True)
    try:
        while True:
            deadline = wait_deadline if started is None else started + args.duration
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            lc.handle_timeout(max(1, min(100, math.ceil(remaining * 1000))))
    finally:
        lc.unsubscribe(subscription)

    if not samples:
        print('No valid confidence received. Check that the demo is running, '
              'confidence.publish is true, and the LCM URLs match.', file=sys.stderr)
        return 1

    values = [s['confidence'] for s in samples]
    result = {'channel': args.channel, 'started_utc': started_utc,
              'duration_seconds': args.duration, 'sample_count': len(values),
              'average_confidence': statistics.mean(values),
              'min_confidence': min(values), 'max_confidence': max(values),
              'skipped_messages': skipped, 'samples': samples}
    output = save_recording(result, args.out)
    print(f'Average confidence: {result["average_confidence"]:.4f} '
          f'({len(values)} samples over {args.duration:g} seconds)')
    print(f'Saved to {output.resolve()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

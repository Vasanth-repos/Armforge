"""Detect and report ARM64 CPU features."""
import os, platform

def detect():
    feats = {}
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('Features'):
                    parts = line.split(':')[1].strip().split()
                    feats = {
                        'i8mm':    'i8mm'    in parts,
                        'dotprod': 'asimddp' in parts,
                        'sve':     'sve'     in parts,
                        'sve2':    'sve2'    in parts,
                        'neon':    'asimd'   in parts,
                        'fp16':    'asimdhp' in parts,
                    }
                    break
    except Exception as e:
        feats = {'error': str(e)}
    feats['cores'] = os.cpu_count()
    feats['arch']  = platform.machine()
    return feats

def optimal_threads():
    """Leave 1 core free for OS; avoids NUMA contention on Neoverse."""
    return max(1, os.cpu_count() - 1)

def report():
    f = detect()
    print("=== ARM64 Feature Report ===")
    for k, v in f.items():
        if isinstance(v, bool):
            print(f"  {'✓' if v else '✗'} {k}")
        else:
            print(f"  {k}: {v}")
    print(f"  Optimal threads: {optimal_threads()}")

if __name__ == '__main__':
    report()

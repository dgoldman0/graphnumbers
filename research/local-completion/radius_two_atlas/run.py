"""Regenerate the finite radius-two atlas and its scoped verification records."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from analyze import analyze
from verify import main as verify


def main():
    root = Path(__file__).resolve().parent
    compiler = shutil.which('c++')
    if compiler is None:
        raise SystemExit('A C++17 compiler named c++ is required.')
    compiler_version = subprocess.run([compiler, '--version'], check=True,
                                      capture_output=True, text=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory(prefix='radius-two-atlas-') as temporary:
        binary = Path(temporary)/'atlas'
        subprocess.run([compiler, '-O3', '-std=c++17', '-Wall', '-Wextra', '-pedantic',
                        str(root/'atlas.cpp'), '-o', str(binary)], check=True)
        for n, degree, name, balance_sizes in [
                (10, 3, 'degree3_n10', [6,7,8,9,10]),
                (9, 4, 'degree4_n9', [5,6,7])]:
            directory = root/name
            subprocess.run([str(binary), str(n), str(degree), str(directory)], check=True)
            analyze(directory, balance_sizes)
        verify()
    sources = ['atlas.cpp', 'analyze.py', 'verify.py', 'run.py']
    generated = ['verification.json'] + [str(p.relative_to(root))
                                         for name in ('degree3_n10', 'degree4_n9')
                                         for p in sorted((root/name).iterdir())
                                         if p.suffix in ('.tsv', '.json')]
    manifest = {'compiler': compiler_version,
                'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
                'generated_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in generated},
                'status': 'catalogues regenerated; exact analysis and scoped independent oracles passed',
                'scope': 'Independent referee review of the written proofs remains deferred.'}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print('Radius-two atlas complete; manifest.json records the source and data hashes.')


if __name__ == '__main__':
    main()

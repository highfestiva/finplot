#!/usr/bin/env python3

import glob
import os
import sys


pyexe = 'python3.exe' if 'win' in sys.platform else 'python3'
pycode = open('README.md').read().split('```')[1::2]
for pc in pycode:
    if not pc.startswith('python'):
        continue
    pc = pc.split(maxsplit=1)[1]
    open('.t.py', 'w').write('import finplot as fplt\n'+pc)
    print('markup example')
    os.system(f'{pyexe} .t.py')
for fn in glob.glob('finplot/examples/*.py') + glob.glob('dumb/*.py'):
    print(fn)
    os.system(f'{pyexe} {fn}')
os.remove('.t.py')
os.remove('screenshot.png')

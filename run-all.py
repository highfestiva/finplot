#!/usr/bin/env python3

import glob
import os

pycode = open('README.md').read().split('```')[1::2]
for pc in pycode:
    if not pc.startswith('python'):
        continue
    pc = pc.split(maxsplit=1)[1]
    open('.t.py', 'w').write(pc)
    print('markup example')
    os.system('python3.exe .t.py')
for fn in glob.glob('finplot/example*.py'):
    print(fn)
    os.system('python3.exe %s' % fn)

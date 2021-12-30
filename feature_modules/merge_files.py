#!/bin/python

from icecube import icetray, dataio, dataclasses
from I3Tray import *
from glob import glob

files=sorted(glob("/data/user/ssarkar/TridentProduction/datasets/trident/07_L2Files/tmp/ww*"))

print(len(files))
tray=I3Tray()
tray.AddModule("I3Reader",'reader',FilenameList=files)
tray.AddModule("I3Writer", 'writer', Filename="full_trident.i3")
tray.Execute()

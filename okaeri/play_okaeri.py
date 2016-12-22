#! /usr/bin/python

# This script is for windows only.
# Please fix if you want to use it on Linux.

import winsound

OKAERI_FILE = './voice-okaeri-yasashime.wav'

def play():
  winsound.PlaySound(OKAERI_FILE, winsound.SND_FILENAME)
  print("okaeried")


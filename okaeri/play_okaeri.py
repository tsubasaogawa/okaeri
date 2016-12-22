#! /usr/bin/python

import winsound

OKAERI_FILE = './voice-okaeri-yasashime.wav'

def play():
  winsound.PlaySound(OKAERI_FILE, winsound.SND_FILENAME)
  print("okaeried")


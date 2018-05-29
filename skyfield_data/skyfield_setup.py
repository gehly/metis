from skyfield.api import load

ts = load.timescale()
planets = load('de430t.bsp')
print('Ready')
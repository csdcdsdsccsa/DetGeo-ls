#!/usr/bin/env python3
"""Validate the fixed HiSym-CRGPE Gaussian-scale sensitivity matrix."""

SCALES = (15, 20, 25, 30, 35)


def check():
    for scale in SCALES:
        drone = (scale, scale, 2 * scale, 2 * scale)
        svi = (scale, 2 * scale, 2 * scale, 4 * scale)
        for label, values in (('Drone G{}'.format(scale), drone), ('SVI G{}'.format(scale), svi)):
            core_y, core_x, outer_y, outer_x = values
            assert outer_y > core_y and outer_x > core_x, label
        assert drone[1] == drone[0] and drone[3] == drone[2] and drone[2] == 2 * drone[0]
        assert svi[1] == 2 * svi[0] and svi[2] == 2 * svi[0] and svi[3] == 2 * svi[1]
        print('G{}: Drone core={}x{} outer={}x{}; SVI core={}x{} outer={}x{}'.format(scale, *drone, *svi))
    print('Gaussian-scale configuration check PASS')


if __name__ == '__main__':
    check()

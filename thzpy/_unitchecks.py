"""Simple functions for checking and correcting units."""

def _check_thickness(thickness, unit):
    # Checks that a length unit is valid and applies it to a value.
    unit = unit.lower()

    match unit:
        case 'nm':
            thickness *= 1e-9
        case 'um':
            thickness *= 1e-6
        case 'μm':
            thickness *= 1e-6
        case 'mm':
            thickness *= 1e-3
        case 'cm':
            thickness *= 1e-2
        case 'm':
            pass
        case _:
            raise ValueError("Invalid thickness unit.")

    return thickness


def _check_time(time, unit):
    # Checks that the time unit is valid and converts it picoseconds.
    unit = unit.lower()

    match unit:
        case 'fs':
            time *= 1e-3
        case 'ps':
            time = time
        case 'ns':
            time *= 1e3
        case 'us':
            time *= 1e6
        case 'ms':
            time *= 1e9
        case 's':
            time *= 1e12
        case _:
            raise ValueError("Invalid time unit.")

    return time


def _check_frequency(frequency, unit):
    # Checks that the frequenct unit is valid and converts it THz.
    unit = unit.lower()

    match unit:
        case 'thz':
            pass
        case 'ghz':
            frequency *= 1e-3
        case 'mhz':
            frequency *= 1e-6
        case 'khz':
            frequency *= 1e-9
        case 'hz':
            frequency *= 1e-12
        case _:
            raise ValueError("Invalid frequency unit.")

    return frequency

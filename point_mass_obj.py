

class PointMassObj1D:

    def __init__(self, mass, t = 0.01):

        self._mass = mass
        self._time_step = t

    def apply_force(self, force, u = 0):

        a = force/self._mass

        t = self._time_step

        s = u*t + 0.5 * a* (t**2)

        v = u + a*t

        return s, v
import time


class TimeIt:
    """
    Print nested timing information.
    """
    print_output = True
    last_parent = None
    level = -1

    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.outputs = []
        self.parent = None

    def __enter__(self):
        self.t0 = time.time()
        self.parent = TimeIt.last_parent
        TimeIt.last_parent = self
        TimeIt.level += 1

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        st = '%s%s: %0.1fms' % ('  ' * TimeIt.level, self.s, (self.t1 - self.t0) * 1000)
        TimeIt.level -= 1

        if self.parent:
            self.parent.outputs.append(st)
            self.parent.outputs += self.outputs
        else:
            if TimeIt.print_output:
                print(st)
                for o in self.outputs:
                    print(o)
            self.outputs = []

        TimeIt.last_parent = self.parent

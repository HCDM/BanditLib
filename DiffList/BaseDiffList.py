class BaseDiffList:
    def __init__(self, alg_name):
        self.diff_list = {}
        self.diff = {}
        self.diff_list[alg_name] = []
        self.diff[alg_name] = 0
        self.name = "Base"

    def add(self, alg_name):
        self.diff_list[alg_name] = []
        self.diff[alg_name] = 0

    def includes(self, alg_name):
        return alg_name in self.diff_list

    def initial_write(self, f):
        f.write(
            "," + ",".join([str(alg_name) + self.name for alg_name in self.diff_list.iterkeys()])
        )

    def iteration_write(self, f):
        f.write(
            ","
            + ",".join(
                [str(self.diff_list[alg_name][-1]) for alg_name in self.diff_list.iterkeys()]
            )
        )

    def append_to_list(self, userSize):
        for i in self.diff_list:
            self.diff_list[i] += [self.diff[i] / userSize]
            self.diff[i] = 0

    def plot_diff_lists(self, axa, time):
        for alg_name in self.diff_list:
            axa.plot(time, self.diff_list[alg_name], label=alg_name + "_" + self.name)

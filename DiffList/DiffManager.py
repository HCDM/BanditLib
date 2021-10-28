from DiffList.DiffListClasses import *


class DiffManager:
    def __init__(self):
        self.lists_dict = {}

    def add_algorithm(self, alg_name, pref_dict):
        if pref_dict["CanEstimateUserPreference"]:
            if "Theta" in self.lists_dict:
                self.lists_dict["Theta"].add(alg_name)
            else:
                self.lists_dict["Theta"] = ThetaDiffList(alg_name)
        if pref_dict["CanEstimateCoUserPreference"]:
            if "CoTheta" in self.lists_dict:
                self.lists_dict["CoTheta"].add(alg_name)
            else:
                self.lists_dict["CoTheta"] = CoThetaDiffList(alg_name)
        if pref_dict["CanEstimateW"]:
            if "W" in self.lists_dict:
                self.lists_dict["W"].add(alg_name)
            else:
                self.lists_dict["W"] = WDiffList(alg_name)

        if pref_dict["CanEstimateV"]:
            if "V" in self.lists_dict:
                self.lists_dict["V"].add(alg_name)
            else:
                self.lists_dict["V"] = VDiffList(alg_name)

    def initial_write(self, f):
        for value in self.lists_dict.values():
            value.initial_write(f)

    def iteration_write(self, f):
        for value in self.lists_dict.values():
            value.iteration_write(f)

    def update_parameters(self, alg_name, reward_manager, user, alg, pickedArticle, reward, noise):
        for value in self.lists_dict.values():
            if value.includes(alg_name):
                value.update_class_parameters(
                    alg_name, reward_manager, user, alg, pickedArticle, reward, noise
                )

    def append_to_lists(self, userSize):
        for value in self.lists_dict.values():
            value.append_to_list(userSize)

    def plot_diff_lists(self, axa, time):
        for value in self.lists_dict.values():
            value.plot_diff_lists(axa, time)

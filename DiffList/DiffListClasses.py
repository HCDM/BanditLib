from DiffList.BaseDiffList import BaseDiffList


class CoThetaDiffList(BaseDiffList):
    def __init__(self, alg_name):
        BaseDiffList.__init__(self, alg_name)
        self.name = "CoTheta"

    def update_class_parameters(
        self, alg_name, reward_manager, user, alg, pickedArticle, reward, noise
    ):
        diff = reward_manager.getL2Diff(
            user.CoTheta[: reward_manager.context_dimension],
            alg.getCoTheta(user.id)[: reward_manager.context_dimension],
        )
        self.diff[alg_name] += diff


class ThetaDiffList(BaseDiffList):
    def __init__(self, alg_name):
        BaseDiffList.__init__(self, alg_name)
        self.name = "Theta"

    def update_class_parameters(
        self, alg_name, reward_manager, user, alg, pickedArticle, reward, noise
    ):
        self.diff[alg_name] += reward_manager.getL2Diff(
            user.theta[: reward_manager.context_dimension], alg.getTheta(user.id)
        )


class WDiffList(BaseDiffList):
    def __init__(self, alg_name):
        BaseDiffList.__init__(self, alg_name)
        self.name = "W"

    def update_class_parameters(
        self, alg_name, reward_manager, user, alg, pickedArticle, reward, noise
    ):
        self.diff[alg_name] += reward_manager.getL2Diff(
            reward_manager.W.T[user.id], alg.getW(user.id)
        )


class VDiffList(BaseDiffList):
    def __init__(self, alg_name):
        BaseDiffList.__init__(self, alg_name)
        self.name = "V"

    def update_class_parameters(
        self, alg_name, reward_manager, user, alg, pickedArticle, reward, noise
    ):
        self.diff[alg_name] += reward_manager.getL2Diff(
            reward_manager.articles[pickedArticle.id].featureVector, alg.getV(pickedArticle.id)
        )

import numpy as np
from .dLinUCB import LinUCBUserStruct

class ExpertAuditor:

    def __init__(self, featureDimension, alpha, lambda_, delta_1, createTime,  NoiseScale, init="zero"):
        self.featureDimension = featureDimension
        self.d = featureDimension
        self.alpha = alpha
        self.fail = 0.0
        self.success = 0.0
        self.failList = []
        self.plays = 0.0
        self.updates = 0.0
        self.emp_loss = 1.0
        self.createTime = createTime
        self.auditor_lambda  = 0.01*lambda_
        self.NoiseScale = NoiseScale
        self.auditor_sigma = 0.5   #Used in the high probability bound, i.e, with probability at least (1 - sigma) the confidence bound. So sigma should be very small
        #self.ObservationInterval = tau
        self.ACTIVE = True
        self.model_id = createTime
        self.age = 0.0

        self.lambda_ = lambda_
        self.badness_A = lambda_*np.identity(n = self.d)
        self.badness_b =  np.zeros(self.d)
        self.badness_AInv = np.linalg.inv(self.badness_A)
        if (init=="random"):
            self.Beta = np.random.rand(self.d)
        else:
            self.Beta = np.zeros(self.d)
        #self.Beta = 0.2*np.ones(self.d)
        
        #self.Beta = np.random.rand(self.d)
        self.badness_time = 0

        self.badness_X = []
        
        self.badness_predicted_reward = []
        self.badness_actual_reward = []
        self.badness_Y = [a - b for a, b in zip(self.badness_actual_reward, self.badness_predicted_reward)]

        self.badness_b_dic = {}
        self.last_theta = None

        self.armObservations_armfeature = {}
        self.armObservations_Y = {}

        self.update_theta_flag = True
        self.update_beta_flag = True

        self.badness_mean = {}
        self.badness_var = {}

        self.mean = {}
        self.var = {}
        self.model_age = 0.0
        self.update_badness = False
        self.update_num_badenss = 0.0
        self.delta_1 = delta_1

    def badness_updateParameters(self, articlePicked_FeatureVector, click):
        self.badness_A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
        self.badness_b += articlePicked_FeatureVector*click
        self.badness_AInv = np.linalg.inv(self.badness_A)
        self.Beta = np.dot(self.badness_AInv, self.badness_b)
        self.badness_time += 1

    def addObservation(self, article_FeatureVector, actual_reward):
        self.badness_X.append(article_FeatureVector)
        self.badness_actual_reward.append(actual_reward)

    def badness_updateParameters_tau(self, article_FeatureVector, predicted_reward, actual_reward, tau, UserTheta):
        if len(self.badness_X) == 0:
            self.badness_X.append(article_FeatureVector)
            self.badness_actual_reward.append(actual_reward)

        self.update_badness = True
        #self.badness_X.append(article_FeatureVector)
        #Since theta got updated, update training instances in beta
        self.badness_predicted_reward = list(np.dot( np.array(self.badness_X), UserTheta) )
        self.badness_Y = [a - b for a, b in zip(self.badness_actual_reward, self.badness_predicted_reward)]

        # use the most recent tau observations
        X_arr = np.array(self.badness_X[-tau:])
        Y_arr = np.array(self.badness_Y[-tau:])
        self.badness_A = self.lambda_*np.identity(n = self.d) + np.dot(X_arr.T, X_arr)
        self.badness_b = np.dot(X_arr.T, Y_arr)
        self.badness_AInv = np.linalg.inv(self.badness_A)
        self.Beta = np.dot(self.badness_AInv, self.badness_b)
        self.badness_time += 1
        self.update_num_badenss = len(X_arr)

    def badness_get_Theta(self):
        return self.Beta

    def badness_getProbInfo(self, alpha, article_FeatureVector, a_id):
        self.alpha_t = 0.002*self.NoiseScale*np.sqrt(self.d * np.log( (self.auditor_lambda + self.update_num_badenss)/float(self.auditor_sigma * self.auditor_lambda) )) \
                       + 0.002* np.sqrt(self.auditor_lambda)
        # if alpha == -1:
        #     #self.alpha_t = self.NoiseScale *np.sqrt(np.log(np.linalg.det(self.badness_A)/float(self.auditor_sigma * self.lambda_) )) + np.sqrt(self.lambda_)
        #     self.alpha_t = self.NoiseScale*np.sqrt(self.d* np.log( (self.lambda_ + self.update_num_badenss)/float(self.auditor_sigma * self.auditor_lambda) )) + np.sqrt(self.auditor_lambda)
        # else:
        #     self.alpha_t = alpha
        mean = np.dot(self.Beta,  article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.badness_AInv),  article_FeatureVector))
        #pta = mean + alpha * var
        return {'mean':mean, 'var':var, 'alpha':alpha, 'alpha_t':self.alpha_t}

    def init_Beta(self):
        self.badness_A = self.lambda_*np.identity(n = self.d)
        self.badness_b =  np.zeros(self.d)
        self.badness_AInv = np.linalg.inv(self.badness_A)
        self.Beta = np.zeros(self.d)
        self.regret = 0.0

    def reset_badenss(self):
        #self.badness_A = self.lambda_*np.identity(n = self.d)
        #self.badness_b =  np.zeros(self.d)
        #self.badness_AInv = np.linalg.inv(self.badness_A)
        #self.Beta = np.zeros(self.d)
        self.badness_X = []
        self.badness_actual_reward = []


class BanditExpert(LinUCBUserStruct):
    """Original class name 'ContextDependent_SlaveStruct'
    """
    def __init__(self, current_true_theta, featureDimension, alpha, lambda_, delta_1, createTime,  NoiseScale, init="zero"):
        LinUCBUserStruct.__init__(self, featureDimension = featureDimension, alpha=alpha, lambda_=lambda_,
            NoiseScale = NoiseScale, init = init)
        self.Auditor = ExpertAuditor(featureDimension = featureDimension, alpha = alpha, lambda_=lambda_, delta_1=delta_1,
            createTime =createTime, NoiseScale = NoiseScale, init="zero")
        self.true_theta = current_true_theta
        self.alpha = alpha
        self.fail = 0.0
        self.success = 0.0
        self.failList = []
        self.plays = 0.0
        self.updates = 0.0
        self.emp_loss = 1.0
        self.createTime = createTime
        self.lambda_  = lambda_
        self.NoiseScale = NoiseScale
        self.expert_sigma = 1.e-10   #Used in the high probability bound, i.e, with probability at least (1 - sigma) the confidence bound. So sigma should be very small
        #self.ObservationInterval = tau
        self.ACTIVE = True
        self.model_id = createTime
        self.update_num = 0.0

        self.lambda_ = lambda_
        self.delta_1 = delta_1
    
        self.last_theta = self.UserTheta

        self.armObservations_armfeature = {}
        self.armObservations_Y = {}

        self.update_theta_flag = True
        self.update_beta_flag = True

        self.mean = {}
        self.var = {}
        self.model_age = 0.0
        self.age = 0.0

        self.DetA = np.linalg.det(self.A)

    
    def updateParameters(self, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector*click
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.time += 1
        #self.DetA = np.linalg.det(self.A)
        self.update_num +=1.0

    def getProb(self, alpha, article_FeatureVector, a_id):
        self.alpha_t = self.NoiseScale ** 2 * np.sqrt(
            self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
            self.lambda_)
        mean = np.dot(self.UserTheta,  article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
        # pta = mean + alpha * var
        pta = mean + self.alpha_t * var
        return pta

    def getProbInfo(self, alpha, article_FeatureVector, a_id):
        self.alpha_t = self.NoiseScale ** 2 * np.sqrt(
                self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
                self.lambda_)
        mean = np.dot(self.UserTheta,  article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
        return {'mean':mean, 'var':var, 'alpha':alpha, 'alpha_t': self.alpha_t}
   

class DenBand_UserModel:

    def __init__(self, createTime, featureDimension, alpha, lambda_, delta_1, NoiseScale):
        self.EXPERTs = []
        self.createTime = createTime
        self.featureDimension = featureDimension
        self.alpha = alpha
        self.lambda_ = lambda_
        self.delta_1 = delta_1
        self.NoiseScale = NoiseScale
    
        self.activeLinUCBs = []
        self.inactiveLinUCBs = []

        self.ModelSelection = []
        self.newUCBs = []
        self.discardUCBs = []
        self.discardUCBIDs = []
        self.ModelSelection = []
        self.ModelUCB = []
        self.SwitchPoints = []
        self.SwitchPoints_UCB = []
        self.SwitchPoints_pool = []
        self.ActiveLinUCBNum = []
        self.selectedAlgList = []
        self.selectedAlg = None
        self.createTime = createTime
        self.time = createTime

    def AddExpert(self, current_true_theta, global_time):
        self.EXPERTs.append(BanditExpert(current_true_theta = current_true_theta, featureDimension = self.featureDimension,
            alpha = self.alpha, lambda_ = self.lambda_, delta_1=self.delta_1, createTime= self.time, NoiseScale = self.NoiseScale))
        self.newUCBs.append(self.time)


class DenBand:

    def __init__(self, dimension, alpha, lambda_, NoiseScale, tau, delta_1, delta_2,
        tilde_delta_1, delta_L, age_threshold=None,init="zero"):  # n is number of users
        self.users = {}
        self.dimension = dimension
        self.alpha = alpha
        self.lambda_ = lambda_
        self.global_time = 0
        
        self.NoiseScale = NoiseScale
        self.ObservationInterval = tau
        self.tau  = tau
       
        self.CanEstimateUserPreference = True
        self.CanEstimateCoUserPreference = False
        self.CanEstimateW = False
        self.CanEstimateV = False

        self.CanEstimateBeta = False
        self.reuseable_model_set = {}
        self.reuseable_model_set_old = {}
        self.reuseable_model_set_new = {}

        self.changed = False
        self.tau = tau
        self.age_threshold = age_threshold or 3000
        self.new_percent = 0.0
        self.total = 0.0

        self.contamination = 0.0
        self.total_admissible = 0.0
        self.total_multipMold = 0.0
        self.change_invariant_arms = 0.0
        self.delta_L = delta_L
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.tilde_delta_1 = tilde_delta_1
        self._debug_counter = 0

    def decide(self, pool_articles, userID, current_true_theta=None, changed=False):
        """
        pool_articles: a list of arms.
        userID: user id. 
        current_true_theta: for debug purpose. Default value is None.
        changed: for debug purpose. Default value is False.
        """
        self._debug_counter +=1
        if userID not in self.users:
            self.users[userID] = DenBand_UserModel(createTime = 0, featureDimension = self.dimension,
                alpha = self.alpha, lambda_ = self.lambda_, delta_1=self.delta_1, NoiseScale = self.NoiseScale)
            self.users[userID].AddExpert(current_true_theta = current_true_theta, global_time = self.global_time )
        
        if changed:
            self.changed = True
        maxPTA = float('-inf')
        articlePicked = None
        modelPicked = None

        EXPERTs = self.users[userID].EXPERTs
        pool_id = []
        totalFeature = np.zeros(self.dimension)
        to_remove = []
        total_reward_estimation_error_new = 0.0
        total_reward_estimation_error_LCB = 0.0
        for a in pool_articles:
            CHANGE_INVARIANT_FLAG = False
            a_feature =a.featureVector
            a_id = a.id
            self.reuseable_model_set[a_id] = []
            CREATENEW_FLAG = True
            #Set bad models as inactive
            EXPERTs = self.users[userID].EXPERTs
            for alg in EXPERTs:
                badness_data = alg.Auditor.badness_getProbInfo(self.alpha, a_feature, a_id)
                data = alg.getProbInfo(self.alpha, a_feature, a_id)
                emp_loss = abs(badness_data['mean'])
                CB = badness_data['var']*badness_data['alpha_t']
                CB_theta = data['var']*data['alpha_t']                
                if emp_loss < (CB + CB_theta) + self.delta_L:
                    self.total_admissible +=1
                    alg.ACTIVE = True
                    CREATENEW_FLAG = False
                    self.reuseable_model_set[a_id].append(alg)
                    # if self._debug_counter % 200 ==0:
                    #     print('admissible', emp_loss, (CB + CB_theta) + self.delta_L,
                    #           CB , CB_theta, self.delta_L)
                else:
                    to_remove.append(alg)
                    # if self._debug_counter % 200 ==0:
                    #     print('Not admissible', emp_loss, (CB + CB_theta) + self.delta_L, CB, CB_theta,
                    #           self.delta_L, 'Nnnnnnn')
    
            if (len(self.reuseable_model_set[a_id]) == 0):
                self.users[userID].AddExpert(current_true_theta, self.global_time) 
                self.reuseable_model_set[a_id].append(self.users[userID].EXPERTs[-1])
                self.total_admissible +=1

            EXPERTs = self.users[userID].EXPERTs

            max_id = float('-inf')
            max_id_alg = None
            for alg in EXPERTs:
                if alg in self.reuseable_model_set[a_id]:
                    model_id = int(alg.model_id)
                    if model_id > max_id:
                        max_id = model_id
                        max_id_alg = alg

            min_loss = float('+inf')
            min_slave = None
            min_alg = None
            for alg in EXPERTs:
                if alg in self.reuseable_model_set[a_id]:
                    pool_id.append(alg.createTime)
                    badness_data = alg.Auditor.badness_getProbInfo(self.alpha, a_feature, a_id)
                    emp_loss = abs(badness_data['mean']) - badness_data['var']*badness_data['alpha_t']
                    if emp_loss < min_loss:
                        min_loss = emp_loss
                        min_alg = alg
            if current_true_theta is not None:
                total_reward_estimation_error_new += abs(np.dot(current_true_theta  - max_id_alg.UserTheta , a_feature ))
                total_reward_estimation_error_LCB += abs(np.dot(current_true_theta - min_alg.UserTheta , a_feature))

            if len(self.reuseable_model_set[a_id]) > 1:
                self.total +=1.0
                self.total_multipMold +=1.0

            index = EXPERTs.index(min_alg)
            self.users[userID].EXPERTs[index].plays += 1
            self.users[userID].selectedAlg = min_alg

            #UpdateStatistics
            self.users[userID].EXPERTs[index].plays +=1
            self.users[userID].ModelSelection.append(self.users[userID].EXPERTs[index].createTime)
            self.users[userID].ModelUCB.append(min_loss)
            if len(self.users[userID].ModelSelection) >1:
                if self.users[userID].ModelSelection[-1] != self.users[userID].ModelSelection[-2]:
                    self.users[userID].SwitchPoints.append(len(self.users[userID].ModelSelection))
                    UCB_Diff = abs(self.users[userID].ModelUCB[-1] - self.users[userID].ModelUCB[-2])
                    self.users[userID].SwitchPoints_UCB.append(1)
                    self.users[userID].SwitchPoints_pool.append(pool_id)
                    self.users[userID].selectedAlgList.append(self.users[userID].ModelSelection[-1])
            self.users[userID].ActiveLinUCBNum.append(len(self.users[userID].EXPERTs))
            x_pta = min_alg.getProb(self.alpha, a_feature, a_id)
            if maxPTA < x_pta:
                articlePicked = a
                maxPTA = x_pta
                modelPicked = min_alg
        return articlePicked

    def updateParameters(self, articlePicked, click, userID):
        self.global_time += 1
        self.users[userID].time +=1
        # Create new LinUCB model as long as the reward is beyond the reward confidence bound.
        # we only have out of bounds if EVERY one is out of bounds
        # i.e. NONE of them are in bounds
        # so if we find one that is in Bounds, we should set it to FALSE
        out_of_conf_bounds = True
        good_alg = []
        arm_id = articlePicked.id
        a_id = articlePicked.id
        model_id_list = []
        obs_num_list = []

        for alg in self.users[userID].EXPERTs:
            #if alg.Auditor.update_badness:
            alg.Auditor.addObservation(articlePicked.featureVector, click)
            alg.age +=1
            ## the following strategy is optional
            #if alg.age > 6000:
            #    self.users[userID].EXPERTs.remove(alg)
            alg.model_age +=1
            if alg in self.reuseable_model_set[a_id]:
                if alg.age < self.age_threshold:
                    model_id_list.append(alg.model_id)
                    observationNum = 0
                    for i in alg.armObservations_Y:
                        observationNum +=len(alg.armObservations_Y[i])
                    obs_num_list.append(observationNum)
                    #Get LinUCB' reward estimation quality
                    data = alg.getProbInfo(alg.alpha, articlePicked.featureVector, a_id)
                    alg.TotalrewardDiff = abs(data['mean'] - click)
                    TotalrewardDiff = abs(data['mean'] - click)
                    if TotalrewardDiff <= data['var']*data['alpha_t'] + self.delta_L:
                        out_of_conf_bounds = False  #Good LinUCB
                        alg.success += 1
                        good_alg.append(alg)
                        badness_value = 0.0
                        alg.updateParameters(articlePicked.featureVector, click)
                        alg.updates +=1

                        if arm_id not in alg.armObservations_armfeature:
                            alg.armObservations_armfeature[arm_id] = articlePicked.featureVector
                            alg.armObservations_Y[arm_id] = []
                        alg.armObservations_Y[arm_id].append(click)

                        self.update_theta_flag = True
                        self.update_beta_flag = False

            data = alg.getProbInfo(alg.alpha, articlePicked.featureVector, a_id)
            alg.TotalrewardDiff = abs(data['mean'] - click)

            if self.users[userID] not in alg.Auditor.badness_b_dic:
                alg.Auditor.badness_b_dic[self.users[userID].time] = 0

            badness_value = float(data['mean'] - click)
            predicted_reward = data['mean']
            actual_reward = click
            alg.Auditor.badness_b_dic[self.users[userID].time] = badness_value
            alg.Auditor.badness_updateParameters_tau(articlePicked.featureVector, predicted_reward, actual_reward, self.tau, alg.UserTheta)
            alg.Auditor.update_theta_flag = False
            alg.Auditor.update_beta_flag = True
            if self.users[userID].time % 300 ==0:
                alg.last_theta = alg.UserTheta

    def getTheta(self, userID):
        max_model_id = float('-inf')
        model = None
        for alg in self.users[userID].EXPERTs:
            if float(alg.model_id) > max_model_id:
                max_model_id = alg.model_id
                model = alg
        return model.UserTheta

    def getBeta(self, userID):
        for alg in self.users[userID].EXPERTs:
            if alg.model_id == self.users[userID].selectedAlg.model_id:
                return alg.Beta

    def CanEstimateUserCluster(self):
        return False
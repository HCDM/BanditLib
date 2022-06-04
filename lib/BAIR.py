import numpy as np
import argparse
from Users.LearningUsers import User


class Bair:
    def __init__(self, delta=0.1, alpha=1.0):
        self.delta = delta
        self.alpha = alpha
        self.user_model = None

    def load_user_model(self, user):
        self.user_model = user
        self.K = len(self.user_model.mu)

    def run(self, N0=None, verbose=False):
        if N0 is None:
            self.N0 = (2*(K-1)/delta)**(1/alpha)/self.user_model.rho0
        else:
            self.N0 = N0
        if verbose:
            print('Phase-1 steps:', N0)
        #### phase 1 ####
        # initialization
        R = 1
        N = 0  # total number of acceptances
        F = [i for i in range(K)]
        while N < N0:
            if verbose:
                print('iteration: ', N)
            if not F:
                F = [i for i in range(K)]
                R += 1
            for i in F:
                if verbose:
                    print('Arm tried:', i)
                    print('user_model.global_time:', user_model.global_time)
                    print('user_model.acceptance:', user_model.num_acceptance_per_arm)
                    print('user_model.rejection:', user_model.num_rejection_per_arm)
                    print('user_model.ucb:', user_model.ucb)
                    print('user_model.mean:', user_model.empirical_mean_per_arm)
                    print('user_model.lcb:', user_model.lcb)
                if R <= 1:
                    accepted = user_model.isUserAccept(i)
                    if not accepted:
                        F.remove(i)
                    else:
                        N += 1
                else:
                    while True:
                        accepted = user_model.isUserAccept(i)
                        if not accepted:
                            F.remove(i)
                            break
                        else:
                            N += 1
                            if N >= N0:
                                break
        if verbose:
            print('Phase-2 starts:')

        F = dict(zip(range(K), [1] * K))
        while len(F) > 1:
            if verbose:
                print(F)
            for i in range(K):
                if i in F.keys():
                    accepted = user_model.isUserAccept(i)
                    if not accepted:
                        F[i] -= 1
                        if F[i] < 0:
                            F.pop(i)
                            if len(F) == 1:
                                break

        last_arm = list(F.keys())[0]

        best_arm = np.argmax(user_model.mu)
        print("best arm is {}, algorithm finds {} in {} rounds".format(best_arm, last_arm, user_model.global_time))
        return last_arm == best_arm, user_model.global_time, user_model.num_acceptance_total


if __name__ == '__main__':

    # environment settings
    parser = argparse.ArgumentParser(description="BAIR test")
    parser.add_argument('--rho', default=1)
    parser.add_argument('--alpha', default=1)
    parser.add_argument('--dp', default=1)
    parser.add_argument('--m', default=1)
    parser.add_argument('--min_gap', default=0.5)
    parser.add_argument('--num_trials', default=1000)
    parser.add_argument('--T', default=1000)
    parser.add_argument('--alg', default='bair')
    parser.add_argument('--delta', default=0.1)
    parser.add_argument('--K', default=2)
    parser.add_argument('--N0', default=1)
    parser.add_argument('--max_ite', default=1000)

    args = vars(parser.parse_args())

    rho = float(args['rho'])
    alpha = float(args['alpha'])
    detpro = float(args['dp'])
    m = float(args['m'])
    min_gap = float(args['min_gap'])
    num_trials = int(args['num_trials'])
    TT = int(args['T'])
    alg = args['alg']
    delta = float(args['delta'])
    K = int(args['K'])
    N0 = float(args['N0'])
    max_ite = int(args['max_ite'])

    res_ls = []
    T = []
    N = []
    print("alg, min_gap, delta, K", alg, min_gap, delta, K)

    bair = Bair(delta=delta, alpha=alpha)

    for run in range(num_trials):
        user_model = User(K=K, rho=rho, alpha=alpha, determ_pro=detpro)
        user_model.initialize(min_gap=min_gap)
        bair.load_user_model(user=user_model)
        res, t, n = bair.run(N0=N0)

        res_ls.append(res)
        if t >= 0 and n >= 0:
            T.append(t)
            N.append(t - n)
        # print average stopping time and acceptance ratio
        print(np.mean(T), np.mean(res_ls))


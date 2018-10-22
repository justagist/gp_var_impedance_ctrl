import GPy
import numpy as np
import matplotlib.pyplot as plt

class GPModel:

    def __init__(self, gpy_kernal = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)):

        self._kernel = gpy_kernal

        
    def fit(self, x0, y0, optimise = True):

        self._x = x0
        self._y = y0

        self._curr_model = GPy.models.GPRegression(self._x, self._y, self._kernel)

        if optimise:
            self.optimise_model()

    def refit(self, x, y, optimise = True):

        new_x = np.vstack([self._x, x])
        new_y = np.vstack([self._y, x])
        self.fit(new_x, new_y, optimise = optimise)


    def visualise_model(self, dense_img = True):
        fig = self._curr_model.plot(plot_density=dense_img)
        GPy.plotting.show(fig)
        # plt.axis('equal')
        # plt.show()

    def optimise_model(self, num_restarts = 1):
        """ 
        maximise likelihood of data by changing kernel parameters

        """
        if num_restarts == 1:
            self._curr_model.optimize(messages=True)

        elif num_restarts > 1:
            self._curr_model.optimize_restarts(num_restarts = num_restarts)



    def predict(self, x, plot = False):

        new_mean, new_cov = self._curr_model.predict(x)
        # print new_mean, new_cov

        # sd = np.sqrt(new_cov)
        # print sd

        if plot:
            self.visualise_model()
            plt.plot(x, new_mean, 'r+')
            # plt.show()

        return new_mean[0,0], new_cov[0,0]



if __name__ == '__main__':
    # GPy.plotting.change_plotting_library('matplotlib')
    X = np.random.uniform(-3.,3.,(20,1))
    Y = np.sin(X) + np.random.randn(20,1)*0.05
    # print X
    # print Y
    ff = GPModel(X,Y)
    ff.fit(optimise = True)
    # ff.visualise_model()

    x = np.array([[1.3],[-2.5]])
    ff.predict(x)

    # spring_const = 2.5

    # traj = np.linspace(0,100,100).reshape(100,1)

    # forces = traj*spring_const

    # # for i in range(num_demos):

    # ff = GPModel()    
    # ff.fit(traj, forces,optimise = True)
    # ff.visualise_model()
    # plt.show()



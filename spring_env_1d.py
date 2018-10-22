
import numpy as np

class LinearSpring1D:

    def __init__(self, k):

        self._k = k


    def compute_resistance(self, extension):

        return extension * self._k


    def compute_resistance_traj(self, extension_traj):

        return self.compute_resistance(extension_traj)


# ====================================================================
# ====================================================================

def sinusoid(X, noise = 0.005):
    '''
        default function for non-linear spring stiffness: 
    '''
    # print (np.sin(X) + np.random.randn(X.shape[0],1)*noise).shape
    if len(X.shape) == 1:
        X = X.reshape(-1,1)

    retval = np.sin(X) + np.random.randn(X.shape[0],1)*noise
    return retval


# ====================================================================
# ===================  NON-LINEAR 1D SRING ===========================
# ====================================================================

class NonLinearSpring1D:

    def __init__(self,k_func = sinusoid):

        self._k_func = k_func

    def compute_resistance(self, extension):

        return (extension * self._k_func(np.asarray(extension).reshape([1,1])))[0]

    def compute_resistance_traj(self, extension_traj):

        return np.multiply(extension_traj,self._k_func(extension_traj))



# ====================================================================

if __name__ == '__main__':
    
    spring = NonLinearSpring1D()

    # print spring.compute_resistance(25)
    x = []
    y = []
    for i in range(1,100):
        x_val = i*0.1
        x.append(x_val)
        # y.append(spring.compute_resistance(x_val))
    y = spring.compute_resistance_traj(np.asarray(x).reshape([-1,1]))

    import matplotlib.pyplot as plt

    plt.plot(np.asarray(x),np.asarray(y))
    plt.show()







import numpy as np
from ff_model import GPModel
from point_mass_obj import PointMassObj1D
from spring_env_1d import LinearSpring1D, NonLinearSpring1D



kp_min = 0.01
kp_max = 2500

C = 10 # from paper
min_var = .001
max_var = 1
def compute_stiffness(cov):

    '''
        method used in paper

    '''

    std = np.sqrt(cov)
    exp_val = -C*(std - min_var)/(max_var - min_var)
    z = np.exp(exp_val)

    kp = kp_min + (1 - z)*(kp_max - kp_min)
    return kp



if __name__ == '__main__':
    
    stiffness = 2.5
    # spring = LinearSpring1D(stiffness)
    
    spring = NonLinearSpring1D()
    obj = PointMassObj1D(mass = 5)



    ff_model = GPModel()

    # extension_traj = np.random.rand(25,1)*50
    orig_extension_traj = np.linspace(0,50,25).reshape(25,1)

    forces = spring.compute_resistance_traj(orig_extension_traj)
    ff_model.fit(orig_extension_traj,forces)
    # ff_model.visualise_model(dense_img = False)

    for i in range(10):
        extension_traj = np.random.rand(25,1)*10
        new_forces = spring.compute_resistance_traj(extension_traj)
        ff_model.refit(extension_traj,forces)

    # ff_model.optimise_model(50)
    # ff_model.visualise_model(dense_img = False)

    import matplotlib.pyplot as plt
    # plt.show()
    # 
    obj_u = 0
    traj = []
    ext = 0
    for k in range(100):

        # ext = 0.5 * k
        ff_pred, cov = ff_model.predict(np.asarray([[ext]]))
        # print ff_pred, cov
        k = compute_stiffness(cov)
        # print k
        computed_force = k*(99*0.5 - ext) + ff_pred

        resistance_force_actual = spring.compute_resistance(ext)
        # print resistance_force_actual

        ext, obj_u = obj.apply_force(computed_force - resistance_force_actual, obj_u)

        traj.append(ext)
        print ext

    plt.plot(np.linspace(0,50,100),traj,'r-')
    plt.plot(np.linspace(0,50,100),np.ones([100,1])*99*0.5)
    plt.show()



    # plt.plot(extension_traj, forces)
    # plt.show()


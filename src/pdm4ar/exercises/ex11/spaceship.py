import sympy as spy

from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters


class SpaceshipDyn:
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SpaceshipGeometry, sp: SpaceshipParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi delta m", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust ddelta", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        0x 1y 2psi 3vx 4vy 5dpsi 6delta 7m
        """

        # state variables
        x, y, psi, vx, vy, dpsi, delta, m = [spy.sympify(var) for var in self.x]
        F_thrust, ddelta = [spy.sympify(var) for var in self.u]
        t_f = spy.sympify(self.p[0])

        # convert numerical parameters
        C_T = spy.sympify(self.sp.C_T)  # hrust coefficient for fuel consumption
        l_T = spy.sympify(self.sg.l_t_half)  # istance from CoG to thruster
        I = spy.sympify(self.sg.Iz)  # moment of inertia

        # Dynamics
        f = spy.Matrix(
            [
                vx * spy.cos(psi) - vy * spy.sin(psi),  # dx/dt
                vx * spy.sin(psi) + vy * spy.cos(psi),  # dy/dt
                dpsi,  # dpsi/dt
                (1 / m) * spy.cos(delta) * F_thrust + dpsi * vy,  # dvx/dt
                (1 / m) * spy.sin(delta) * F_thrust - dpsi * vx,  # dvy/dt
                -l_T / I * spy.sin(delta) * F_thrust,  # ddpsi/dt
                ddelta,  # d(delta)/dt
                -C_T * F_thrust,  # dm/dt (fuel consumption)
            ]
        )

        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func

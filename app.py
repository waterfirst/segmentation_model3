import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from scipy.integrate import odeint

st.title("Intergroup Network Segmentation Model")

# 파라미터 설정
c0 = st.sidebar.slider("c0 (Basic education cost)", 1.0, 10.0, 7.5, 0.1)
ahat = st.sidebar.slider("ahat (Mean ability)", -2.0, 2.0, 0.0, 0.1)
sigma = st.sidebar.slider("σ (Ability standard deviation)", 0.1, 2.0, 1.6, 0.1)
psi = st.sidebar.slider("ψ (Scale parameter)", 0.1, 5.0, 1.0, 0.1)
p = st.sidebar.slider("p (Education period network effect)", 0.1, 5.0, 2.5, 0.1)
alpha = st.sidebar.slider("α (Death rate)", 0.01, 0.5, 0.1, 0.01)
rho = st.sidebar.slider("ρ (Time discount rate)", 0.01, 0.5, 0.1, 0.01)
f0 = st.sidebar.slider("f0 (Base externality)", 0.0, 2.0, 0.0, 0.1)
delta_bar = st.sidebar.slider("δ_bar (Base skill premium)", 0.1, 2.0, 0.6, 0.1)
q = st.sidebar.slider("q (Network externality strength)", 0.1, 10.0, 3.2, 0.1)


def Π_t_Locus(st):
    return (q * st + (delta_bar + f0)) / (alpha + rho)


def s_t_Locus(st):
    return c0 - p * st - ahat * psi + np.sqrt(2) * sigma * psi * erfinv(1 - 2 * st)


def system(X, t, alpha, rho):
    s, Pi = X
    dsdt = alpha * (1 - s - s_t_Locus(s))
    dPidt = (alpha + rho) * (Π_t_Locus(s) - Pi)
    return [dsdt, dPidt]


def plot_graph():
    s = np.linspace(0.001, 0.999, 1000)  # Avoid 0 and 1 for numerical stability
    Π_zero = Π_t_Locus(s)
    s_zero = np.array([s_t_Locus(si) for si in s])

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(s, Π_zero, "orange", label="Π_t = 0 locus")
    ax.plot(s, s_zero, "blue", label="s_t = 0 locus")

    # Plot streamlines
    X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 10, 20))
    U = alpha * (1 - X - s_t_Locus(X))
    V = (alpha + rho) * (Π_t_Locus(X) - Y)
    ax.streamplot(X, Y, U, V, density=0.5, color="gray", linewidth=0.5, arrowsize=0.5)

    # Find intersections (steady states)
    intersections = []
    for i in range(len(s) - 1):
        if (Π_zero[i] - s_zero[i]) * (Π_zero[i + 1] - s_zero[i + 1]) < 0:
            intersections.append((s[i], Π_zero[i]))

    # Plot steady states
    labels = ["l", "m", "h"]
    for i, (x, y) in enumerate(intersections):
        ax.plot(x, y, "ro")
        ax.annotate(f"E_{labels[i]}", (x, y), xytext=(5, 5), textcoords="offset points")

    # Plot optimistic and pessimistic paths if there are 3 steady states
    if len(intersections) == 3:
        t = np.linspace(0, 100, 1000)
        sol_opt = odeint(
            system,
            [intersections[1][0] + 0.01, intersections[1][1]],
            t,
            args=(alpha, rho),
        )
        sol_pes = odeint(
            system,
            [intersections[1][0] - 0.01, intersections[1][1]],
            t,
            args=(alpha, rho),
        )
        ax.plot(sol_opt[:, 0], sol_opt[:, 1], "g-", label="Optimistic Path")
        ax.plot(sol_pes[:, 0], sol_pes[:, 1], "r-", label="Pessimistic Path")

        # Range of Indeterminacy
        ax.axvline(x=intersections[0][0], color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=intersections[2][0], color="gray", linestyle="--", alpha=0.5)
        ax.annotate(
            "Range of Indeterminacy",
            xy=(0.4, 0.5),
            xytext=(0.4, 0.3),
            arrowprops=dict(arrowstyle="<->"),
            ha="center",
        )

    ax.set_xlabel("s_t (Skilled worker ratio)")
    ax.set_ylabel("Π_t (Expected return on human capital investment)")
    ax.set_title("Intergroup network segmentation model")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 1)

    return fig


st.pyplot(plot_graph())

st.markdown(
    """
## Model Explanation

This model simulates the intergroup network segmentation based on Professor Young-Chul Kim's work.

- **Orange line (Π_t = 0 locus)**: States where the expected return on human capital investment is constant.
- **Blue line (s_t = 0 locus)**: States where the ratio of skilled workers is constant.
- **Red dots**: Steady states of the system.
- **Gray streamlines**: Show the direction of system dynamics.
- **Green line**: Optimistic path (if applicable).
- **Red line**: Pessimistic path (if applicable).

Adjust the parameters to explore various scenarios. Observe how changes in parameters affect the number and position of steady states, as well as the system's dynamic characteristics.
"""
)

import numpy as np
import networkx as nx
import hyperbolic_embedding

np.random.seed(42)
hyperbolic_embedding.set_seed(42)

edgelist = np.loadtxt("edgelist.txt", dtype=int)
coordinate = np.loadtxt("coordinate.txt")

gs = hyperbolic_embedding.GradState()
gs.set_edgelist(edgelist)
gs.infer_beta()
print("beta:", gs.beta)
gs.infer_mu()
print("mu:", gs.mu)
gs.init_embedding("laplacian")
print("Laplacian embedding complete")
gs.set_theta(0.2)

n_epochs = 8_000
lr = 0.001

def train_session(gs, n_epochs, lr=0.01, barnes_hut=True, verbose=False):
    for epoch in range(n_epochs):
        interaction_count = (
            gs.compute_repulsive_force_bh() if barnes_hut else gs.compute_repulsive_force_exact()
        )
        interaction_percentage = 100*interaction_count/gs.N/gs.N
        if not barnes_hut:
            interaction_count = gs.N * gs.N
        gs.add_attractive_force()
        gs.add_l1_penalty_z(epoch / 2_000)
        gs.step_adam(lr)
        if verbose and epoch % 100 == 0:
            forces = gs.get_forces()
            pos = gs.get_positions()
            avg_force = np.mean(np.abs(forces))
            print(
                f"Epoch {epoch}: avg_z={np.mean(np.abs(pos[:, 2])):.2f}, "
                f"lnp={gs.get_lnp_exact()}, lnp_bh={gs.get_lnp_bh()}"
            )

def train_session_decay(gs, n_epochs, lr=0.01, barnes_hut=True, verbose=False):
    for epoch in range(n_epochs):
        interaction_count = (
            gs.compute_repulsive_force_bh() if barnes_hut else gs.compute_repulsive_force_exact()
        )
        interaction_percentage = 100*interaction_count/gs.N/gs.N
        if not barnes_hut:
            interaction_count = gs.N * gs.N
        gs.add_attractive_force()
        gs.add_l1_penalty_z(epoch / 2_000)
        gs.step_adam_decay_z(lr, 1.0)
        if verbose and epoch % 100 == 0:
            forces = gs.get_forces()
            pos = gs.get_positions()
            avg_force = np.mean(np.abs(forces))
            print(
                f"Epoch {epoch}: avg_z={np.mean(np.abs(pos[:, 2])):.2f}, "
                f"lnp={gs.get_lnp_exact()}, lnp_bh={gs.get_lnp_bh()}"
            )

train_session(gs, n_epochs, lr=lr, barnes_hut=True, verbose=True)
train_session_decay(gs, 2_000, lr=lr, barnes_hut=True, verbose=True)

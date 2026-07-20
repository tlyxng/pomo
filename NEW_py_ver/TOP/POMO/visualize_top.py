"""
visualize_top.py

Sanity-check / visualization helpers for TOPEnv instances.

Usage (from inside your NEW_py_ver/TOP/POMO folder, so the imports resolve):

    from TOPEnv import TOPEnv
    from visualize_top import plot_instance, plot_solution, sanity_check, random_rollout

    env = TOPEnv(problem_size=20, pomo_size=20)
    env.load_problems(batch_size=4)
    reset_state, _, _ = env.reset()

    sanity_check(env)                      # prints structural checks
    plot_instance(env, idx=0)              # draws the raw instance
    selected = random_rollout(env)         # random policy full rollout
    plot_solution(env, selected, idx=0, pomo_idx=0)   # draws the resulting route(s)
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def sanity_check(env):
    """Print structural checks that catch the most common 'this isn't really
    a TOP instance' bugs: missing multi-leg support, degenerate prizes,
    max_length that never binds, etc."""
    rs = env.reset_state
    print("=== TOP instance sanity check ===")
    print(f"batch_size={env.batch_size}, problem_size={env.problem_size}, pomo_size={env.pomo_size}")
    print(f"depot_xy shape:   {tuple(rs.depot_xy.shape)}  (expect (batch, 1, 2))")
    print(f"node_xy shape:    {tuple(rs.node_xy.shape)}   (expect (batch, problem, 2))")
    print(f"node_prize shape: {tuple(rs.node_prize.shape)} (expect (batch, problem))")
    print(f"node_prize range: [{rs.node_prize.min().item():.3f}, {rs.node_prize.max().item():.3f}]")
    if rs.node_prize.min().item() < 0:
        print("  !! WARNING: negative prize found")
    if rs.node_prize.max().item() == rs.node_prize.min().item():
        print("  note: all prizes identical -> effectively unit-prize OP, easier than a general TOP")

    print(f"max_length: {rs.max_length.flatten().tolist()}")
    print(f"num_vehicles: {rs.num_vehicles.flatten().tolist()}")
    if (rs.num_vehicles <= 1).all():
        print("  !! num_vehicles is always 1 for every instance in this batch -> the env is degrading to a "
              "single-vehicle Orienteering Problem (OP), not a genuine multi-leg TOP. This alone would make "
              "training noticeably faster/easier than a real TOP, since the policy never has to learn "
              "return-to-depot-and-continue behavior.")

    # feasibility sanity: is max_length large enough to reach ANY single node and back?
    depot = rs.depot_xy[:, 0, :]  # (batch, 2)
    dists = (rs.node_xy - depot[:, None, :]).norm(dim=2)  # (batch, problem)
    round_trip = 2 * dists
    reachable_frac = (round_trip <= rs.max_length).float().mean(dim=1)
    print(f"fraction of nodes individually reachable (depot->node->depot) per instance: "
          f"{reachable_frac.tolist()}")
    if (reachable_frac == 0).any():
        print("  !! WARNING: at least one instance has ZERO reachable nodes given max_length "
              "-> reward will always be 0 for it")
    if (reachable_frac == 1).any():
        print("  note: at least one instance can reach every node in a single leg -> max_length may be too "
              "generous, making the length constraint non-binding")


def plot_instance(env, idx=0, ax=None, title_extra=""):
    """Draw depot, nodes (sized/colored by prize), and the single-leg reachability
    circle (radius = max_length/2, since a leg must go depot->node->depot)."""
    rs = env.reset_state
    depot = rs.depot_xy[idx, 0].detach().cpu().numpy()
    nodes = rs.node_xy[idx].detach().cpu().numpy()
    prize = rs.node_prize[idx].detach().cpu().numpy()
    max_length = rs.max_length[idx].item()
    num_vehicles = rs.num_vehicles[idx].item()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))

    # reachability circle: any single node with 2*dist(depot,node) <= max_length
    # can be visited on one leg by itself.
    circle = patches.Circle(depot, radius=max_length / 2, fill=False,
                             linestyle="--", edgecolor="gray", linewidth=1)
    ax.add_patch(circle)

    sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=prize, cmap="viridis",
                     s=60 + 300 * (prize / (prize.max() + 1e-9)), edgecolors="black", zorder=3)
    ax.scatter(*depot, marker="*", s=400, c="red", edgecolors="black", zorder=4, label="depot")

    plt.colorbar(sc, ax=ax, label="prize", fraction=0.046, pad=0.04)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"TOP instance {idx}  |  problem_size={len(prize)}  "
                 f"max_length={max_length:.2f}  num_vehicles={num_vehicles:.0f}{title_extra}")
    ax.legend(loc="upper right")

    if own_fig:
        plt.tight_layout()
        plt.savefig("top_instance.png", dpi=150)
        print("saved top_instance.png")
        plt.close(fig)


def random_rollout(env):
    """Run a full episode with a uniform-random *feasible* policy (samples only
    among unmasked actions). Returns selected_node_list, shape (batch, pomo, steps)."""
    state, reward, done = env.pre_step()
    while not done:
        mask = state.ninf_mask  # (batch, pomo, problem+1), 0 = allowed, -inf = forbidden
        probs = (mask == 0).float()
        probs = probs / probs.sum(dim=2, keepdim=True)
        b, p, n = probs.shape
        selected = probs.reshape(b * p, n).multinomial(1).reshape(b, p)
        state, reward, done = env.step(selected)
    return env.selected_node_list, reward


def plot_solution(env, selected_node_list, idx=0, pomo_idx=0, ax=None):
    """Draw the route(s) actually taken for one (batch, pomo) pair, coloring each
    leg (depot -> ... -> depot) differently so you can visually confirm multi-leg
    behavior is really happening."""
    rs = env.reset_state
    depot = rs.depot_xy[idx, 0].detach().cpu().numpy()
    node_xy = env.depot_node_xy[idx].detach().cpu().numpy()  # (problem+1, 2), index 0 = depot

    seq = selected_node_list[idx, pomo_idx].detach().cpu().numpy().tolist()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))

    plot_instance(env, idx=idx, ax=ax)

    # split sequence into legs at each return to depot (node index 0)
    legs = []
    current_leg = [0]  # start at depot
    for node in seq:
        current_leg.append(node)
        if node == 0:
            legs.append(current_leg)
            current_leg = [0]
    if len(current_leg) > 1:
        legs.append(current_leg)

    cmap = plt.get_cmap("tab10")
    for i, leg in enumerate(legs):
        pts = node_xy[leg]
        ax.plot(pts[:, 0], pts[:, 1], color=cmap(i % 10), linewidth=2, alpha=0.8,
                label=f"leg {i+1}", zorder=2)

    ax.set_title(ax.get_title() + f"\n{len(legs)} leg(s) used, {len(seq)} steps total")
    ax.legend(loc="upper right", fontsize=8)

    if own_fig:
        plt.tight_layout()
        plt.savefig("top_solution.png", dpi=150)
        print("saved top_solution.png")
        plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from TOPEnv import TOPEnv

    env = TOPEnv(problem_size=20, pomo_size=20)
    env.load_problems(batch_size=4)
    env.reset()

    sanity_check(env)
    plot_instance(env, idx=0)

    env2 = TOPEnv(problem_size=20, pomo_size=20)
    env2.load_problems(batch_size=4)
    env2.reset()
    selected, reward = random_rollout(env2)
    print("random-policy reward for instance 0, pomo 0:", reward[0, 0].item())
    plot_solution(env2, selected, idx=0, pomo_idx=0)

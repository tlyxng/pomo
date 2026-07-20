"""
TOP Sanity-Check / Debugger
============================
Run this in the SAME folder as TOPEnv.py, TOPModel.py, train_n20.py.
It does NOT touch your training loop -- it's a standalone script.

    python debug_top.py                     # env + random-model checks only
    python debug_top.py ./result/xxx 510    # also loads a checkpoint and
                                             # compares its rollout to optimal

What it checks:
  1. A tiny 5-node instance small enough to brute-force the TRUE optimal
     prize, so you have ground truth to compare the model against.
  2. The env's step-by-step behavior on that instance (budget tracking,
     masking, reward) via manually forced actions.
  3. Common bug patterns: missing Step_State fields, NaN in probabilities,
     all -inf mask rows, negative "remaining_length", wrong reward sign.
  4. A plot of the model's chosen route vs. the depot/nodes (matplotlib),
     saved to route_debug.png, so you can eyeball whether it respects the
     budget and looks like a sane route.
"""

import sys
import itertools
import torch

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../../..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

from TOPEnv import TOPEnv
from TOPModel import TOPModel

# ---------------------------------------------------------------------------
# 1. Toy instance with a known-by-brute-force optimal answer
# ---------------------------------------------------------------------------
def build_toy_instance():
    """
    5 nodes around the depot at distance ~1, one high-value node (10),
    three low-value nodes (1 each), one medium node (5). Budget=3.0 per leg,
    1 vehicle/leg total -- deliberately too tight to visit everything, so
    the optimal solution has to make a real prize/budget tradeoff.
    """
    depot_xy = torch.tensor([[[0.0, 0.0]]])                    # (1,1,2)
    node_xy = torch.tensor([[[1.0, 0.0],
                              [0.0, 1.0],
                              [-1.0, 0.0],
                              [0.0, -1.0],
                              [0.5, 0.5]]])                      # (1,5,2)
    node_prize = torch.tensor([[10.0, 1.0, 1.0, 1.0, 5.0]])     # (1,5)
    max_length = torch.tensor([[3.0]])                          # (1,1)
    num_vehicles = torch.tensor([[1]])                          # (1,1)
    return depot_xy, node_xy, node_prize, max_length, num_vehicles


def _partitions(total, max_parts):
    """All ways to split `total` positions into <= max_parts contiguous,
    possibly-empty, ordered groups. Used to try every leg-cut of a route."""
    if max_parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _partitions(total - first, max_parts - 1):
            yield (first,) + rest


def brute_force_optimal(depot_xy, node_xy, node_prize, max_length, num_vehicles):
    """Exhaustive search over subsets/orderings/leg-splits. Only tractable
    for small n (<=6 or so) -- that's the point, it's ground truth."""
    depot = depot_xy[0, 0]
    nodes = node_xy[0]
    prizes = node_prize[0]
    budget = max_length[0, 0].item()
    m = int(num_vehicles[0, 0].item())
    n = nodes.shape[0]

    def dist(a, b):
        return ((a - b) ** 2).sum().sqrt().item()

    best_prize, best_route = 0.0, []
    for k in range(n + 1):
        for subset in itertools.combinations(range(n), k):
            for perm in itertools.permutations(subset):
                for cuts in _partitions(len(perm), m):
                    idx, legs, feasible = 0, [], True
                    for c in cuts:
                        leg = perm[idx:idx + c]
                        idx += c
                        leg_len, prev = 0.0, depot
                        for node_i in leg:
                            leg_len += dist(prev, nodes[node_i])
                            prev = nodes[node_i]
                        leg_len += dist(prev, depot)
                        if leg_len > budget + 1e-6:
                            feasible = False
                            break
                        legs.append(leg)
                    if feasible:
                        total = sum(prizes[i].item() for i in subset)
                        if total > best_prize:
                            best_prize, best_route = total, legs
    return best_prize, best_route


# ---------------------------------------------------------------------------
# 2. Env-level checks: load the toy instance directly into TOPEnv's internals
#    and verify reward/masking behave as expected under a forced route.
# ---------------------------------------------------------------------------
def make_env_with_toy_instance(depot_xy, node_xy, node_prize, max_length, num_vehicles):
    env = TOPEnv(problem_size=node_xy.shape[1], pomo_size=1)
    env.batch_size = 1
    env.max_length = max_length
    env.num_vehicles = num_vehicles
    env.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
    depot_prize = torch.zeros(size=(1, 1))
    env.depot_node_prize = torch.cat((depot_prize, node_prize), dim=1)
    env.BATCH_IDX = torch.arange(1)[:, None].expand(1, 1)
    env.POMO_IDX = torch.arange(1)[None, :].expand(1, 1)
    env.reset_state.depot_xy = depot_xy
    env.reset_state.node_xy = node_xy
    env.reset_state.node_prize = node_prize
    env.reset_state.max_length = max_length
    env.reset_state.num_vehicles = num_vehicles
    env.step_state.BATCH_IDX = env.BATCH_IDX
    env.step_state.POMO_IDX = env.POMO_IDX
    return env


def run_forced_route(env, route_legs):
    """route_legs: list of legs, each a tuple of node indices (1-indexed
    into the problem, i.e. node i -> selected id i+1). Depot returns are
    inserted automatically between legs. Returns the final reward."""
    env.reset()
    env.pre_step()
    actions = [0]  # forced first move must be depot per TOPModel convention
    for leg in route_legs:
        for node_i in leg:
            actions.append(node_i + 1)
        actions.append(0)  # return to depot at end of each leg
    # env.reset() already put us "before" the first move; TOPModel forces
    # the very first selected action to 0 (depot) -- replicate that here.
    reward = None
    done = False
    for a in actions:
        selected = torch.tensor([[a]])
        state, reward, done = env.step(selected)
        if done:
            break
    return reward, done


def check_env_matches_brute_force():
    print("=" * 70)
    print("CHECK 1: env reward vs. brute-force optimal on toy instance")
    print("=" * 70)
    depot_xy, node_xy, node_prize, max_length, num_vehicles = build_toy_instance()
    optimal_prize, optimal_route = brute_force_optimal(
        depot_xy, node_xy, node_prize, max_length, num_vehicles)
    print(f"Brute-force optimal prize: {optimal_prize}")
    print(f"Brute-force optimal route (0-indexed node ids per leg): {optimal_route}")

    env = make_env_with_toy_instance(depot_xy, node_xy, node_prize, max_length, num_vehicles)
    reward, done = run_forced_route(env, optimal_route)

    if reward is None:
        print("[FAIL] Route did not terminate 'done' -- check env.finished logic")
        return
    achieved = reward[0, 0].item()
    print(f"Env reward when forced through the optimal route: {achieved}")

    if abs(achieved - optimal_prize) < 1e-4:
        print("[OK] Env reward matches brute-force optimal (correct sign, correct magnitude)")
    elif abs(achieved + optimal_prize) < 1e-4:
        print("[FAIL] Env reward is the NEGATIVE of the optimal prize -- reward sign bug is back")
    else:
        print("[FAIL] Env reward does not match optimal prize at all -- "
              "check prize accumulation / budget tracking logic")


# ---------------------------------------------------------------------------
# 3. Common-bug assertion checklist, run against a live env + model
# ---------------------------------------------------------------------------
def run_bug_checklist(env_params, model_params, batch_size=8):
    print("=" * 70)
    print("CHECK 2: common-bug checklist on a random real-sized instance")
    print("=" * 70)
    env = TOPEnv(**env_params)
    model = TOPModel(**model_params)
    model.eval()

    env.load_problems(batch_size)
    reset_state, _, _ = env.reset()

    # Bug #1: current_node must not be None once we start stepping
    if env.current_node is None:
        print("[FAIL] env.current_node is None after reset() -- "
              "first env.step() will crash. Initialize it to zeros in reset().")
    else:
        print("[OK] current_node initialized (not None) after reset()")

    model.pre_forward(reset_state)
    state, reward, done = env.pre_step()

    # Bug #2: Step_State must carry everything TOPModel.forward needs
    required_fields = ['selected_count', 'load', 'current_node', 'ninf_mask',
                        'BATCH_IDX', 'POMO_IDX']
    missing = [f for f in required_fields
               if not hasattr(state, f) or getattr(state, f) is None]
    if missing:
        print(f"[FAIL] Step_State missing fields the model needs: {missing}")
    else:
        print("[OK] Step_State has selected_count/load/current_node/ninf_mask")

    nan_seen = False
    neg_remaining_seen = False
    all_inf_row_seen = False
    step_count = 0

    with torch.no_grad():
        while not done:
            selected, prob = model(state)
            state, reward, done = env.step(selected)
            step_count += 1

            if torch.isnan(state.ninf_mask).any():
                nan_seen = True
            if (state.ninf_mask == float('-inf')).all(dim=2).any():
                # a fully-masked row is only OK if that (batch,pomo) is finished
                bad = (state.ninf_mask == float('-inf')).all(dim=2) & (~state.finished)
                if bad.any():
                    all_inf_row_seen = True
            if (state.remaining_length < -1e-4).any():
                neg_remaining_seen = True

            if step_count > env.problem_size * 3:
                print("[FAIL] Rollout did not terminate after 3x problem_size steps "
                      "-- possible infinite loop (check `finished` logic)")
                break

    print(f"Rollout finished after {step_count} steps.")

    if nan_seen:
        print("[FAIL] NaN detected in ninf_mask/probs during rollout")
    else:
        print("[OK] No NaNs detected during rollout")

    if all_inf_row_seen:
        print("[FAIL] Found an all -inf mask row on a non-finished (batch,pomo) -- "
              "the depot is getting masked out, softmax will NaN. "
              "Make sure depot masking ignores the budget-feasibility check.")
    else:
        print("[OK] No illegal all -inf mask rows found")

    if neg_remaining_seen:
        print("[FAIL] remaining_length went negative -- a move was allowed that "
              "violates the budget constraint")
    else:
        print("[OK] remaining_length stayed non-negative throughout")

    if reward is not None:
        if (reward < 0).any():
            print("[FAIL] Final reward has negative entries -- collected prize "
                  "should never be negative. Check the reward sign in env.step().")
        else:
            print(f"[OK] Final reward is non-negative "
                  f"(mean={reward.float().mean().item():.3f}, "
                  f"max={reward.max().item():.3f})")


# ---------------------------------------------------------------------------
# 4. Compare a (trained or untrained) model's rollout to the toy optimum,
#    and plot the chosen route.
# ---------------------------------------------------------------------------
def evaluate_and_plot(model_params, checkpoint_path=None, checkpoint_epoch=None):
    print("=" * 70)
    print("CHECK 3: model rollout on the toy instance vs. brute-force optimal")
    print("=" * 70)
    depot_xy, node_xy, node_prize, max_length, num_vehicles = build_toy_instance()
    optimal_prize, _ = brute_force_optimal(depot_xy, node_xy, node_prize, max_length, num_vehicles)

    model = TOPModel(**model_params)
    if checkpoint_path is not None:
        ckpt_file = f'{checkpoint_path}/checkpoint-{checkpoint_epoch}.pt'
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {ckpt_file}")
    else:
        print("No checkpoint given -- using a randomly-initialized model "
              "(expect a poor, near-arbitrary result, this just confirms the "
              "pipeline runs end-to-end).")
    model.eval()

    env = TOPEnv(problem_size=node_xy.shape[1], pomo_size=node_xy.shape[1])
    env.batch_size = 1
    env.max_length = max_length
    env.num_vehicles = num_vehicles
    env.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
    depot_prize = torch.zeros(size=(1, 1))
    env.depot_node_prize = torch.cat((depot_prize, node_prize), dim=1)
    env.BATCH_IDX = torch.arange(1)[:, None].expand(1, env.pomo_size)
    env.POMO_IDX = torch.arange(env.pomo_size)[None, :].expand(1, env.pomo_size)
    env.reset_state.depot_xy = depot_xy
    env.reset_state.node_xy = node_xy
    env.reset_state.node_prize = node_prize
    env.reset_state.max_length = max_length
    env.reset_state.num_vehicles = num_vehicles
    env.step_state.BATCH_IDX = env.BATCH_IDX
    env.step_state.POMO_IDX = env.POMO_IDX

    with torch.no_grad():
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

    best_pomo_idx = reward[0].argmax().item()
    achieved = reward[0, best_pomo_idx].item()
    route = env.selected_node_list[0, best_pomo_idx].tolist()

    print(f"Brute-force optimal prize: {optimal_prize}")
    print(f"Model's best-of-pomo prize: {achieved}  (route node ids, 0=depot: {route})")
    gap = (optimal_prize - achieved) / max(optimal_prize, 1e-6) * 100
    print(f"Optimality gap: {gap:.1f}%")
    if checkpoint_path is None and achieved > optimal_prize:
        print("[FAIL] Untrained model beat brute-force optimal -- "
              "this means the optimal computation or the reward is wrong, "
              "not that the model is good.")

    _plot_route(depot_xy, node_xy, node_prize, route)


def _plot_route(depot_xy, node_xy, node_prize, route):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping route plot)")
        return

    depot = depot_xy[0, 0].tolist()
    nodes = node_xy[0].tolist()
    prizes = node_prize[0].tolist()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([depot[0]], [depot[1]], c='black', marker='s', s=120, label='depot', zorder=3)
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    ax.scatter(xs, ys, c='steelblue', s=[p * 30 for p in prizes], zorder=3, label='node (size=prize)')
    for i, (x, y) in enumerate(nodes):
        ax.annotate(f"{i}: {prizes[i]:.0f}", (x, y), textcoords="offset points", xytext=(6, 6))

    coords = [depot] + nodes
    path_xy = [coords[node_id] for node_id in route]
    px = [p[0] for p in path_xy]
    py = [p[1] for p in path_xy]
    ax.plot(px, py, c='coral', linewidth=2, zorder=2, label='chosen route')

    ax.set_title('Model route vs. instance')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    out_path = 'route_debug.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved route plot to {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env_params = {'problem_size': 20, 'pomo_size': 20}
    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** 0.5,
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
    }

    check_env_matches_brute_force()
    print()
    run_bug_checklist(env_params, model_params, batch_size=8)
    print()

    ckpt_path, ckpt_epoch = None, None
    if len(sys.argv) == 3:
        ckpt_path, ckpt_epoch = sys.argv[1], sys.argv[2]
    evaluate_and_plot(model_params, ckpt_path, ckpt_epoch)
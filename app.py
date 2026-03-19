from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

ACTIONS = ['up', 'down', 'left', 'right']
DELTA   = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}
ARROW   = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

GAMMA       = 0.9
THETA       = 1e-6
REWARD_STEP = -0.1
REWARD_GOAL = 1.0
MAX_ITER    = 10000


def transition(r, c, action, n, obs_set):
    dr, dc = DELTA[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in obs_set:
        return nr, nc
    return r, c


def get_reward(nr, nc, goal):
    return REWARD_GOAL if (nr, nc) == goal else REWARD_STEP


# ── HW1-2 ─────────────────────────────────────────────────────────────────────
def make_random_policy(n, obs_set, goal):
    policy = {}
    for r in range(n):
        for c in range(n):
            if (r, c) in obs_set or (r, c) == goal:
                policy[(r, c)] = None
            else:
                policy[(r, c)] = random.choice(ACTIONS)
    return policy


def policy_evaluation(n, policy, obs_set, goal):
    """Iterative policy evaluation (synchronous update) until convergence.
    V[goal] = 0 (terminal state). Reward is received upon entering goal.
    Neighbour of goal: V = R_goal + gamma*V[goal] = 1.0 + 0.9*0 = 1.0
    Two steps away:    V = R_step + gamma*1.0      = -0.1 + 0.9 = 0.8
    """
    V = {(r, c): 0.0 for r in range(n) for c in range(n)}
    # goal is terminal: V[goal] = 0, no further transitions
    for _ in range(MAX_ITER):
        delta = 0.0
        new_V = V.copy()
        for r in range(n):
            for c in range(n):
                if (r, c) in obs_set or (r, c) == goal:
                    continue
                a = policy.get((r, c))
                if a is None:
                    continue
                nr, nc = transition(r, c, a, n, obs_set)
                v = get_reward(nr, nc, goal) + GAMMA * V[(nr, nc)]
                delta = max(delta, abs(v - V[(r, c)]))
                new_V[(r, c)] = v
        V = new_V
        if delta < THETA:
            break
    return V


# ── HW1-3 ─────────────────────────────────────────────────────────────────────
def value_iteration(n, obs_set, goal):
    """Value iteration → optimal V* and optimal policy.
    V[goal] = 0 (terminal). Reward received on entering goal.
    """
    V = {(r, c): 0.0 for r in range(n) for c in range(n)}
    # V[goal] stays 0 — it's a terminal absorbing state
    for _ in range(MAX_ITER):
        delta = 0.0
        new_V = V.copy()
        for r in range(n):
            for c in range(n):
                if (r, c) in obs_set or (r, c) == goal:
                    continue
                best = max(
                    get_reward(*transition(r, c, a, n, obs_set), goal)
                    + GAMMA * V[transition(r, c, a, n, obs_set)]
                    for a in ACTIONS
                )
                delta = max(delta, abs(best - V[(r, c)]))
                new_V[(r, c)] = best
        V = new_V
        if delta < THETA:
            break

    opt_policy = {}
    for r in range(n):
        for c in range(n):
            if (r, c) in obs_set or (r, c) == goal:
                opt_policy[(r, c)] = None
                continue
            opt_policy[(r, c)] = max(
                ACTIONS,
                key=lambda a: (
                    get_reward(*transition(r, c, a, n, obs_set), goal)
                    + GAMMA * V[transition(r, c, a, n, obs_set)]
                )
            )
    return V, opt_policy


def find_path(n, start, goal, obs_set, opt_policy):
    path, cur, visited = [start], start, {start}
    for _ in range(n * n * 2):
        if cur == goal:
            break
        a = opt_policy.get(cur)
        if not a:
            break
        nxt = transition(*cur, a, n, obs_set)
        if nxt == cur or nxt in visited:
            break
        visited.add(nxt)
        path.append(nxt)
        cur = nxt
    return path


def ser_V(V, n):
    return {f"{r},{c}": round(V[(r, c)], 3) for r in range(n) for c in range(n)}


def ser_policy(policy, n):
    return {f"{r},{c}": ARROW.get(policy.get((r, c)), '') for r in range(n) for c in range(n)}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/random_policy', methods=['POST'])
def api_random_policy():
    """Step 1: generate & return a random policy (no evaluation yet)."""
    d       = request.json
    n       = int(d['n'])
    obs_set = set(tuple(o) for o in d['obstacles'])
    goal    = tuple(d['goal'])
    policy  = make_random_policy(n, obs_set, goal)
    return jsonify({'rand_policy': ser_policy(policy, n)})


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """Step 2: evaluate a given fixed policy + run value iteration."""
    d       = request.json
    n       = int(d['n'])
    obs_set = set(tuple(o) for o in d['obstacles'])
    start   = tuple(d['start'])
    goal    = tuple(d['goal'])
    # rebuild policy from arrows sent by frontend
    arrow_to_action = {'↑':'up','↓':'down','←':'left','→':'right'}
    raw = d['rand_policy']   # dict "r,c" -> arrow str
    policy = {}
    for r in range(n):
        for c in range(n):
            k = f"{r},{c}"
            a = arrow_to_action.get(raw.get(k,''))
            policy[(r, c)] = a  # None for goal/obstacle/unknown

    # HW1-2: policy evaluation on the fixed random policy
    V_rand = policy_evaluation(n, policy, obs_set, goal)

    # HW1-3: value iteration
    V_opt, opt_pol = value_iteration(n, obs_set, goal)
    path = find_path(n, start, goal, obs_set, opt_pol)

    return jsonify({
        'rand_values': ser_V(V_rand, n),
        'opt_policy':  ser_policy(opt_pol, n),
        'opt_values':  ser_V(V_opt, n),
        'path':        [[r, c] for r, c in path],
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)


@app.route('/standalone')
def standalone():
    return render_template('standalone.html')
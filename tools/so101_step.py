"""Step response of the SO-101's servos: the measurement that replaces the guesses in so101.SERVO (P-loop stiffness,
damping, stall torque, no-load speed, rotor inertia, the servo's own goal ramp, the bus delay).

On the computer the arm is plugged into, with nothing else holding the serial port (stop the collector / LeRobot).
Start pose: every joint at the middle of its range (LeRobot's 0 in degrees mode, the URDF zero: upper arm up, forearm
and gripper pointing forward). Clear 0.5 m around the arm and keep a hand on the power switch. Needs only numpy and
the Feetech SDK (`pip install feetech-servo-sdk`, which LeRobot already installs); this file can be copied on its own.

    python so101_step.py check --port /dev/ttyACM0                         # registers, supply, pose; nothing moves
    python so101_step.py run --port /dev/ttyACM0 --out so101_step.json     # 4 joints x 10/20/30 deg x both ways, ~1 min
                                                                           # add --cup once the cup is on the wrist
    python so101_step.py run --port /dev/ttyACM0 --accel 0 --speed 0 --out so101_step_fast.json   # servo ramps off

Here, with the repository:

    python tools/so101_step.py sim --out so101_step_sim.json               # the same protocol on the simulator
    python tools/so101_step.py report so101_step.json so101_step_sim.json  # dead time, speed, acceleration, overshoot
    python tools/so101_step.py fit so101_step.json --out so101_servo.json  # then SO101_SERVO=so101_servo.json

Each step: 0.1 s before, the goal moves by +-amp on one joint and stays `hold` s, then returns and stays `hold` s; the
other joints keep their goals. Raw goals only: no gravity compensation, so the droop is in the record too.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
IDS = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "wrist_roll": 5}   # LeRobot SO-101
TESTED = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
TICKS = 4096                           # per turn
DEG = 360.0 / TICKS
# STS3215 control table (address, bytes), as LeRobot's feetech tables
REG = {"Model_Number": (3, 2), "Min_Position_Limit": (9, 2), "Max_Position_Limit": (11, 2), "Max_Torque_Limit": (16, 2),
       "P_Coefficient": (21, 1), "D_Coefficient": (22, 1), "I_Coefficient": (23, 1), "Homing_Offset": (31, 2),
       "Operating_Mode": (33, 1), "Torque_Enable": (40, 1), "Acceleration": (41, 1), "Goal_Position": (42, 2),
       "Goal_Velocity": (46, 2), "Torque_Limit": (48, 2), "Present_Position": (56, 2), "Present_Voltage": (62, 1),
       "Present_Temperature": (63, 1), "Maximum_Acceleration": (85, 1)}
BLOCK = (56, 15)                       # Present_Position .. Present_Current in one sync read


def signed(value, bit):
    """Feetech sign-magnitude: `bit` is the sign."""
    return -(value & ((1 << bit) - 1)) if value & (1 << bit) else value


# ----- hardware -----------------------------------------------------------------------------------------------------
class Bus:
    def __init__(self, port, baud, ids):
        import scservo_sdk as scs
        self.scs, self.ids = scs, ids
        self.port = scs.PortHandler(port)
        self.packet = scs.PacketHandler(0)                 # protocol 0: STS / SMS
        if not self.port.openPort() or not self.port.setBaudRate(baud):
            sys.exit(f"cannot open {port} at {baud}")
        self.sync = scs.GroupSyncRead(self.port, self.packet, *BLOCK)
        for i in ids.values():
            self.sync.addParam(i)

    def read(self, motor, name):
        addr, n = REG[name]
        fn = self.packet.read1ByteTxRx if n == 1 else self.packet.read2ByteTxRx
        value, comm, err = fn(self.port, self.ids[motor], addr)
        if comm != self.scs.COMM_SUCCESS:
            raise IOError(f"{motor}: read {name}: {self.packet.getTxRxResult(comm)}")
        return value

    def write(self, motor, name, value):
        addr, n = REG[name]
        fn = self.packet.write1ByteTxRx if n == 1 else self.packet.write2ByteTxRx
        comm, err = fn(self.port, self.ids[motor], addr, int(value))
        if comm != self.scs.COMM_SUCCESS:
            raise IOError(f"{motor}: write {name}: {self.packet.getTxRxResult(comm)}")

    def state(self):
        """Raw position, velocity, load, current of every motor; None when a reply is missing."""
        if self.sync.txRxPacket() != self.scs.COMM_SUCCESS:
            return None
        out = []
        for m in JOINTS:
            i = self.ids[m]
            if not self.sync.isAvailable(i, *BLOCK):
                return None
            g = lambda a, n: self.sync.getData(i, a, n)
            out.append((g(56, 2), signed(g(58, 2), 15), signed(g(60, 2), 10), signed(g(69, 2), 15)))
        return out


def wrap(raw, ref):
    """Ticks from ref, unwrapped."""
    return ((raw - ref + TICKS // 2) % TICKS) - TICKS // 2


def registers(bus):
    return {m: {k: bus.read(m, k) for k in REG} for m in JOINTS}


def check(args, quiet=False):
    bus = Bus(args.port, args.baud, IDS)
    regs = registers(bus)
    mid = {m: (regs[m]["Min_Position_Limit"] + regs[m]["Max_Position_Limit"]) / 2 for m in JOINTS}
    if not quiet:
        print(f"{'joint':14s} {'id':>3s} {'mode':>4s} {'P':>3s} {'D':>3s} {'I':>3s} {'acc':>4s} {'speed':>5s} "
              f"{'volt':>5s} {'temp':>4s} {'torque':>6s} {'from middle':>11s}")
        for m in JOINTS:
            r = regs[m]
            print(f"{m:14s} {IDS[m]:3d} {r['Operating_Mode']:4d} {r['P_Coefficient']:3d} {r['D_Coefficient']:3d} "
                  f"{r['I_Coefficient']:3d} {r['Acceleration']:4d} {r['Goal_Velocity']:5d} "
                  f"{r['Present_Voltage'] / 10:5.1f} "
                  f"{r['Present_Temperature']:4d} {r['Torque_Enable']:6d} "
                  f"{wrap(r['Present_Position'], mid[m]) * DEG:+10.1f} deg")
    return bus, regs, mid


def run(args):
    bus, regs, mid = check(args)
    volts = np.mean([regs[m]["Present_Voltage"] for m in JOINTS]) / 10
    bad = [m for m in JOINTS if regs[m]["Operating_Mode"] != 0]
    if bad:
        sys.exit(f"not in position mode: {bad}")
    off = {m: wrap(regs[m]["Present_Position"], mid[m]) * DEG for m in JOINTS}
    far = [m for m in TESTED if abs(off[m]) > args.start_tolerance]
    if far:
        sys.exit(f"start pose: {far} more than {args.start_tolerance} deg from the middle of their range; "
                 "move the arm to LeRobot's 0 pose first (or raise --start-tolerance)")
    for m in args.joints:
        lo = wrap(regs[m]["Min_Position_Limit"], mid[m]) * DEG
        hi = wrap(regs[m]["Max_Position_Limit"], mid[m]) * DEG
        if off[m] - max(args.amps) - 5 < lo or off[m] + max(args.amps) + 5 > hi:
            sys.exit(f"{m}: +-{max(args.amps)} deg from {off[m]:+.1f} leaves its range [{lo:.0f}, {hi:.0f}]")
    print(f"\nsupply {volts:.1f} V. {len(args.joints)} joints x {args.amps} deg x both ways, "
          f"about {len(args.joints) * len(args.amps) * 2 * (2 * args.hold + 0.5):.0f} s.")
    if not args.yes and input("Area clear and power switch in reach? type yes: ").strip() != "yes":
        sys.exit("stopped")
    saved = {m: (regs[m]["Acceleration"], regs[m]["Goal_Velocity"]) for m in JOINTS}
    start = {m: regs[m]["Present_Position"] for m in JOINTS}
    for m in JOINTS:                                # hold where it is, then torque on
        bus.write(m, "Goal_Position", start[m])
        if args.accel is not None:
            bus.write(m, "Acceleration", args.accel)
        if args.speed is not None:
            bus.write(m, "Goal_Velocity", args.speed)
        bus.write(m, "Torque_Enable", 1)
    time.sleep(0.5)
    records = []
    try:
        for m in args.joints:
            for amp in args.amps:
                for sign in (1, -1):
                    rec = step(bus, m, amp, sign, start, args.hold)
                    if rec is None:
                        raise RuntimeError("aborted")
                    records.append(rec)
                    s = summary(rec)
                    print(f"{m:14s} {sign * amp:+5.0f} deg: dead {s['dead_ms']:5.1f} ms  rise {s['rise_ms']:5.0f} ms  "
                          f"peak {s['peak_speed']:5.0f} deg/s  accel {s['accel']:6.0f} deg/s2  overshoot "
                          f"{s['overshoot']:4.0f}%  settle {s['settle_ms']:5.0f} ms  "
                          f"droop {s['steady_err']:+5.2f} deg  "
                          f"({s['rate_hz']:.0f} Hz)", flush=True)
                    temp = max(bus.read(j, "Present_Temperature") for j in JOINTS)
                    if temp >= args.max_temp:
                        print(f"servo at {temp} C: stopping")
                        raise RuntimeError("hot")
                    time.sleep(0.3)
    except (KeyboardInterrupt, RuntimeError, IOError) as e:
        print(f"stopped: {e!r}")
    finally:
        now = bus.state()
        for k, m in enumerate(JOINTS):             # hold where the arm is, put the ramps back
            try:
                bus.write(m, "Goal_Position", now[k][0] if now else start[m])
                bus.write(m, "Acceleration", saved[m][0])
                bus.write(m, "Goal_Velocity", saved[m][1])
            except IOError:
                pass
    meta = {"source": "so101", "port": args.port, "supply_v": volts, "registers": regs, "middle": mid,
            "accel_set": args.accel, "speed_set": args.speed, "cup": args.cup, "wrist_camera": not args.no_wrist_camera,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(args.out).write_text(json.dumps({"meta": meta, "records": records}) + "\n")
    print(f"{len(records)} steps -> {args.out}")


def step(bus, motor, amp, sign, start, hold, pre=0.1):
    """One step out and back on `motor`: samples (t from the first command, s) of every joint in degrees from its
    start, with the commands' times."""
    k = JOINTS.index(motor)
    goal_out = int(round(start[motor] + sign * amp / DEG)) % TICKS
    t0 = time.perf_counter() + pre
    phase, cmd = 0, {}
    rows = []
    while True:
        now = time.perf_counter()
        if phase == 0 and now >= t0:
            bus.write(motor, "Goal_Position", goal_out)
            cmd["out"], phase = time.perf_counter() - t0, 1
        elif phase == 1 and now >= t0 + hold:
            bus.write(motor, "Goal_Position", start[motor])
            cmd["back"], phase = time.perf_counter() - t0, 2
        elif phase == 2 and now >= t0 + 2 * hold:
            break
        s = bus.state()
        t = time.perf_counter() - t0
        if s is None:
            continue
        pos = [wrap(s[j][0], start[m]) * DEG for j, m in enumerate(JOINTS)]
        moved = [abs(p) for j, p in enumerate(pos) if j != k]
        if max(moved) > 10:
            print(f"another joint moved {max(moved):.0f} deg: stop")
            return None
        rows.append([t, *pos, *(x[1] for x in s), *(x[2] for x in s), *(x[3] for x in s)])
    R = np.array(rows)
    n = len(JOINTS)
    return {"joint": motor, "amp": amp, "sign": sign, "hold": hold, "cmd": cmd, "t": R[:, 0].tolist(),
            "pos": R[:, 1:1 + n].tolist(), "vel_raw": R[:, 1 + n:1 + 2 * n].tolist(),
            "load_raw": R[:, 1 + 2 * n:1 + 3 * n].tolist(), "current_raw": R[:, 1 + 3 * n:].tolist()}


# ----- analysis -----------------------------------------------------------------------------------------------------
def summary(rec):
    """The out step of one record: dead time, 10-90 % rise, peak speed, mean acceleration to it, overshoot, settling
    (inside max(1 deg, 5 %)), steady error (droop) over the last 0.15 s, sample rate."""
    k = JOINTS.index(rec["joint"])
    t = np.asarray(rec["t"]) - rec["cmd"].get("out", 0.0)
    x = np.asarray(rec["pos"])[:, k] * rec["sign"]                  # along the step
    amp, hold = rec["amp"], rec["hold"]
    base = np.mean(x[t < 0]) if (t < 0).any() else x[0]
    w = (t >= 0) & (t < hold - 0.005)
    tt, xx = t[w], x[w] - base
    out = {"rate_hz": len(t) / (t[-1] - t[0]) if len(t) > 1 else 0.0}
    moved = np.flatnonzero(np.abs(xx) > 0.5)
    out["dead_ms"] = 1000 * tt[moved[0]] if moved.size else np.nan
    i10, i90 = np.flatnonzero(xx > 0.1 * amp), np.flatnonzero(xx > 0.9 * amp)
    out["rise_ms"] = 1000 * (tt[i90[0]] - tt[i10[0]]) if i10.size and i90.size else np.nan
    v = np.gradient(xx, tt) if len(tt) > 3 else np.zeros_like(xx)
    v = np.convolve(v, np.ones(3) / 3, mode="same")
    j = int(np.argmax(v)) if len(v) else 0
    out["peak_speed"] = float(v[j]) if len(v) else np.nan
    out["accel"] = float(v[j] / max(tt[j] - (tt[moved[0]] if moved.size else 0.0), 1e-3)) if len(v) else np.nan
    out["overshoot"] = float(max(xx.max() - amp, 0.0) / amp * 100) if len(xx) else np.nan
    band = max(1.0, 0.05 * amp)
    outside = np.flatnonzero(np.abs(xx - amp) > band)
    out["settle_ms"] = 1000 * tt[outside[-1]] if outside.size else 0.0
    tail = tt > tt[-1] - 0.15
    out["steady_err"] = float(np.mean(xx[tail]) - amp) if tail.any() else np.nan
    return out


def report(args):
    tables = [(Path(f).name, json.loads(Path(f).read_text())) for f in args.files]
    keys = ("dead_ms", "rise_ms", "peak_speed", "accel", "overshoot", "settle_ms", "steady_err")
    print(f"{'step':24s} " + " ".join(f"{k:>11s}" for k in keys))
    for name, data in tables:
        print(f"-- {name} ({data['meta'].get('source')}, {data['meta'].get('supply_v', '?')} V)")
        for rec in data["records"]:
            s = summary(rec)
            print(f"{rec['joint']:14s} {rec['sign'] * rec['amp']:+5.0f} deg " + " ".join(f"{s[k]:11.1f}" for k in keys))


# ----- simulator ----------------------------------------------------------------------------------------------------
def _sim(supply=None, meta=None):
    """The simulated arm as the record's arm: its supply, with or without the cup and the wrist camera."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from taucatch.vla_catch import so101
    meta = meta or {}
    if supply:
        so101.SUPPLY = supply
    so101.WRIST_CAMERA = meta.get("wrist_camera", True)
    if not meta.get("cup", False):
        so101.CUP_MASS = 1e-4                  # the arm as it comes: the jaws, no cup
    from taucatch.vla_catch.collect import make_sim
    return so101, make_sim("so101")


def sim_record(so101, sim, motor, amp, sign, hold, pre=0.1, settle=0.5, every=0.002, flip=None):
    """The step protocol on the simulator from the all-zero pose. flip: per-joint sign of the real raw direction
    against the model's angle (1 when they agree)."""
    import mujoco
    m, d = sim.m, sim.d
    mujoco.mj_resetDataKeyframe(m, d, sim.ready_key)
    d.qpos[sim.qadr] = 0.0
    d.qvel[:] = 0.0
    sim.place_ball(np.array([3.0, 3.0, 0.035]))
    mujoco.mj_forward(m, d)
    k = JOINTS.index(motor)
    s = 1.0 if flip is None else flip[k]
    zero = np.zeros(len(JOINTS))
    sim.goal, sim.setpoint, sim.setpoint_v = zero.copy(), zero.copy(), zero.copy()
    for _ in range(int(settle / m.opt.timestep)):
        sim.step_physics()
    rows, n = [], int(round((pre + 2 * hold) / m.opt.timestep))
    out_goal = zero.copy()
    out_goal[k] = np.radians(sign * amp) * s
    for i in range(n):
        t = i * m.opt.timestep - pre
        sim.goal = out_goal if 0 <= t < hold else zero
        sim.step_physics()
        if i % int(round(every / m.opt.timestep)) == 0:
            rows.append([t, *(np.degrees(d.qpos[sim.qadr]) * (1.0 if flip is None else np.asarray(flip)))])
    R = np.array(rows)
    return {"joint": motor, "amp": amp, "sign": sign, "hold": hold, "cmd": {"out": 0.0, "back": hold},
            "t": R[:, 0].tolist(), "pos": R[:, 1:].tolist()}


def simulate(args):
    payload = {"cup": args.cup, "wrist_camera": not args.no_wrist_camera}
    so101, sim = _sim(args.supply, payload)
    if args.servo:
        so101.load_servo(args.servo)
        sim.apply_servo()
    records = [sim_record(so101, sim, m, a, s, args.hold) for m in args.joints for a in args.amps for s in (1, -1)]
    meta = {"source": "simulator", "supply_v": so101.SUPPLY, "servo": dict(so101.SERVO), **payload}
    Path(args.out).write_text(json.dumps({"meta": meta, "records": records}) + "\n")
    for rec in records:
        s = summary(rec)
        print(f"{rec['joint']:14s} {rec['sign'] * rec['amp']:+5.0f} deg: rise {s['rise_ms']:5.0f} ms  peak "
              f"{s['peak_speed']:5.0f} deg/s  accel {s['accel']:6.0f} deg/s2  overshoot {s['overshoot']:4.0f}%  "
              f"settle {s['settle_ms']:5.0f} ms  droop {s['steady_err']:+5.2f} deg")
    print(f"-> {args.out}")


# ----- fit ----------------------------------------------------------------------------------------------------------
FIT = {                                # parameter: (low, high), searched in log space (delay: linear)
    "sat_deg": (0.3, 30.0),
    "zeta": (0.1, 2.0),
    "stall_74": (0.5, 5.0),
    "no_load_74": (2.0, 10.0),
    "armature": (0.005, 0.1),
    "setpoint_accel": (5.0, 1000.0),   # rad/s^2; the top of the range means no ramp (None)
    "delay": (0.0, 0.03),              # s
}


def _params(x):
    p = {}
    for (k, (lo, hi)), v in zip(FIT.items(), x):
        p[k] = float(np.clip(v / 100, lo, hi)) if k == "delay" else float(np.exp(np.clip(v, np.log(lo), np.log(hi))))
    if p["setpoint_accel"] > 0.9 * FIT["setpoint_accel"][1]:
        p["setpoint_accel"] = None
    return p


def _vector(p):
    return np.array([100 * p[k] if k == "delay" else np.log(p[k] if p[k] is not None else hi)
                     for k, (lo, hi) in FIT.items()])


def _error(so101, sim, records, params, flip):
    """RMS (deg) of the model against the measured steps, the model delayed by params['delay']."""
    so101.SERVO.update({k: v for k, v in params.items() if k != "delay"})
    sim.apply_servo()
    err = []
    for rec in records:
        k = JOINTS.index(rec["joint"])
        model = sim_record(so101, sim, rec["joint"], rec["amp"], rec["sign"], rec["hold"], flip=flip)
        t = np.asarray(rec["t"]) - rec["cmd"]["out"]
        x = np.asarray(rec["pos"])[:, k]
        mt, mx = np.asarray(model["t"]) + params["delay"], np.asarray(model["pos"])[:, k]
        w = (t > 0) & (t < 2 * rec["hold"])
        err.append(x[w] - np.interp(t[w], mt, mx))
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


def _descend(job):
    """Nelder-Mead from one start, in a fresh process (the simulator's module settings are per process)."""
    meta, records, flip, x0, evals = job
    from scipy.optimize import minimize
    so101, sim = _sim(meta.get("supply_v"), meta)
    res = minimize(lambda x: _error(so101, sim, records, _params(x), flip), x0, method="Nelder-Mead",
                   options={"maxfev": evals, "xatol": 1e-3, "fatol": 1e-4,
                            "initial_simplex": np.vstack([x0, x0 + 0.4 * np.eye(len(x0))])})
    return float(res.fun), res.x


def _flips(meta, records, params, before=None):
    """Raw direction against the model's angle, per joint (gravity makes the two directions differ): the better one.
    A joint gravity does not load (shoulder_pan) fits both alike and keeps its direction."""
    so101, sim = _sim(meta.get("supply_v"), meta)
    flip = np.ones(len(JOINTS)) if before is None else before.copy()
    for m in sorted({r["joint"] for r in records}):
        k = JOINTS.index(m)
        mine = [r for r in records if r["joint"] == m]
        trial = {}
        for s in (1.0, -1.0):
            flip[k] = s
            trial[s] = _error(so101, sim, mine, params, flip)
        if abs(trial[1.0] - trial[-1.0]) < 0.05 * min(trial.values()) + 0.01:
            flip[k] = 1.0 if before is None else before[k]
            word = "either way (no gravity load)"
        else:
            flip[k] = min(trial, key=trial.get)
            word = "agrees with the model" if flip[k] > 0 else "is opposite to the model"
        print(f"{m}: raw direction {word} (rms {trial[1.0]:.2f} vs {trial[-1.0]:.2f} deg)", flush=True)
    return flip


def fit(args):
    import multiprocessing as mp
    data = json.loads(Path(args.file).read_text())
    meta, records = data["meta"], data["records"]
    so101, _ = _sim(meta.get("supply_v"), meta)
    params = {k: so101.SERVO[k] for k in FIT if k != "delay"}
    params["delay"] = 0.005
    if params["setpoint_accel"] is None:
        params["setpoint_accel"] = FIT["setpoint_accel"][1]
    rng = np.random.default_rng(0)
    flip = _flips(meta, records, _params(_vector(params)))
    for round_ in range(2):
        def draw():
            return np.array([rng.uniform(100 * lo, 100 * hi) if k == "delay" else rng.uniform(np.log(lo), np.log(hi))
                             for k, (lo, hi) in FIT.items()])
        starts = [_vector(params)] + [draw() for _ in range(args.starts - 1)]
        with mp.Pool(min(args.starts, mp.cpu_count()), maxtasksperchild=1) as pool:
            results = pool.map(_descend, [(meta, records, flip, x0, args.evals) for x0 in starts])
        results.sort(key=lambda r: r[0])
        best, params = results[0][0], _params(results[0][1])
        print(f"round {round_ + 1}: rms {best:.3f} deg over {args.starts} starts "
              f"(next best {', '.join(f'{r[0]:.3f}' for r in results[1:4])})", flush=True)
        again = _flips(meta, records, params, flip)
        if np.array_equal(again, flip):
            break
        flip = again
    servo = {k: v for k, v in params.items() if k != "delay"}
    for k, v in params.items():
        print(f"  {k:15s} {v}")
    out = {"servo": servo, "delay_s": params["delay"], "flip": flip.tolist(), "rms_deg": best,
           "supply_v": meta.get("supply_v"), "cup": meta.get("cup", False),
           "wrist_camera": meta.get("wrist_camera", True), "from": args.file}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"fit rms {best:.3f} deg -> {args.out}  (use: SO101_SERVO={args.out} SO101_SUPPLY={out['supply_v']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--port", required=True)
        p.add_argument("--baud", type=int, default=1_000_000)
    p = sub.choices["run"]
    p.add_argument("--out", required=True)
    p.add_argument("--joints", nargs="+", default=list(TESTED), choices=JOINTS)
    p.add_argument("--amps", nargs="+", type=float, default=[10.0, 20.0, 30.0])
    p.add_argument("--hold", type=float, default=0.8)
    p.add_argument("--accel", type=int, default=None, help="set register 41 (Acceleration) for the test; 0: no ramp")
    p.add_argument("--speed", type=int, default=None, help="set register 46 (Goal_Velocity) for the test; 0: max")
    p.add_argument("--start-tolerance", type=float, default=15.0, help="deg from the middle of each range")
    p.add_argument("--max-temp", type=int, default=60)
    p.add_argument("--yes", action="store_true")
    for p in (sub.choices["run"], sub.add_parser("sim")):
        p.add_argument("--cup", action="store_true", help="the cup is on the arm (default: the jaws as they come)")
        p.add_argument("--no-wrist-camera", action="store_true", help="the wrist camera and its mount are off")
    p = sub.choices["sim"]
    p.add_argument("--out", required=True)
    p.add_argument("--joints", nargs="+", default=list(TESTED), choices=JOINTS)
    p.add_argument("--amps", nargs="+", type=float, default=[10.0, 20.0, 30.0])
    p.add_argument("--hold", type=float, default=0.8)
    p.add_argument("--supply", type=float, default=None)
    p.add_argument("--servo", default=None, help="fitted servo JSON")
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("fit")
    p.add_argument("file")
    p.add_argument("--out", required=True)
    p.add_argument("--starts", type=int, default=8, help="Nelder-Mead starts, run in parallel")
    p.add_argument("--evals", type=int, default=400, help="model evaluations per start")
    args = ap.parse_args()
    {"check": check, "run": run, "sim": simulate, "report": report, "fit": fit}[args.cmd](args)


if __name__ == "__main__":
    main()

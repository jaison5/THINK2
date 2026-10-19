from __future__ import annotations

import numpy as np


class T:
    __slots__ = ("data", "grad", "_parents", "_back", "requires_grad")
    __array_ufunc__ = None

    def __init__(self, data, parents=(), back=None, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._parents = parents
        self._back = back
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    shape = property(lambda self: self.data.shape)

    def backward(self, g=None):
        order, seen = [], set()

        def visit(t):
            if id(t) in seen or not t.requires_grad:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)

        visit(self)
        self.grad = np.ones_like(self.data) if g is None else g
        for t in reversed(order):
            if t._back is not None and t.grad is not None:
                grads = t._back(t.grad)
                for p, gp in zip(t._parents, grads):
                    if p.requires_grad and gp is not None:
                        p.grad = gp if p.grad is None else p.grad + gp
            if t._parents:
                t.grad = None if t._back is not None else t.grad

    def __add__(self, o): return add(self, o)
    def __radd__(self, o): return add(o, self)
    def __sub__(self, o): return add(self, neg(o))
    def __rsub__(self, o): return add(o, neg(self))
    def __mul__(self, o): return mul(self, o)
    def __rmul__(self, o): return mul(o, self)
    def __neg__(self): return neg(self)
    def __matmul__(self, o): return matmul(self, o)
    def __getitem__(self, idx): return getitem(self, idx)


class Param(T):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)
        self.grad = None


def _t(x):
    return x if isinstance(x, T) else T(x)


def _unb(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def add(a, b):
    a, b = _t(a), _t(b)
    return T(a.data + b.data, (a, b), lambda g: (_unb(g, a.shape), _unb(g, b.shape)))


def neg(a):
    a = _t(a)
    return T(-a.data, (a,), lambda g: (-g,))


def mul(a, b):
    a, b = _t(a), _t(b)
    return T(a.data * b.data, (a, b), lambda g: (_unb(g * b.data, a.shape), _unb(g * a.data, b.shape)))


def matmul(a, b):
    a, b = _t(a), _t(b)

    def back(g):
        ga = g @ np.swapaxes(b.data, -1, -2)
        gb = np.swapaxes(a.data, -1, -2) @ g
        return _unb(ga, a.shape), _unb(gb, b.shape)

    return T(a.data @ b.data, (a, b), back)


def sigmoid(a):
    s = 1.0 / (1.0 + np.exp(-np.clip(a.data, -60, 60)))
    return T(s, (a,), lambda g: (g * s * (1 - s),))


def tanh(a):
    t = np.tanh(a.data)
    return T(t, (a,), lambda g: (g * (1 - t * t),))


def exp(a):
    e = np.exp(a.data)
    return T(e, (a,), lambda g: (g * e,))


def log(a):
    return T(np.log(a.data), (a,), lambda g: (g / a.data,))


def square(a):
    return T(a.data ** 2, (a,), lambda g: (2 * g * a.data,))


def sum_(a, axis=None, keepdims=False):
    def back(g):
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        return (np.broadcast_to(g, a.shape).copy(),)

    return T(a.data.sum(axis=axis, keepdims=keepdims), (a,), back)


def mean(a, axis=None):
    n = a.data.size if axis is None else a.data.shape[axis]
    return mul(sum_(a, axis), 1.0 / n)


def concat(ts, axis=-1):
    ts = [_t(t) for t in ts]
    sizes = [t.shape[axis] for t in ts]
    cuts = np.cumsum(sizes)[:-1]

    def back(g):
        return tuple(np.split(g, cuts, axis=axis))

    return T(np.concatenate([t.data for t in ts], axis=axis), tuple(ts), back)


def getitem(a, idx):
    def back(g):
        z = np.zeros_like(a.data)
        np.add.at(z, idx, g)
        return (z,)

    return T(a.data[idx], (a,), back)


def reshape(a, shape):
    return T(a.data.reshape(shape), (a,), lambda g: (g.reshape(a.shape),))


def tile_rows(a, n):
    return T(np.broadcast_to(a.data, (n,) + a.shape).copy(), (a,), lambda g: (g.sum(axis=0),))


def log_softmax(a):
    m = a.data.max(axis=-1, keepdims=True)
    z = a.data - m
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    out = z - lse
    sm = np.exp(out)
    return T(out, (a,), lambda g: (g - sm * g.sum(axis=-1, keepdims=True),))


_RY, _RZ, _CNOT, _CZ = 0, 1, 2, 3


class CircuitRef:

    def __init__(self, n_qubits, ops, n_params, measure, init="zero"):
        self.n, self.n_params, self.init = n_qubits, n_params, init
        self.measure = list(measure)
        self.ops = []
        for op in ops:
            kind = {"ry": _RY, "rz": _RZ, "cnot": _CNOT}[op[0]]
            self.ops.append((kind, op[1], op[2]))
        self.has_rz = any(k == _RZ for k, _, _ in self.ops)
        dim = 2 ** n_qubits
        bits = (np.arange(dim)[:, None] >> (n_qubits - 1 - np.arange(n_qubits))[None, :]) & 1
        self.zsign = 1.0 - 2.0 * bits[:, self.measure]  # (dim, m)
        self.perm = {}
        for kind, a, b in self.ops:
            if kind == _CNOT and (a, b) not in self.perm:
                src = np.arange(dim)
                flip = bits[:, a] == 1
                tgt = src.copy()
                tgt[flip] = src[flip] ^ (1 << (n_qubits - 1 - b))
                self.perm[(a, b)] = tgt

    def _psi0(self, B):
        dim = 2 ** self.n
        dt = np.complex128 if self.has_rz else np.float64
        if self.init == "plus":
            return np.full((B, dim), 1 / np.sqrt(dim), dtype=dt)
        psi = np.zeros((B, dim), dtype=dt)
        psi[:, 0] = 1
        return psi

    def _view(self, psi, q):
        B = psi.shape[0]
        return psi.reshape(B, 2 ** q, 2, 2 ** (self.n - 1 - q))

    def _apply(self, psi, kind, a, theta, inverse=False, deriv=False):
        if kind == _CNOT:
            return psi[:, self.perm[(a, theta)]]
        v = self._view(psi, a)
        x0, x1 = v[:, :, 0, :], v[:, :, 1, :]
        th = theta[:, None, None] * (-1.0 if inverse else 1.0)
        out = np.empty_like(v)
        if kind == _RY:
            c, s = np.cos(th / 2), np.sin(th / 2)
            if deriv:
                out[:, :, 0, :] = 0.5 * (-s * x0 - c * x1)
                out[:, :, 1, :] = 0.5 * (c * x0 - s * x1)
            else:
                out[:, :, 0, :] = c * x0 - s * x1
                out[:, :, 1, :] = s * x0 + c * x1
        else:  # RZ
            e0, e1 = np.exp(-0.5j * th), np.exp(0.5j * th)
            if deriv:
                out[:, :, 0, :] = -0.5j * e0 * x0
                out[:, :, 1, :] = 0.5j * e1 * x1
            else:
                out[:, :, 0, :] = e0 * x0
                out[:, :, 1, :] = e1 * x1
        return out.reshape(psi.shape)

    def __call__(self, P):
        P = _t(P)
        th = P.data
        B = th.shape[0]
        psi = self._psi0(B)
        for kind, a, b in self.ops:
            psi = self._apply(psi, kind, a, b if kind == _CNOT else th[:, b])
        prob = (psi.conj() * psi).real
        ez = prob @ self.zsign  # (B, m)
        final = psi

        def back(g):
            obs = g @ self.zsign.T
            lam = obs * final
            psi_ = final
            gP = np.zeros_like(th)
            for kind, a, b in reversed(self.ops):
                if kind == _CNOT:
                    psi_ = psi_[:, self.perm[(a, b)]]
                    lam = lam[:, self.perm[(a, b)]]
                    continue
                t = th[:, b]
                psi_ = self._apply(psi_, kind, a, t, inverse=True)
                d = self._apply(psi_, kind, a, t, deriv=True)
                gP[:, b] += 2.0 * (lam.conj() * d).real.sum(axis=1)
                lam = self._apply(lam, kind, a, t, inverse=True)
            return (gP,)

        return T(ez, (P,), back)


def glorot(rng, fan_in, fan_out):
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=(fan_in, fan_out))


class Linear:
    def __init__(self, rng, n_in, n_out):
        self.W = Param(glorot(rng, n_in, n_out))
        self.b = Param(np.zeros(n_out))

    def __call__(self, x):
        return x @ self.W + self.b

    def params(self):
        return [self.W, self.b]


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = list(params), lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in self.p]
        self.v = [np.zeros_like(p.data) for p in self.p]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.p):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)

    def zero(self):
        for p in self.p:
            p.grad = None


class RMSprop(Adam):

    def __init__(self, params, lr=5e-3, alpha=0.99, eps=1e-8):
        super().__init__(params, lr)
        self.alpha, self.eps = alpha, eps

    def step(self):
        for i, p in enumerate(self.p):
            if p.grad is None:
                continue
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * p.grad ** 2
            p.data -= self.lr * p.grad / (np.sqrt(self.v[i]) + self.eps)


def n_params(params):
    return int(sum(p.data.size for p in params))


class Circuit(CircuitRef):

    def __init__(self, n_qubits, ops, n_params, measure, init="zero"):
        super().__init__(n_qubits, ops, n_params, measure, init)
        dim = 2 ** n_qubits
        bits = (np.arange(dim)[:, None] >> (n_qubits - 1 - np.arange(n_qubits))[None, :]) & 1
        self.zmat = 1.0 - 2.0 * bits
        prog = []
        for kind, a, b in self.ops:
            if kind == _CNOT:
                p = self.perm[(a, b)]
                if prog and prog[-1][0] == "perm":
                    prog[-1] = ("perm", prog[-1][1][p], None)
                else:
                    prog.append(("perm", p, None))
            elif kind == _RZ and prog and prog[-1][0] == "diag" and a not in prog[-1][1]:
                prog[-1][1].append(a); prog[-1][2].append(b)
            elif kind == _RZ:
                prog.append(("diag", [a], [b]))
            else:
                prog.append(("ry", a, b))
        self.prog = []
        for op in prog:
            if op[0] == "perm":
                inv = np.empty_like(op[1]); inv[op[1]] = np.arange(len(op[1]))
                self.prog.append(("perm", op[1], inv))
            elif op[0] == "diag":
                self.prog.append(("diag", np.array(op[1]), np.array(op[2])))
            else:
                self.prog.append(op)

    def _phase(self, th, qs, ks, sign=1.0):
        return np.exp(-0.5j * sign * (th[:, ks] @ self.zmat[:, qs].T))  # (B, dim)

    def __call__(self, P):
        P = _t(P)
        th = P.data
        B = th.shape[0]
        psi = self._psi0(B)
        for op in self.prog:
            if op[0] == "perm":
                psi = psi[:, op[1]]
            elif op[0] == "diag":
                psi = psi * self._phase(th, op[1], op[2])
            else:
                psi = self._apply(psi, _RY, op[1], th[:, op[2]])
        ez = (psi.conj() * psi).real @ self.zsign
        final = psi

        def back(g):
            lam = (g @ self.zsign.T) * final
            psi_ = final
            gP = np.zeros_like(th)
            for op in reversed(self.prog):
                if op[0] == "perm":
                    psi_ = psi_[:, op[2]]
                    lam = lam[:, op[2]]
                elif op[0] == "diag":
                    u = (-1j * lam.conj() * psi_).real  # 2·Re<λ|(−i/2)z ψ> = Re(−i λ̄ ψ)·z
                    gP[:, op[2]] += u @ self.zmat[:, op[1]]
                    ph = self._phase(th, op[1], op[2], sign=-1.0)
                    psi_ = psi_ * ph
                    lam = lam * ph
                else:
                    t = th[:, op[2]]
                    psi_ = self._apply(psi_, _RY, op[1], t, inverse=True)
                    d = self._apply(psi_, _RY, op[1], t, deriv=True)
                    gP[:, op[2]] += 2.0 * (lam.conj() * d).real.sum(axis=1)
                    lam = self._apply(lam, _RY, op[1], t, inverse=True)
            return (gP,)

        return T(ez, (P,), back)

"""Periodic, quasi-static isotropic rod authoring (N, mm, N*mm).

Independent implementation of straight-rest DER bending, axial elasticity and
periodic segment contact. HYLC inspires the micro-scale equilibrium workflow;
this is NOT its full Newton/homogenization solver. No friction or twist DOF.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


def segment_ends(q, period):
    end = np.roll(q, -1, axis=0).copy()
    end[-1, 0] += period[0]
    return end


def closest_segments(a, b, c, d):
    """Vectorized exact closest points of closed segments; robust parallel case.

    The minimum is among the unconstrained interior solution and four edge
    minima of the parameter square [0,1]^2. Returns distance, s, t, separation.
    """
    u, v, w = b-a, d-c, a-c
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    aa, bb, cc = dot(u, u), dot(u, v), dot(v, v)
    dd, ee = dot(u, w), dot(v, w)
    if np.any(aa <= 0) or np.any(cc <= 0):
        raise ValueError("Zero length contact segment")
    det = aa*cc-bb*bb
    safe = det > 1e-12*aa*cc
    si = np.divide(bb*ee-cc*dd, det, out=np.zeros_like(det), where=safe)
    ti = np.divide(aa*ee-bb*dd, det, out=np.zeros_like(det), where=safe)
    inside = safe & (si >= 0) & (si <= 1) & (ti >= 0) & (ti <= 1)
    ss = np.stack((np.zeros_like(aa), np.ones_like(aa),
                   np.clip(-dd/aa, 0, 1), np.clip((bb-dd)/aa, 0, 1), si), axis=1)
    tt = np.stack((np.clip(ee/cc, 0, 1), np.clip((ee+bb)/cc, 0, 1),
                   np.zeros_like(aa), np.ones_like(aa), ti), axis=1)
    delta = w[:, None, :] + ss[..., None]*u[:, None, :] - tt[..., None]*v[:, None, :]
    dist2 = np.einsum("ijk,ijk->ij", delta, delta)
    dist2[~inside, 4] = np.inf
    pick = np.argmin(dist2, axis=1)
    row = np.arange(len(a))
    return np.sqrt(dist2[row, pick]), ss[row, pick], tt[row, pick], delta[row, pick]


class RodModel:
    def __init__(self, rest_lengths, radius, settings):
        self.rest_lengths = np.asarray(rest_lengths, dtype=float).copy()
        self.radius = float(radius)
        if (self.rest_lengths.ndim != 1 or len(self.rest_lengths) < 3
                or not np.isfinite(self.rest_lengths).all() or np.any(self.rest_lengths <= 0)
                or not math.isfinite(self.radius) or self.radius <= 0):
            raise ValueError("Rod requires positive finite rest lengths and radius")
        self.settings = settings
        self.n = len(self.rest_lengths)
        self.dual = .5*(self.rest_lengths + np.roll(self.rest_lengths, 1))
        self.arc = np.r_[0., np.cumsum(self.rest_lengths)]
        self._pairs_cache = {}

    def pairs(self, q, period, other=None, other_period=None):
        """Unique unordered periodic pairs per material cell; local arc exclusion.

        Contact quadrature weights are segment rest-length products / diameter^2
        so that increasing node count does not simply multiply contact stiffness.
        """
        qs = [q, segment_ends(q, period)]
        if other is not None:
            qs += [other, segment_ends(other, other_period)]
        allq = np.concatenate(qs)
        minperiod = np.minimum(period, other_period) if other_period is not None else period
        pads = tuple(int(math.ceil((np.ptp(allq[:, k])+2*self.radius)/minperiod[k]))+1 for k in (0, 1))
        if max(pads) > 12:
            raise ValueError("Deformation exceeds supported periodic neighbor reach")
        if pads not in self._pairs_cache:
            ii, jj = np.meshgrid(np.arange(self.n), np.arange(self.n), indexing="ij")
            ii, jj = ii.ravel(), jj.ravel()
            records = []
            for dy in range(pads[1]+1):
                for dx in range(-pads[0], pads[0]+1):
                    if dy == 0 and dx < 0:
                        continue
                    keep = ii < jj if dx == dy == 0 else np.ones(len(ii), bool)
                    if dy == 0:
                        # Distance between material intervals along the same row.
                        first = self.arc[jj] + dx*self.arc[-1] - self.arc[ii+1]
                        second = self.arc[ii] - (self.arc[jj+1] + dx*self.arc[-1])
                        arcgap = np.maximum(first, second)
                        keep &= arcgap > math.pi*self.radius
                    i, j = ii[keep], jj[keep]
                    records.append(np.column_stack((i, j, np.full(len(i), dx), np.full(len(i), dy))))
            self._pairs_cache[pads] = np.concatenate(records).astype(int)
        return self._pairs_cache[pads]

    def contact_data(self, q, period, cutoff=None):
        pair = self.pairs(q, period)
        i, j, dx, dy = pair.T
        end = segment_ends(q, period)
        shift = np.column_stack((dx*period[0], dy*period[1], np.zeros(len(i))))
        a, b, c, d = q[i], end[i], q[j]+shift, end[j]+shift
        if cutoff is not None:
            lo = np.maximum(np.minimum(a,b), np.minimum(c,d))
            hi = np.minimum(np.maximum(a,b), np.maximum(c,d))
            keep = np.sum(np.maximum(lo-hi, 0)**2, axis=1) < cutoff**2
            pair, a, b, c, d = pair[keep], a[keep], b[keep], c[keep], d[keep]
        dist, s, t, delta = closest_segments(a,b,c,d)
        return pair, dist, s, t, delta

    def energy_gradient(self, q, period, components=False):
        p = self.settings
        q, period = np.asarray(q, dtype=float), np.asarray(period, dtype=float)
        if (q.shape != (self.n, 3) or not np.isfinite(q).all() or period.shape != (2,)
                or not np.isfinite(period).all() or np.any(period <= 0)):
            raise ValueError("Invalid periodic rod state")
        end = segment_ends(q, period)
        edges = end-q
        length = np.linalg.norm(edges, axis=1)
        if np.any(length < 1e-8*self.radius):
            raise ValueError("Collapsed rod edge")
        tangent = edges/length[:, None]
        strain = length/self.rest_lengths-1
        stretch = .5*p.axial_stiffness_N*np.sum(self.rest_lengths*strain**2)
        ge = p.axial_stiffness_N*strain[:, None]*tangent
        previous = np.roll(tangent, 1, axis=0)
        cosine = np.einsum("ij,ij->i", previous, tangent)
        if np.any(1+cosine < 1e-6):
            raise ValueError("Rod folded back at a vertex")
        bend = np.sum(2*p.bending_stiffness_N_mm2/self.dual*(1-cosine)/(1+cosine))
        dc = -4*p.bending_stiffness_N_mm2/self.dual/(1+cosine)**2
        gt = dc[:, None]*previous + np.roll(dc[:, None]*tangent, -1, axis=0)
        ge += (gt-tangent*np.sum(gt*tangent, axis=1)[:, None])/length[:, None]
        gradient = np.roll(ge, 1, axis=0)-ge
        margin = p.contact_margin_ratio*self.radius
        pair, dist, s, t, delta = self.contact_data(q, period, 2*self.radius+margin)
        gap = dist-2*self.radius
        if np.any(gap <= 0):
            raise ValueError("Intersecting nonlocal rod segments")
        contact = 0.
        if len(pair):
            i, j = pair[:, 0], pair[:, 1]
            weight = self.rest_lengths[i]*self.rest_lengths[j]/(2*self.radius)**2
            active = gap < margin
            g, w = gap[active], weight[active]
            k = p.contact_stiffness_N_per_mm
            log = np.log(g/margin)
            contact = float(np.sum(-k*w*(g-margin)**2*log))
            derivative = -k*w*(2*(g-margin)*log+(g-margin)**2/g)
            force = derivative[:, None]*delta[active]/dist[active, None]
            i, j, s, t = i[active], j[active], s[active], t[active]
            np.add.at(gradient, i, (1-s[:, None])*force)
            np.add.at(gradient, (i+1)%self.n, s[:, None]*force)
            np.add.at(gradient, j, -(1-t[:, None])*force)
            np.add.at(gradient, (j+1)%self.n, -t[:, None]*force)
        if not math.isfinite(stretch+bend+contact) or not np.isfinite(gradient).all():
            raise ValueError("Energy or forces exceed supported numeric range")
        if components:
            return float(stretch+bend+contact), gradient, {
                "stretch_energy_N_mm": float(stretch), "bend_energy_N_mm": float(bend),
                "contact_energy_N_mm": contact, "max_axial_strain": float(np.max(np.abs(strain))),
                "yarn_length_mm": float(length.sum()), "rest_yarn_length_mm": float(self.rest_lengths.sum()),
                "residual_force_N": float(np.max(np.linalg.norm(gradient, axis=1)))}
        return float(stretch+bend+contact), gradient

    def path_clear(self, q0, period0, q1, period1, max_depth=18):
        """Conservative swept-segment test using distance Lipschitz bounds.

        Tests the entire linear step, not just endpoints. Undecidable intervals
        are rejected. Applies only to the nonlocal capsule pairs in this model;
        local tube fold-over is separately checked in mesh reconstruction.
        """
        pair = self.pairs(q0, period0, q1, period1)
        i, j, dx, dy = pair.T
        e0, e1 = segment_ends(q0, period0), segment_ends(q1, period1)
        shift0 = np.column_stack((dx*period0[0], dy*period0[1], np.zeros(len(i))))
        shift1 = np.column_stack((dx*period1[0], dy*period1[1], np.zeros(len(i))))
        a,b,c,d = q0[i],e0[i],q0[j]+shift0,e0[j]+shift0
        va,vb,vc,vd = q1[i]-a,e1[i]-b,q1[j]+shift1-c,e1[j]+shift1-d
        # Remove a common velocity for tighter, translation-invariant bounds.
        vb,vc,vd,va = vb-va,vc-va,vd-va,np.zeros_like(va)
        lo1 = np.minimum.reduce((a,b,a+va,b+vb))
        hi1 = np.maximum.reduce((a,b,a+va,b+vb))
        lo2 = np.minimum.reduce((c,d,c+vc,d+vd))
        hi2 = np.maximum.reduce((c,d,c+vc,d+vd))
        near = np.sum(np.maximum(np.maximum(lo1,lo2)-np.minimum(hi1,hi2),0)**2,axis=1) <= (2*self.radius)**2
        a,b,c,d,va,vb,vc,vd = [x[near] for x in (a,b,c,d,va,vb,vc,vd)]
        if not len(a):
            return True
        speed = np.maximum(np.linalg.norm(va,axis=1),np.linalg.norm(vb,axis=1)) + np.maximum(np.linalg.norm(vc,axis=1),np.linalg.norm(vd,axis=1))
        def visit(ids, lo, hi, depth):
            middle = (lo+hi)*.5
            distance = closest_segments(a[ids]+middle*va[ids],b[ids]+middle*vb[ids],
                                        c[ids]+middle*vc[ids],d[ids]+middle*vd[ids])[0]
            if np.any(distance <= 2*self.radius*(1+1e-10)):
                return False
            uncertain = ids[distance-speed[ids]*(hi-lo)*.5 <= 2*self.radius*(1+1e-10)]
            if not len(uncertain):
                return True
            if depth == max_depth:
                return False
            return visit(uncertain,lo,middle,depth+1) and visit(uncertain,middle,hi,depth+1)
        return visit(np.arange(len(a)),0.,1.,0)


def initial_nodes(params, count):
    from .geometry import centerline
    t = np.linspace(0, 2*math.pi, max(8192,count*64)+1)
    dense = centerline(t, params)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
    target = np.linspace(0,arc[-1],count,endpoint=False)
    sampled = np.column_stack([np.interp(target,arc,dense[:,k]) for k in range(3)])
    # Keep phase anchored by the centroid, without pinning individual nodes.
    return sampled


def equilibrate(model, initial, period, progress=None):
    radius, p = model.radius, model.settings
    scale = p.bending_stiffness_N_mm2/radius
    center = initial.mean(axis=0)
    def evaluate(q):
        value, gradient = model.energy_gradient(q,period)
        gradient -= gradient.mean(axis=0)
        return value/scale, (gradient*radius/scale).ravel()
    start=time.perf_counter()
    final=initial.copy()
    value,grad=evaluate(final)
    memory=[]
    rejected=0
    message="Maximum iterations reached"
    # A feasible Armijo search is explicit here: an invalid trial is reduced,
    # never returned to a generic optimizer as a zero-gradient high-energy point.
    for iteration in range(p.max_iterations+1):
        residual=float(np.max(np.linalg.norm(grad.reshape(-1,3),axis=1))*scale/radius)
        if residual <= p.gradient_tolerance:
            message="Residual force tolerance satisfied"
            break
        if iteration == p.max_iterations:
            break
        if progress and iteration and iteration%250==0:
            progress(f"  iteration {iteration} residual={residual:.3g} N")
        direction=grad.copy()
        alphas=[]
        for s,y,rho in reversed(memory):
            alpha=rho*np.dot(s,direction)
            alphas.append(alpha)
            direction-=alpha*y
        if memory:
            s,y,rho=memory[-1]
            direction*=np.dot(s,y)/np.dot(y,y)
        for (s,y,rho),alpha in zip(memory,reversed(alphas)):
            direction+=s*(alpha-rho*np.dot(y,direction))
        direction=-direction
        slope=float(np.dot(grad,direction))
        if slope >= 0 or not np.isfinite(direction).all():
            memory.clear()
            direction=-grad
            slope=-float(np.dot(grad,grad))
        step=min(1.,.25/max(float(np.max(np.linalg.norm(direction.reshape(-1,3),axis=1))),1e-30))
        accepted=False
        for _ in range(48):
            trial=final+step*radius*direction.reshape(-1,3)
            trial+=center-trial.mean(axis=0)
            try:
                next_value,next_grad=evaluate(trial)
                sufficient=next_value <= value+1e-4*step*slope
                if sufficient and model.path_clear(final,period,trial,period):
                    accepted=True
                    break
            except ValueError:
                pass
            rejected+=1
            step*=.5
        if not accepted:
            message="Feasible line search could not make progress"
            break
        s=((trial-final)/radius).ravel()
        y=next_grad-grad
        sy=float(np.dot(s,y))
        if sy > 1e-10*np.linalg.norm(s)*np.linalg.norm(y):
            memory.append((s,y,1./sy))
            if len(memory)>30:
                memory.pop(0)
        final,value,grad=trial,next_value,next_grad
    energy,_,stats=model.energy_gradient(final,period,True)
    all_dist=model.contact_data(final,period)[1]
    stats.update(energy_N_mm=energy,min_nonlocal_gap_mm=float(np.min(all_dist)-2*radius),
                 iterations=iteration,rejected_trials=rejected,
                 optimizer="L-BFGS with feasible Armijo backtracking",
                 optimizer_message=message,elapsed_seconds=round(time.perf_counter()-start,3),
                 converged=bool(stats["residual_force_N"] <= p.gradient_tolerance))
    if not stats["converged"]:
        raise RuntimeError(f"Equilibrium did not converge: {stats}")
    if stats["max_axial_strain"] > p.max_strain:
        raise RuntimeError(f"Near-inextensible strain budget exceeded: {stats['max_axial_strain']:.4g}")
    return final,stats


@dataclass
class StretchResult:
    initial_nodes_mm: np.ndarray
    reference_nodes_mm: np.ndarray
    nodes_mm: np.ndarray
    rest_lengths_mm: np.ndarray
    reference_period_mm: tuple
    period_mm: tuple
    history: list
    states: list


def solve_stretch(params, settings, progress=print, reference_nodes=None):
    params.validate(); settings.validate()
    q=initial_nodes(params,settings.nodes)
    period0=np.asarray(params.period_mm)
    rest=np.linalg.norm(segment_ends(q,period0)-q,axis=1)
    model=RodModel(rest,params.R*params.scale_mm,settings)
    model.energy_gradient(q,period0)  # Reject initially intersecting curves.
    progress("Relaxing straight-rest yarn at the authored reference period")
    # Optional warm start is for a previously validated reference of this model.
    # Forces and strain are rechecked; rest lengths still come from Crane nodes.
    start_nodes = q if reference_nodes is None else np.asarray(reference_nodes, dtype=float)
    reference,stats=equilibrate(model,start_nodes,period0,progress)
    history=[dict(stage="reference",lambda_x=1.,lambda_y=1.,period_mm=period0.tolist(),**stats)]
    states=[reference.copy()]
    current,period=reference,period0.copy()
    for step in range(1,settings.load_steps+1):
        f=step/settings.load_steps
        ratios=np.array([1+f*(settings.lambda_x-1),1+f*(settings.lambda_y-1)])
        target=period0*ratios
        guess=current.copy()
        guess[:,:2]*=target/period
        if not model.path_clear(current,period,guess,target):
            raise RuntimeError("Loading predictor would cross yarns; increase load_steps")
        progress(f"Loading {step}/{settings.load_steps}: lambda=({ratios[0]:.4f}, {ratios[1]:.4f})")
        current,stats=equilibrate(model,guess,target,progress)
        period=target
        history.append(dict(stage="load",lambda_x=float(ratios[0]),lambda_y=float(ratios[1]),
                            period_mm=period.tolist(),**stats))
        states.append(current.copy())
    return StretchResult(q,reference,current,rest,tuple(period0),tuple(period),history,states)

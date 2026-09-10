"""TopoRetarget optimisation driver (paper Stages 1-4) with per-loss ablation.

Variables solved per frame:  base rotation d6 (6), base translation t (3),
joint angles q (20).  Objective:

    L = w_IM * E_IM            (interaction-mesh / Laplacian)        -- Stage 3
      + w_bone * E_bone        (relative bone-direction)             -- Stage 1
      + w_reg  * E_reg         (smoothness + base prior)             -- Stage 4
      + w_pen  * E_pen         (no-penetration, slack-like hinge)    -- Stage 4
      + w_lim  * E_lim         (joint-limit barrier)

Every term is logged each iteration and any term can be switched off (weight 0)
to visualise its effect -- that is the ablation mechanism for the viewer.
"""
import numpy as np
import torch

from .interaction import build_edges, laplacian_operator, edge_kind

TIP_IDS = [4, 8, 12, 16, 20]
MM = 1000.0  # positional losses are computed in millimetres so weights stay O(1)

# Tuned on the synthetic grasp; the SAME set is reused across objects/ablations.
DEFAULT_WEIGHTS = dict(IM=1.0, bone=8.0, reg=0.4, pen=25.0, lim=200.0)


def _unit(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def e_bone(kp_r, kp_s, bones, adj_pairs):
    dr = torch.stack([_unit(kp_r[b] - kp_r[a]) for a, b in bones])   # (Nbone,3)
    ds = torch.stack([_unit(kp_s[b] - kp_s[a]) for a, b in bones])
    loss = kp_r.new_zeros(())
    for k, l in adj_pairs:
        loss = loss + ((dr[k] - dr[l]) - (ds[k] - ds[l])).pow(2).sum()
    return loss / max(len(adj_pairs), 1)


def e_im(L, V_r, V_s):
    """Laplacian-coordinate mismatch, in mm^2."""
    dR = L @ V_r
    dS = L @ V_s
    return ((dR - dS) * MM).pow(2).sum(-1).mean()


def e_pen(obj, pts):
    """slack-like hinge: penalise points inside the object (phi<0), in mm^2."""
    phi = obj.sdf(pts)
    return (torch.clamp(-phi, min=0.0) * MM).pow(2).mean(), phi


def e_lim(q, lower, upper):
    return (torch.clamp(lower - q, min=0).pow(2).sum()
            + torch.clamp(q - upper, min=0).pow(2).sum())


def retarget(hand, frame, weights=None, kappa=80.0, n_iters=350, lr=0.02,
             log_every=5, device="cpu", verbose=False):
    """Run the optimisation for one frame.  Returns a dict with the full trace."""
    w = dict(DEFAULT_WEIGHTS); w.update(weights or {})
    dt = torch.float32

    human = torch.tensor(frame.human_kpts, dtype=dt, device=device)        # (21,3)
    obj_pts = torch.tensor(frame.obj_pts, dtype=dt, device=device)         # (No,3)
    n_hand = 21

    # ---- Stage 2: interaction mesh on the SOURCE vertices --------------------
    V_s_np = np.concatenate([frame.human_kpts, frame.obj_pts], 0)
    edges = build_edges(V_s_np)
    kinds = edge_kind(edges, n_hand)
    L, W = laplacian_operator(V_s_np, edges, kappa, device=device)
    V_s = torch.tensor(V_s_np, dtype=dt, device=device)

    # ---- Stage 1: base init by Procrustes alignment -------------------------
    q0, d6_0, t0 = hand.procrustes_init(frame.human_kpts)
    q = torch.tensor(q0, dtype=dt, device=device, requires_grad=True)
    d6 = torch.tensor(d6_0, dtype=dt, device=device, requires_grad=True)
    t = torch.tensor(t0, dtype=dt, device=device, requires_grad=True)
    q_ref = torch.tensor(q0, dtype=dt, device=device)
    d6_ref = torch.tensor(d6_0, dtype=dt, device=device)
    t_ref = torch.tensor(t0, dtype=dt, device=device)
    lower, upper = hand.lower, hand.upper

    opt = torch.optim.Adam([q, d6, t], lr=lr)
    trace = []

    def current_kpts():
        return hand.keypoints(q[None], d6[None], t[None])[0]   # (21,3)

    for it in range(n_iters + 1):
        kp = current_kpts()
        V_r = torch.cat([kp, obj_pts], 0)

        L_im = e_im(L, V_r, V_s) if w["IM"] else V_r.new_zeros(())
        L_bone = e_bone(kp, human, hand.bones, hand.adjacent_bone_pairs) if w["bone"] else V_r.new_zeros(())
        L_reg = (w_reg_term(q, q_ref, d6, d6_ref, t, t_ref)) if w["reg"] else V_r.new_zeros(())
        # penetration sample points: keypoints + bone midpoints
        mids = torch.stack([(kp[a] + kp[b]) * 0.5 for a, b in hand.bones])
        pen_pts = torch.cat([kp, mids], 0)
        L_pen, phi = e_pen(frame.obj, pen_pts)
        L_pen = L_pen if w["pen"] else V_r.new_zeros(())
        L_lim = e_lim(q, lower, upper) if w["lim"] else V_r.new_zeros(())

        total = (w["IM"] * L_im + w["bone"] * L_bone + w["reg"] * L_reg
                 + w["pen"] * L_pen + w["lim"] * L_lim)

        opt.zero_grad()
        total.backward()
        opt.step()
        with torch.no_grad():
            q.clamp_(lower, upper)

        if it % log_every == 0 or it == n_iters:
            with torch.no_grad():
                kp_d = current_kpts()
                tip_phi = frame.obj.sdf(kp_d[TIP_IDS])
                allpts = torch.cat([kp_d, torch.stack(
                    [(kp_d[a] + kp_d[b]) * 0.5 for a, b in hand.bones])], 0)
                phi_all = frame.obj.sdf(allpts)
                rec = dict(
                    iter=it,
                    losses=dict(IM=float(L_im), bone=float(L_bone), reg=float(L_reg),
                                pen=float(L_pen), lim=float(L_lim), total=float(total)),
                    q=q.detach().cpu().numpy().tolist(),
                    d6=d6.detach().cpu().numpy().tolist(),
                    t=t.detach().cpu().numpy().tolist(),
                    kpts=kp_d.cpu().numpy().tolist(),
                    contact_mm=float(tip_phi.abs().mean() * 1000),
                    max_pen_mm=float(torch.clamp(-phi_all, min=0).max() * 1000),
                )
                trace.append(rec)
            if verbose and it % (log_every * 10) == 0:
                print(f"  it{it:4d} total={float(total.detach()):.5f} "
                      f"IM={float(L_im):.5f} contact={rec['contact_mm']:.1f}mm "
                      f"pen={rec['max_pen_mm']:.1f}mm")

    return dict(trace=trace, edges=edges, edge_kinds=kinds, weights=w, kappa=kappa,
                final=trace[-1])


def w_reg_term(q, q_ref, d6, d6_ref, t, t_ref,
               lam_reg=1.0, lam_basepos=0.001, lam_baserot=0.5):
    """Gentle smoothness/base prior.  q in rad; base translation in mm."""
    return (lam_reg * (q - q_ref).pow(2).mean()
            + lam_basepos * ((t - t_ref) * MM).pow(2).mean()
            + lam_baserot * (d6 - d6_ref).pow(2).mean())

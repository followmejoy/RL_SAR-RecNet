import torch
import torch.nn as nn
import numpy as np
from collections import deque



# ============================================================
# Image utility
# ============================================================

def _to_2d(img: torch.Tensor):

    if img.dim() == 4:
        return img[0,0]

    if img.dim() == 3:
        return img[0]

    return img



# ============================================================
# Entropy
# ============================================================

@torch.no_grad()
def compute_image_entropy(
        img: torch.Tensor,
        bins:int=256
):

    img_2d = _to_2d(img)


    img_min = img_2d.min()

    img_max = img_2d.max()


    img_normalized = (

        (img_2d-img_min)

        /

        (img_max-img_min+1e-8)

        *

        255

    ).clamp(0,255)



    hist = torch.histc(

        img_normalized.float(),

        bins=bins,

        min=0,

        max=255

    )


    prob = hist/(hist.sum()+1e-8)


    entropy = -torch.sum(

        prob[prob>0]

        *

        torch.log2(
            prob[prob>0]
        )

    )


    return float(entropy.item())



# ============================================================
# Gradient
# ============================================================

@torch.no_grad()
def compute_gradient_mean(img):


    img_2d=_to_2d(img)


    img_4d = (

        img_2d

        .unsqueeze(0)

        .unsqueeze(0)

    )


    sobel_x=torch.tensor(
        [
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
        ],
        dtype=torch.float32,
        device=img.device
    ).view(1,1,3,3)


    sobel_y=torch.tensor(
        [
            [-1,-2,-1],
            [0,0,0],
            [1,2,1]
        ],
        dtype=torch.float32,
        device=img.device
    ).view(1,1,3,3)



    grad_x=nn.functional.conv2d(
        img_4d,
        sobel_x,
        padding=1
    )


    grad_y=nn.functional.conv2d(
        img_4d,
        sobel_y,
        padding=1
    )


    grad=torch.sqrt(

        grad_x.pow(2)

        +

        grad_y.pow(2)

        +

        1e-8

    )


    return float(
        grad.mean().item()
    )



# ============================================================
# Observable statistics
# ============================================================

@torch.no_grad()
def get_observable_stats(
        x_rec,
        residual
):


    res_abs=residual.abs()


    return {

        "x_mean":
        float(x_rec.mean().item()),


        "x_std":
        float(x_rec.std().item()),


        "res_mean":
        float(res_abs.mean().item()),


        "res_std":
        float(res_abs.std().item()),


        "res_max":
        float(res_abs.max().item()),


        "res_energy":
        float(
            torch.mean(
                res_abs.pow(2)
            ).item()
        ),


        "grad":
        compute_gradient_mean(x_rec),


        "entropy":
        compute_image_entropy(x_rec)

    }



# ============================================================
# State construction
# ============================================================

def build_state(

        stats:dict,

        init_stats:dict,

        history:deque,

        t:int,

        total_iter:int,

        prev_lambda_dn:float,

        prev_lambda_pr:float,

        max_lambda_dn:float,

        max_lambda_pr:float,

        device

):


    init_res = (
        init_stats["res_mean"]
        +
        1e-8
    )


    init_grad = (
        abs(init_stats["grad"])
        +
        1e-8
    )


    init_entropy = (
        abs(init_stats["entropy"])
        +
        1e-8
    )



    rel_res_drop = (

        init_stats["res_mean"]

        -

        stats["res_mean"]

    ) / init_res



    rel_grad = (

        stats["grad"]

        -

        init_stats["grad"]

    ) / init_grad



    rel_entropy=(

        stats["entropy"]

        -

        init_stats["entropy"]

    ) / init_entropy



    if len(history)==0:


        h1_res_drop=0.0

        h1_grad_change=0.0

        h1_entropy_change=0.0

        h2_res_drop=0.0

        h2_grad_change=0.0

        h2_entropy_change=0.0



    else:


        h1=history[-1]


        h1_res_drop=h1["res_drop"]

        h1_grad_change=h1["grad_change"]

        h1_entropy_change=h1["entropy_change"]



        if len(history)>=2:


            h2=history[-2]


            h2_res_drop=h2["res_drop"]

            h2_grad_change=h2["grad_change"]

            h2_entropy_change=h2["entropy_change"]


        else:

            h2_res_drop=0.0

            h2_grad_change=0.0

            h2_entropy_change=0.0



    state=torch.tensor(

        [

            stats["x_mean"],

            stats["x_std"],

            stats["res_mean"],

            stats["res_std"],

            stats["res_max"],

            stats["res_energy"],

            stats["grad"],

            stats["entropy"],


            t/max(total_iter-1,1),


            rel_res_drop,

            rel_grad,

            rel_entropy,


            h1_res_drop,

            h1_grad_change,

            h1_entropy_change,


            h2_res_drop,

            h2_grad_change,

            h2_entropy_change,


            prev_lambda_dn/(max_lambda_dn+1e-8),

            prev_lambda_pr/(max_lambda_pr+1e-8)

        ],

        dtype=torch.float32,

        device=device

    )


    return torch.nan_to_num(
        state,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4
    )
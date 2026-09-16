import torch
import numpy as np

from PIL import Image
import torchvision.transforms as transforms


from models.PnP_MFAMP import PnP_MFAMP_Feat
from models.SAC_Agent import SACAgent


from utils.SARop import CSA_echo


from utils.train_utils import (
    load_mask,
    load_thetas,
    compute_all_metrics
)


from utils.state_utils import (
    get_observable_stats,
    build_state
)



from collections import deque



# ======================================================
# Configuration
# ======================================================


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



PNP_CKPT = (
    "checkpoints/"
    "best_MFAMP_SAR.pth"
)


SAC_CKPT = (
    "checkpoints/"
    "best_Sac_Actor.pth"
)



import os


ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


import glob

TEST_IMAGE = glob.glob(
    os.path.join(
        ROOT,
        "test",
        "*.png"
    )
)[0]


SAMPLING_RATE = 50


SNR = 20





# ======================================================
# Load PnP
# ======================================================


def load_pnp():


    print("Loading PnP model...")


    checkpoint = torch.load(

        PNP_CKPT,

        map_location=DEVICE,
        weights_only=False,

    )



    model = PnP_MFAMP_Feat(

        iter_num=4,

        in_channels=1,

        cond_channels=16,

        feature_downsample=2,

        max_lambda_dn=0.14,

        max_lambda_pr=0.055,

        max_onsager=0.2,

        use_onsager=False,

        use_prior=True,

        use_beta_scale=True,

        film_strength=0.1

    )



    if "model_state" in checkpoint:

        state_dict = checkpoint["model_state"]

    else:

        state_dict = checkpoint



    model.load_state_dict(

        state_dict,

        strict=False

    )



    model.to(DEVICE)

    model.eval()



    return model





# ======================================================
# Load SAC
# ======================================================


def load_sac(model):


    print("Loading SAC actor...")


    agent = SACAgent(

        state_dim=20,

        action_dim=2,

        max_lambda_dn=model.max_lambda_dn,

        max_lambda_pr=model.max_lambda_pr,

        device=DEVICE

    )



    checkpoint = torch.load(

        SAC_CKPT,

        map_location=DEVICE,
        weights_only=False,

    )



    if (

        isinstance(checkpoint,dict)

        and

        "model_state" in checkpoint

    ):


        actor_state = checkpoint["model_state"]



    elif (

        isinstance(checkpoint,dict)

        and

        "actor" in checkpoint

    ):


        actor_state = checkpoint["actor"]



    else:

        actor_state = checkpoint




    agent.actor.load_state_dict(

        actor_state,

        strict=True

    )



    agent.actor.eval()



    return agent





# ======================================================
# Load image
# ======================================================


def load_image():


    print(
        "Loading:",
        TEST_IMAGE
    )


    img = Image.open(

        TEST_IMAGE

    ).convert("L")



    transform = transforms.ToTensor()



    x = transform(img)



    # [1,H,W]

    x = (

        x

        .unsqueeze(0)

        .float()

    )


    # normalize

    x = (

        x /

        (x.max()+1e-8)

    )


    x = x.to(DEVICE)



    return x





# ======================================================
# Reconstruction
# ======================================================


@torch.no_grad()
def test():



    model = load_pnp()


    sac = load_sac(model)



    x_gt = load_image()



    print(
        "Image size:",
        x_gt.shape
    )



    # ----------------------------
    # SAR operator
    # ----------------------------


    mask = load_mask(

        SAMPLING_RATE,

        DEVICE

    )



    thetas = load_thetas(

        DEVICE

    )



    # ----------------------------
    # Generate measurement
    # ----------------------------


    Y = CSA_echo(

        x_gt,

        thetas

    )



    power = torch.mean(

        torch.abs(Y).pow(2),

        dim=[2,3],

        keepdim=True

    )



    noise_std = (

        power.sqrt()

        *

        10**(-SNR/20)

    )



    noise = torch.complex(

        torch.randn_like(Y.real)

        *

        noise_std,


        torch.randn_like(Y.imag)

        *

        noise_std

    )



    y = (

        Y + noise

    ) * mask





    # ----------------------------
    # Initialization
    # ----------------------------


    x = model.initialize(

        y,

        thetas,

        mask

    )



    r_prev=None



    history = deque(

        maxlen=2

    )


    prev_dn=0.0

    prev_pr=0.0



    alpha=[]

    beta=[]




    # ----------------------------
    # Unfolding
    # ----------------------------


    for t in range(model.r):


        residual = (

            y -

            model.forward_G(

                x,

                thetas,

                mask

            )

        )



        stats = get_observable_stats(

            x,

            residual

        )



        if t==0:

            init_stats=stats



        state = build_state(

            stats,

            init_stats,

            history,

            t,

            model.r,

            prev_dn,

            prev_pr,

            model.max_lambda_dn,

            model.max_lambda_pr,

            DEVICE

        )



        action = sac.select_raw_action(

            state,

            deterministic=True

        )



        action = sac.map_action(

            action.unsqueeze(0)

        ).squeeze(0)




        lambda_dn=float(

            action[0]

        )


        lambda_pr=float(

            action[1]

        )



        alpha.append(

            lambda_dn

        )


        beta.append(

            lambda_pr

        )



        x,r = model.forward_step(

            x,

            y,

            thetas,

            mask,

            t,

            lambda_dn,

            lambda_pr,

            r_prev

        )



        r_prev=r


        prev_dn=lambda_dn

        prev_pr=lambda_pr



        history.append(

            {

            "res_drop":0,

            "grad_change":0,

            "entropy_change":0

            }

        )





    # ----------------------------
    # Metrics
    # ----------------------------


    result = compute_all_metrics(

        x,

        x_gt

    )



    print("\n==============================")

    print(" Reconstruction Result ")

    print("==============================")



    print(

        f"PSNR   : {result['psnr']:.4f}"

    )


    print(

        f"SSIM   : {result['ssim']:.4f}"

    )


    print(

        f"NMSE   : {result['nmse']:.6f}"

    )


    print(

        f"Entropy: {result['entropy']:.4f}"

    )



    print("\nLambda schedule")



    for i in range(len(alpha)):


        print(

            f"Stage {i+1}: "

            f"alpha={alpha[i]:.6f}, "

            f"beta={beta[i]:.6f}"

        )




if __name__=="__main__":


    test()
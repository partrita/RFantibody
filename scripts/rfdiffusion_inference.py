import glob
import argparse
import sys
import os
import time
import pickle
import re
import random

# Try-except imports for core dependencies
try:
    from omegaconf import OmegaConf
    # print("DEBUG_IMPORT: OmegaConf OK") # Keep these commented out for cleaner final script
except ImportError as e:
    print(f"CRITICAL_IMPORT_ERROR: OmegaConf: {e}")
    sys.exit(1)

try:
    from hydra.core.hydra_config import HydraConfig
    import hydra
    # print("DEBUG_IMPORT: Hydra OK")
except ImportError as e:
    print(f"CRITICAL_IMPORT_ERROR: Hydra: {e}")
    sys.exit(1)

import logging # Standard library

try:
    import numpy as np
    # print("DEBUG_IMPORT: NumPy OK")
except ImportError as e:
    print(f"CRITICAL_IMPORT_ERROR: NumPy: {e}")
    sys.exit(1)


# Global placeholders for heavy dependencies
torch = None
writepdb_multi, writepdb, generate_Cbeta = None, None, None
ab_write_pdblines = None
num2aa = None
Quiver = None
model_runners = None

def _load_heavy_dependencies():
    global torch, writepdb_multi, writepdb, generate_Cbeta, ab_write_pdblines, num2aa, Quiver, model_runners

    import torch as actual_torch
    torch = actual_torch

    from rfantibody.rfdiffusion.util import writepdb_multi as wpm, writepdb as wp, generate_Cbeta as gcbeta
    writepdb_multi, writepdb, generate_Cbeta = wpm, wp, gcbeta

    from rfantibody.util.io import ab_write_pdblines as awp
    ab_write_pdblines = awp

    from rfantibody.rfdiffusion.chemical import num2aa as n2a
    num2aa = n2a

    from rfantibody.util.quiver import Quiver as Qv
    Quiver = Qv

    from rfantibody.rfdiffusion.inference import model_runners as mr
    model_runners = mr

conversion = "ARNDCQEGHILKMFPSTWYV-"

def make_deterministic(seed=0):
    # Ensure torch is loaded before using it
    if torch is None: _load_heavy_dependencies() # Should ideally not happen if dry_run is false
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="config/inference", config_name="base")
def main_hydra(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)

    # These prints are for the older --test-dry-run, will be superseded by --test-argparse-only for CLI part
    print(f"TEST_CONF_OUTPUT_PREFIX: {conf.inference.output_prefix}")
    print(f"TEST_CONF_NUM_DESIGNS: {conf.inference.num_designs}")

    if conf.inference.get("test_dry_run", False):
        print("TEST_DRY_RUN_ACTIVE: Exiting early for test purposes.")
        sys.exit(0)

    # Load heavy dependencies only if not a dry run (already handled by exit above)
    _load_heavy_dependencies() # This ensures they are loaded for actual runs

    if conf.inference.deterministic:
        make_deterministic()

    sampler = model_runners.AbSampler(conf)
    design_startnum = conf.inference.design_startnum
    quiver, quiver_xt, quiver_px0, tags = None, None, None, []

    if conf.inference.quiver is not None:
        quiver = Quiver(conf.inference.quiver, mode="w")
        tags = quiver.get_tags()
        if conf.inference.write_trajectory:
            quiver_base = os.path.splitext(conf.inference.quiver)[0]
            quiver_xt = Quiver(quiver_base + "_Xt-1_traj.qv", mode="w")
            quiver_px0 = Quiver(quiver_base + "_pX0_traj.qv", mode="w")

    if conf.inference.quiver is not None:
        if conf.inference.design_startnum == -1:
            if len(tags) == 0:
                design_startnum = 0
            else:
                indices = [-1]
                for tag in tags:
                    m = re.match(".*_(\d+)$", tag)
                    if not m: continue
                    m = m.groups()[0]
                    indices.append(int(m))
                design_startnum = max(indices) + 1
            conf.inference.design_startnum = design_startnum
    elif conf.inference.design_startnum == -1:
        existing = glob.glob(conf.inference.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            m = re.match(".*_(\d+)\.pdb$", e)
            if not m: continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1
        conf.inference.design_startnum = design_startnum

    for i_des in range(design_startnum, design_startnum + conf.inference.num_designs):
        if conf.inference.deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f"{conf.inference.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix}")
        if conf.inference.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists.")
            continue
        if conf.inference.quiver is not None:
            if out_prefix in tags:
                log.info(f"Skipping this design because tag {out_prefix} already exists.")
                continue

        failed = 0
        while True:
            x_init, seq_init = sampler.sample_init()
            denoised_xyz_stack = []
            px0_xyz_stack = []
            seq_stack = []
            chi1_stack = []
            plddt_stack = []
            x_t = torch.clone(x_init)
            seq_t = torch.clone(seq_init)
            for t in range(int(sampler.t_step_input), conf.inference.final_step - 1, -1):
                px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
                    t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_init, final_step=conf.inference.final_step,
                )
                px0_xyz_stack.append(px0)
                denoised_xyz_stack.append(x_t)
                seq_stack.append(seq_t)
                chi1_stack.append(tors_t[:, :])
                plddt_stack.append(plddt[0])
                # print("Sequence of Hotspot Residues:", "".join(conversion[i] for i in torch.argmax(seq_t, dim=1)[sampler.ab_item.hotspots]))
                if conf.antibody.terminate_bad_targeting is not None:
                    Cb = generate_Cbeta(N=px0[:, 0], Ca=px0[:, 1], C=px0[:, 2])
                    dist = torch.cdist(Cb[sampler.ab_item.hotspots], Cb[sampler.ab_item.loop_mask])
                    mindist = torch.min(dist, dim=1).values
                    overallmin = torch.min(mindist)
                    # print(f"Overall min distance hotspot to designed loop: {overallmin}")
                    if (conf.antibody.terminate_bad_targeting == t and overallmin > conf.antibody.hotspot_termination_threshold):
                        # print("Not targeting correctly")
                        failed += 1
                        if (failed >= conf.antibody.hotspot_termination_failures_permitted):
                            sys.exit("This set of inputs is not efficiently targeting the hotspots")
                        continue
            break
        denoised_xyz_stack = torch.stack(denoised_xyz_stack); denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
        px0_xyz_stack = torch.stack(px0_xyz_stack); px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])
        plddt_stack = torch.stack(plddt_stack)

        if conf.inference.quiver is None:
            os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1]
        if conf.seq_diffuser.seqdiff is not None:
            final_seq = final_seq[:, :20]
        final_seq = torch.where(torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1))
        bfacts = torch.ones_like(final_seq.squeeze())
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        out = f"{out_prefix}.pdb"
        trb = dict(
            config=OmegaConf.to_container(conf, resolve=True),
            plddt=plddt_stack.cpu().numpy(),
            device=torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
            time=time.time() - start_time,
        )
        if sampler.ab_design():
            for loop in sampler.loop_map: trb[f"{loop.upper()}_len"] = len(sampler.loop_map[loop])
        if (sampler.ab_design() and torch.any(sampler.ab_item.target_mask) and torch.any(sampler.ab_item.hotspots)):
            Cb = generate_Cbeta(N=denoised_xyz_stack[0, :, 0], Ca=denoised_xyz_stack[0, :, 1], C=denoised_xyz_stack[0, :, 2])
            dist = torch.cdist(Cb[sampler.ab_item.hotspots], Cb[sampler.ab_item.loop_mask])
            mindist = torch.min(dist, dim=1).values
            overallmin = torch.min(mindist); averagemin = torch.mean(mindist)
            # print(f"Overall min distance hotspot to designed loop: {overallmin}"); print(f"Average min distance hotspot to designed loop: {averagemin}")
            trb["mindist"] = overallmin.cpu().numpy(); trb["averagemin"] = averagemin.cpu().numpy()
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items(): trb[key] = value
        if conf.inference.quiver is None:
            with open(f"{out_prefix}.trb", "wb") as f_out: pickle.dump(trb, f_out)
        if sampler.ab_design():
            bfacts[sampler.ab_item.hotspots] = 0
            if conf.inference.quiver is None:
                pdblines = ab_write_pdblines(atoms=denoised_xyz_stack[0, :, :4].cpu().numpy(), seq=final_seq.cpu().numpy(), chain_idx=sampler.chain_idx, bfacts=bfacts.cpu().numpy(), loop_map=sampler.loop_map, num2aa=num2aa)
                with open(out, "w") as f_out: f_out.write("\n".join(line.rstrip() for line in pdblines))
            else:
                pdblines = ab_write_pdblines(atoms=denoised_xyz_stack[0, :, :4].cpu().numpy(), seq=final_seq.cpu().numpy(), chain_idx=sampler.chain_idx, bfacts=bfacts.cpu().numpy(), loop_map=sampler.loop_map, num2aa=num2aa)
                outtag = out_prefix.replace("/", "_")
                if torch.any(sampler.ab_item.target_mask) and torch.any(sampler.ab_item.hotspots):
                    scoreline = f"mindist={float(overallmin):.2f}|averagemin={float(averagemin):.2f}"
                    if quiver: quiver.add_pdb(pdblines, outtag, scoreline)
                else:
                    if quiver: quiver.add_pdb(pdblines, outtag)
        else:
            writepdb(out, denoised_xyz_stack[0, :, :4], final_seq, sampler.binderlen, chain_idx=sampler.chain_idx, bfacts=bfacts)
        if conf.inference.write_trajectory:
            outtag = out_prefix.replace("/", "_") # Define outtag here for both quiver and non-quiver
            if conf.inference.quiver is not None:
                xtpdblines = writepdb_multi(None, denoised_xyz_stack, bfacts, final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx, return_pdblines=True)
                px0pdblines = writepdb_multi(None, px0_xyz_stack, bfacts, final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx, return_pdblines=True)
                if quiver_xt: quiver_xt.add_pdb(xtpdblines, f"{outtag}_Xt-1")
                if quiver_px0: quiver_px0.add_pdb(px0pdblines, f"{outtag}_pX0")
            else:
                traj_prefix = (os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix))
                os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)
                out_xt = f"{traj_prefix}_Xt-1_traj.pdb"; out_px0 = f"{traj_prefix}_pX0_traj.pdb"
                writepdb_multi(out_xt, denoised_xyz_stack, bfacts, final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx)
                writepdb_multi(out_px0, px0_xyz_stack, bfacts, final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx)
        log.info(f"Finished design in {(time.time() - start_time) / 60:.2f} minutes")

def cli_main() -> None:
    parser = argparse.ArgumentParser(description="RFDiffusion Inference Script")
    parser.add_argument('--config-name', type=str, help='Hydra config name (e.g., base). This is the primary Hydra config.')
    parser.add_argument('--output_prefix', type=str, help='Override inference.output_prefix.')
    parser.add_argument('--num_designs', type=int, help='Override inference.num_designs.')
    parser.add_argument('--design_startnum', type=int, help='Override inference.design_startnum.')
    parser.add_argument('--quiver', type=str, help='Override inference.quiver.')
    parser.add_argument('--write_trajectory', action=argparse.BooleanOptionalAction, help='Override inference.write_trajectory.')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, help='Override inference.deterministic.')
    parser.add_argument('--cautious', action=argparse.BooleanOptionalAction, help='Override inference.cautious.')
    parser.add_argument('--test-dry-run', action='store_true', help='Exit after printing config for testing purposes.')
    parser.add_argument('--test-argparse-only', action='store_true', help='Print parsed argparse args and exit.')

    args, unknown_hydra_args = parser.parse_known_args()

    if args.test_argparse_only:
        print(f"ARGPARSE_CONFIG_NAME:{args.config_name}")
        print(f"ARGPARSE_OUTPUT_PREFIX:{args.output_prefix}")
        print(f"ARGPARSE_NUM_DESIGNS:{args.num_designs}")
        print(f"ARGPARSE_DESIGN_STARTNUM:{args.design_startnum}")
        print(f"ARGPARSE_QUIVER:{args.quiver}")
        print(f"ARGPARSE_WRITE_TRAJECTORY:{args.write_trajectory}")
        print(f"ARGPARSE_DETERMINISTIC:{args.deterministic}")
        print(f"ARGPARSE_CAUTIOUS:{args.cautious}")
        print(f"ARGPARSE_TEST_DRY_RUN:{args.test_dry_run}")
        print(f"ARGPARSE_UNKNOWN_ARGS:{unknown_hydra_args}")
        sys.exit(0)

    hydra_overrides = []
    if args.config_name is None: args.config_name = 'base' # Default config name

    if args.output_prefix is not None: hydra_overrides.append(f"inference.output_prefix={args.output_prefix}")
    if args.num_designs is not None: hydra_overrides.append(f"inference.num_designs={args.num_designs}")
    if args.design_startnum is not None: hydra_overrides.append(f"inference.design_startnum={args.design_startnum}")
    if args.quiver is not None: hydra_overrides.append(f"inference.quiver={args.quiver}")
    if args.write_trajectory is not None: hydra_overrides.append(f"inference.write_trajectory={args.write_trajectory}")
    if args.deterministic is not None: hydra_overrides.append(f"inference.deterministic={args.deterministic}")
    if args.cautious is not None: hydra_overrides.append(f"inference.cautious={args.cautious}")
    if args.test_dry_run: hydra_overrides.append(f"inference.test_dry_run={args.test_dry_run}")

    original_sys_argv = sys.argv.copy()
    new_sys_argv = [original_sys_argv[0], f"--config-name={args.config_name}"] + hydra_overrides + unknown_hydra_args
    sys.argv = new_sys_argv

    main_hydra()

if __name__ == "__main__":
    cli_main()

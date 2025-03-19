"""
This script converts antibody PDB files from Chothia format to HLT format.
HLT format requirements:
- Heavy chain is renamed to chain H
- Light chain is renamed to chain L
- Target chain(s) are renamed to chain T
- Chains are ordered as Heavy, Light, then Target
- CDR loops are annotated with REMARK statements at the end of the file
"""

import argparse
import numpy as np

from biotite.structure.io.pdb import PDBFile
from biotite.structure import array
from biotite.structure import residue_iter

protein_residues = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


def parse_args():
    """Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - input_pdb: Path to input PDB file
            - heavy: Heavy chain ID in input file
            - light: Light chain ID in input file
            - target: Comma-separated list of target chain IDs
            - output: Optional output file path
    """
    parser = argparse.ArgumentParser(
        description="Convert Chothia-formatted PDB to HLT format"
    )
    parser.add_argument("--inpdb", "-i", help="Input PDB file in Chothia format")
    parser.add_argument("--heavy", "-H", help="Heavy chain ID")
    parser.add_argument("--light", "-L", help="Light chain ID")
    parser.add_argument("--target", "-T", help="Target chain ID(s), comma-separated")
    parser.add_argument("--output", "-o", help="Output HLT file path")
    parser.add_argument(
        "--whole_fab", "-w", action="store_true", help="Keep entire Fab region"
    )
    parser.add_argument(
        "--Hcrop",
        default=115,
        help="Chothia residue number to crop to for heavy chain a "
        + "reasonable number is between 105 and 115",
    )
    parser.add_argument(
        "--Lcrop",
        default=110,
        help="Chothia residue number to crop to for light chain a "
        + "reasonable number is between 100 and 110",
    )

    args = parser.parse_args()

    if not (args.heavy or args.light):
        raise ValueError("Either heavy or light chain must be specified")

    return args


def get_Fv_ranges():
    """Define the residue ranges for each Fv loop according to Chothia numbering scheme.

    Returns:
        dict: Dictionary mapping Fv names to their residue ranges (start, end) inclusive
    """

    return {"H": (1, 102), "L": (1, 97)}


def get_cdr_ranges():
    """Define the residue ranges for each CDR loop according to Chothia numbering scheme.

    The Chothia numbering scheme is a standardized way to number antibody residues,
    making it possible to identify CDR loops based on residue numbers.

    Returns:
        dict: Dictionary mapping CDR names to their residue ranges (start, end) inclusive
    """
    return {
        "H": {
            "H1": (26, 32),  # Heavy chain CDR1: residues 26-32
            "H2": (52, 56),  # Heavy chain CDR2: residues 52-56
            "H3": (95, 102),  # Heavy chain CDR3: residues 95-102
        },
        "L": {
            "L1": (24, 34),  # Light chain CDR1: residues 24-34
            "L2": (50, 56),  # Light chain CDR2: residues 50-56
            "L3": (89, 97),  # Light chain CDR3: residues 89-97
        },
    }


def convert_to_hlt(
    input_pdb,
    heavy_chain,
    light_chain,
    target_chains,
    whole_fab,
    Hcrop,
    Lcrop,
):
    pdb_file = PDBFile.read(input_pdb)
    structure = pdb_file.get_structure(model=1)

    protein_atom_list = [
        atom for atom in structure if atom.res_name in protein_residues
    ]
    structure = array(protein_atom_list)

    chain_mapping = {heavy_chain: "H", light_chain: "L"}
    for t in target_chains:
        chain_mapping[t] = "T"

    cdr_residues = {"H1": [], "H2": [], "H3": [], "L1": [], "L2": [], "L3": []}

    cdr_ranges = get_cdr_ranges()
    hl_structure = []
    t_structure = []
    processed_residues = set()

    for chain_id, new_chain_id in chain_mapping.items():
        chain_mask = structure.chain_id == chain_id
        if not np.any(chain_mask):
            continue

        atoms = structure[chain_mask]
        atoms.chain_id = np.full(len(atoms), new_chain_id)

        if new_chain_id in ["H", "L"]:
            curr_ranges = cdr_ranges[new_chain_id]
            for residue in residue_iter(atoms):
                res_id = residue.res_id[0]
                if (new_chain_id, res_id) in processed_residues:
                    continue

                if not whole_fab:
                    if new_chain_id == "H" and res_id > Hcrop:
                        continue
                    elif new_chain_id == "L" and res_id > Lcrop:
                        continue

                for cdr, (start, end) in curr_ranges.items():
                    if start <= res_id <= end:
                        cdr_residues[cdr].append(res_id)

                hl_structure.extend(residue)
                processed_residues.add((new_chain_id, res_id))
        else:  # Target chain
            t_structure.extend(atoms)

    return array(hl_structure), array(t_structure), cdr_residues


def main():
    """
    Main function to run the conversion process
    """
    # Parse command line arguments
    args = parse_args()
    target_chains = args.target.split(",") if args.target else []

    output_base = args.output or args.inpdb.replace(".pdb", "")
    output_hl = f"{output_base}_HL.pdb"
    output_t = f"{output_base}_T.pdb"

    hl_structure, t_structure, cdr_residues = convert_to_hlt(
        args.inpdb,
        args.heavy,
        args.light,
        target_chains,
        args.whole_fab,
        args.Hcrop,
        args.Lcrop,
    )

    # Write HL structure
    pdb_file_hl = PDBFile()
    pdb_file_hl.set_structure(hl_structure)
    with open(output_hl, "w") as f:
        pdb_file_hl.write(f)
        for cdr in sorted(cdr_residues.keys()):
            for res_num in sorted(cdr_residues[cdr]):
                f.write(f"REMARK PDBinfo-LABEL: {res_num:4d} {cdr}\n")

    # Write T structure
    if len(t_structure) > 0:
        pdb_file_t = PDBFile()
        pdb_file_t.set_structure(t_structure)
        with open(output_t, "w") as f:
            pdb_file_t.write(f)

    print(f"HL chains saved to: {output_hl}")
    if len(t_structure) > 0:
        print(f"T chain saved to: {output_t}")
    else:
        print("No T chain found in the structure.")


if __name__ == "__main__":
    main()

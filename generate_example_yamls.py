"""
generate_example_yamls.py
=========================
Fetch real PDB sequences from RCSB and create YAML input files
for the diverse-input benchmarking suite.

Usage:
    python generate_example_yamls.py

Creates (or skips if already present):
  examples/protein_medium.yaml    — 6LU7-A  SARS-CoV-2 Mpro         (~306 aa)
  examples/protein_large.yaml     — 1XCK-A  E. coli GroEL subunit    (~524 aa)
  examples/protein_rna.yaml       — 4WZJ    Csy4 endonuclease + 16nt crRNA
  examples/protein_ligand.yaml    — 1HSG    HIV-1 protease dimer + indinavir (IDV)
  examples/antibody.yaml          — 1MLC    D1.3 anti-lysozyme Fv (VH + VL)
  examples/homodimer.yaml         — 1TIM    Triosephosphate isomerase homodimer
  examples/large_complex.yaml     — 1CTS    Pig citrate synthase homodimer (~874 aa)
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXAMPLES_DIR = Path(__file__).parent / "examples"

# ---------------------------------------------------------------------------
# RCSB fetcher
# ---------------------------------------------------------------------------

def fetch_fasta(pdb_id: str) -> Dict[str, str]:
    """Download FASTA from RCSB and return {chain_id: sequence}."""
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}/download"
    print(f"  Fetching {url} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "boltz-profiler/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"FAILED: {exc}")
        return {}
    print("OK")

    seqs: Dict[str, str] = {}
    cur_chain: Optional[str] = None
    cur_seq: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            # Save previous
            if cur_chain:
                seqs[cur_chain] = "".join(cur_seq)
            # Parse chain from header like
            #   >6LU7_1|Chain A|...|Homo sapiens
            #   >4WZJ_2|Chains A[auth A],B[auth B]|...
            # The auth chain label is what we need.
            # We parse "Chain X" or "Chains X[auth Y]" → use auth if present, else label
            chain_part = line.split("|")[1] if "|" in line else ""
            # Try "auth X" form
            m_auth = re.search(r"\[auth\s+([A-Za-z0-9])\]", chain_part)
            if m_auth:
                cur_chain = m_auth.group(1)
            else:
                # "Chain X" or "Chains X,Y"
                m_plain = re.search(r"Chain[s]?\s+([A-Za-z0-9])", chain_part)
                cur_chain = m_plain.group(1) if m_plain else None
            cur_seq = []
        else:
            cur_seq.append(line)

    if cur_chain:
        seqs[cur_chain] = "".join(cur_seq)

    return seqs


def write_yaml(path: Path, content: str, force: bool = False) -> None:
    if path.exists() and not force:
        print(f"  Exists, skipping: {path.name}")
        return
    path.write_text(content)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Individual YAML generators
# ---------------------------------------------------------------------------

def make_protein_medium(force: bool = False) -> None:
    """6LU7 chain A — SARS-CoV-2 Mpro (~306 residues)."""
    print("protein_medium.yaml (6LU7-A, SARS-CoV-2 Mpro):")
    seqs = fetch_fasta("6LU7")
    if "A" not in seqs:
        print(f"  WARNING: chain A not found. Available: {list(seqs)}")
        return
    seq = seqs["A"]
    print(f"  sequence length: {len(seq)}")
    content = f"""\
# SARS-CoV-2 main protease (Mpro / 3CLpro / Nsp5)
# PDB: 6LU7, chain A — {len(seq)} residues
version: 1
sequences:
  - protein:
      id: A
      sequence: {seq}
"""
    write_yaml(EXAMPLES_DIR / "protein_medium.yaml", content, force=force)


def make_protein_large(force: bool = False) -> None:
    """1XCK chain A — E. coli GroEL subunit (~524 residues)."""
    print("protein_large.yaml (1XCK-A, GroEL subunit):")
    seqs = fetch_fasta("1XCK")
    if "A" not in seqs:
        print(f"  WARNING: chain A not found. Available: {list(seqs)}")
        return
    seq = seqs["A"]
    print(f"  sequence length: {len(seq)}")
    content = f"""\
# E. coli GroEL chaperonin subunit (monomer)
# PDB: 1XCK, chain A — {len(seq)} residues
version: 1
sequences:
  - protein:
      id: A
      sequence: {seq}
"""
    write_yaml(EXAMPLES_DIR / "protein_large.yaml", content, force=force)


def make_protein_rna(force: bool = False) -> None:
    """4WZJ — Csy4 endonuclease + 16 nt crRNA hairpin."""
    print("protein_rna.yaml (4WZJ, Csy4 + crRNA):")
    seqs = fetch_fasta("4WZJ")

    # Identify protein chain (largest sequence) and RNA chain (shortest)
    protein_chains = {}
    rna_chains = {}
    for chain, seq in seqs.items():
        # RNA chains typically contain only A, U, G, C, T (no amino acid letters)
        # Rough heuristic: if all chars are in AUGCT+X, it's nucleic acid
        non_nt = set(seq.upper()) - set("AUGCTXNI-")
        if not non_nt and len(seq) <= 50:
            rna_chains[chain] = seq
        else:
            protein_chains[chain] = seq

    if not protein_chains:
        # Fallback: largest = protein
        longest = max(seqs, key=lambda c: len(seqs[c]))
        protein_chains = {longest: seqs[longest]}

    if not rna_chains:
        # Fallback: shortest remaining
        remaining = {c: s for c, s in seqs.items() if c not in protein_chains}
        if remaining:
            shortest = min(remaining, key=lambda c: len(remaining[c]))
            rna_chains = {shortest: remaining[shortest]}

    if not protein_chains or not rna_chains:
        print(f"  WARNING: Could not identify protein and RNA chains. Available: {list(seqs)}")
        return

    prot_chain = sorted(protein_chains.keys())[0]
    prot_seq = protein_chains[prot_chain]
    rna_chain = sorted(rna_chains.keys())[0]
    rna_seq = rna_chains[rna_chain].upper().replace("T", "U")  # DNA T → RNA U

    print(f"  protein chain {prot_chain}: {len(prot_seq)} aa")
    print(f"  RNA chain {rna_chain}: {len(rna_seq)} nt")

    content = f"""\
# Csy4 CRISPR endonuclease + 16 nt crRNA hairpin
# PDB: 4WZJ
# Protein chain {prot_chain}: {len(prot_seq)} aa  |  RNA chain {rna_chain}: {len(rna_seq)} nt
version: 1
sequences:
  - protein:
      id: {prot_chain}
      sequence: {prot_seq}
  - rna:
      id: {rna_chain}
      sequence: {rna_seq}
"""
    write_yaml(EXAMPLES_DIR / "protein_rna.yaml", content, force=force)


def make_protein_ligand(force: bool = False) -> None:
    """1HSG — HIV-1 protease dimer (chains A+B) + indinavir (CCD: IDV)."""
    print("protein_ligand.yaml (1HSG, HIV-1 protease + IDV):")
    seqs = fetch_fasta("1HSG")

    prot_chains = [(c, s) for c, s in seqs.items() if len(s) > 50]
    prot_chains.sort(key=lambda x: x[0])  # sort by chain name

    if not prot_chains:
        print(f"  WARNING: no protein chains found. Available: {list(seqs)}")
        return

    print(f"  protein chains: {[(c, len(s)) for c, s in prot_chains]}")

    # Use first two protein chains (A and B for HIV protease homodimer)
    chains_yaml = ""
    for chain_id, seq in prot_chains[:2]:
        chains_yaml += f"  - protein:\n      id: {chain_id}\n      sequence: {seq}\n"

    # Indinavir ligand — chain C in 1HSG
    content = f"""\
# HIV-1 protease homodimer + indinavir
# PDB: 1HSG  |  protein chains: {'+'.join(c for c,_ in prot_chains[:2])} (each ~99 aa)
# Ligand: indinavir (CCD code IDV)
version: 1
sequences:
{chains_yaml}  - ligand:
      id: C
      ccd: IDV
"""
    write_yaml(EXAMPLES_DIR / "protein_ligand.yaml", content, force=force)


def make_antibody(force: bool = False) -> None:
    """1MLC — D1.3 anti-lysozyme Fv (VH chain H + VL chain L)."""
    print("antibody.yaml (1MLC, D1.3 Fv):")
    seqs = fetch_fasta("1MLC")

    # 1MLC has chains A (lysozyme antigen), H (VH), L (VL)
    # We want only VH + VL (Fv only, no antigen)
    vh_seq = seqs.get("H", seqs.get("h", ""))
    vl_seq = seqs.get("L", seqs.get("l", ""))

    if not vh_seq or not vl_seq:
        # Fall back to the two protein chains
        prot = [(c, s) for c, s in seqs.items() if len(s) > 50]
        prot.sort(key=lambda x: len(x[1]))
        if len(prot) >= 2:
            # Shorter two chains = VH + VL (lysozyme is ~129 aa, VH ~124, VL ~110)
            # Actually pick the two non-lysozyme chains if we can identify chain A
            non_a = [(c, s) for c, s in prot if c != "A"]
            if len(non_a) >= 2:
                vl_seq = non_a[0][1]   # shorter = VL
                vh_seq = non_a[1][1]   # longer  = VH
            else:
                vh_seq = prot[-1][1]
                vl_seq = prot[-2][1]
        if not vh_seq:
            print(f"  WARNING: VH/VL chains not found. Available: {list(seqs)}")
            return

    print(f"  VH (H): {len(vh_seq)} aa")
    print(f"  VL (L): {len(vl_seq)} aa")
    content = f"""\
# D1.3 anti-hen lysozyme Fv antibody fragment
# PDB: 1MLC  |  VH chain H ({len(vh_seq)} aa) + VL chain L ({len(vl_seq)} aa)
# Antigen (lysozyme chain A) intentionally excluded — Fv only
version: 1
sequences:
  - protein:
      id: H
      sequence: {vh_seq}
  - protein:
      id: L
      sequence: {vl_seq}
"""
    write_yaml(EXAMPLES_DIR / "antibody.yaml", content, force=force)


def make_homodimer(force: bool = False) -> None:
    """1TIM — Triosephosphate isomerase homodimer (2 × ~249 aa = ~498 aa)."""
    print("homodimer.yaml (1TIM, TIM homodimer):")
    seqs = fetch_fasta("1TIM")

    prot = [(c, s) for c, s in seqs.items() if len(s) > 50]
    prot.sort(key=lambda x: x[0])

    if not prot:
        print(f"  WARNING: no protein chains. Available: {list(seqs)}")
        return

    # TIM homodimer: chains A and B are identical ~249 aa
    chain_a = prot[0]
    chain_b = prot[1] if len(prot) > 1 else prot[0]
    print(f"  chain {chain_a[0]}: {len(chain_a[1])} aa")
    print(f"  chain {chain_b[0]}: {len(chain_b[1])} aa")
    content = f"""\
# Triosephosphate isomerase (TIM) homodimer
# PDB: 1TIM  |  chain A ({len(chain_a[1])} aa) + chain B ({len(chain_b[1])} aa)
version: 1
sequences:
  - protein:
      id: A
      sequence: {chain_a[1]}
  - protein:
      id: B
      sequence: {chain_b[1]}
"""
    write_yaml(EXAMPLES_DIR / "homodimer.yaml", content, force=force)


def make_large_complex(force: bool = False) -> None:
    """1CTS — Pig heart citrate synthase homodimer (~874 aa total)."""
    print("large_complex.yaml (1CTS, citrate synthase homodimer):")
    seqs = fetch_fasta("1CTS")

    prot = [(c, s) for c, s in seqs.items() if len(s) > 100]
    prot.sort(key=lambda x: x[0])

    if not prot:
        print(f"  WARNING: no large protein chains. Available: {list(seqs)}")
        return

    chains_yaml = ""
    total = 0
    for chain_id, seq in prot[:2]:  # homodimer → 2 chains
        chains_yaml += f"  - protein:\n      id: {chain_id}\n      sequence: {seq}\n"
        total += len(seq)
        print(f"  chain {chain_id}: {len(seq)} aa")

    print(f"  total: {total} aa")
    content = f"""\
# Pig heart citrate synthase homodimer
# PDB: 1CTS  |  total ~{total} residues
# Tests whether OPM abs_max outliers scale with sequence length.
version: 1
sequences:
{chains_yaml}"""
    write_yaml(EXAMPLES_DIR / "large_complex.yaml", content, force=force)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Generate example YAML input files from RCSB.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing YAML files.")
    args = p.parse_args()

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    generators = [
        make_protein_medium,
        make_protein_large,
        make_protein_rna,
        make_protein_ligand,
        make_antibody,
        make_homodimer,
        make_large_complex,
    ]
    for gen in generators:
        print()
        try:
            gen(force=args.force)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print("\nDone. YAML files written to examples/")


if __name__ == "__main__":
    main()

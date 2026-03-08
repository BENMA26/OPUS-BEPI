"""
Prepare protein-protein binding site dataset for pre-training.

This script helps extract protein-protein interfaces from PDB complexes
to create a pre-training dataset similar to ScanNet's Dockground dataset.

Data sources:
1. Dockground: https://dockground.compbio.ku.edu/
2. PDB complexes with multiple chains
3. DB5.5 / SKEMPI datasets

Output format: Same as BCE_633 (train.pkl, test.pkl, cross-validation.npy)

Usage:
    python prepare_pretrain_dataset.py \
        --input_dir ./data/pdb_complexes \
        --output_dir ./data/Dockground_5K \
        --distance_cutoff 6.0 \
        --min_interface_size 5

Requirements:
    - PDB files with multiple chains
    - BioPython for structure parsing
    - Same preprocessing pipeline as BCE_633
"""

import os
import pickle as pk
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser, NeighborSearch
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import your existing preprocessing utilities
# Adjust these imports based on your actual code structure
try:
    from utils import Protein  # Your Protein class
    from preprocess import extract_features  # Your feature extraction
except ImportError:
    print("Warning: Could not import preprocessing utilities.")
    print("Make sure utils.py and preprocess.py are available.")


class ProteinInterfaceExtractor:
    """Extract protein-protein interfaces from PDB complexes."""

    def __init__(self, distance_cutoff=6.0, min_interface_size=5):
        """
        Args:
            distance_cutoff: Distance threshold (Å) for interface residues
            min_interface_size: Minimum number of interface residues per chain
        """
        self.distance_cutoff = distance_cutoff
        self.min_interface_size = min_interface_size
        self.parser = PDBParser(QUIET=True)

    def extract_interface(self, pdb_file):
        """
        Extract interface residues from a PDB complex.

        Returns:
            List of (chain_id, interface_residues) tuples
            interface_residues: list of residue indices (0-based)
        """
        structure = self.parser.get_structure('complex', pdb_file)
        model = structure[0]

        # Get all chains
        chains = list(model.get_chains())
        if len(chains) < 2:
            return []

        interfaces = []

        # For each chain pair, find interface residues
        for i, chain_a in enumerate(chains):
            for chain_b in chains[i+1:]:
                interface_a, interface_b = self._find_interface_residues(
                    chain_a, chain_b
                )

                if (len(interface_a) >= self.min_interface_size and
                    len(interface_b) >= self.min_interface_size):
                    interfaces.append((chain_a.id, interface_a))
                    interfaces.append((chain_b.id, interface_b))

        return interfaces

    def _find_interface_residues(self, chain_a, chain_b):
        """Find interface residues between two chains."""
        # Get all atoms from both chains
        atoms_a = [atom for residue in chain_a for atom in residue.get_atoms()]
        atoms_b = [atom for residue in chain_b for atom in residue.get_atoms()]

        # Build neighbor search for chain B
        ns = NeighborSearch(atoms_b)

        # Find interface residues in chain A
        interface_a = set()
        for atom in atoms_a:
            neighbors = ns.search(atom.coord, self.distance_cutoff)
            if neighbors:
                interface_a.add(atom.get_parent().id[1])  # residue number

        # Find interface residues in chain B
        ns = NeighborSearch(atoms_a)
        interface_b = set()
        for atom in atoms_b:
            neighbors = ns.search(atom.coord, self.distance_cutoff)
            if neighbors:
                interface_b.add(atom.get_parent().id[1])

        return sorted(interface_a), sorted(interface_b)


def process_pdb_complex(pdb_file, extractor, output_name=None):
    """
    Process a single PDB complex and create Protein objects.

    Args:
        pdb_file: Path to PDB file
        extractor: ProteinInterfaceExtractor instance
        output_name: Optional name for the protein (default: filename)

    Returns:
        List of Protein objects with interface labels
    """
    if output_name is None:
        output_name = Path(pdb_file).stem

    # Extract interfaces
    interfaces = extractor.extract_interface(pdb_file)
    if not interfaces:
        return []

    # Group by chain
    chain_interfaces = defaultdict(list)
    for chain_id, residues in interfaces:
        chain_interfaces[chain_id].extend(residues)

    proteins = []

    # Create Protein object for each chain
    for chain_id, interface_residues in chain_interfaces.items():
        try:
            # TODO: Adapt this to your Protein class initialization
            # This is a placeholder - adjust based on your actual code
            protein = Protein(name=f"{output_name}_{chain_id}")

            # Load structure and sequence
            # protein.load_structure(pdb_file, chain_id)

            # Create binary labels (1 for interface, 0 for non-interface)
            # Assuming protein has a sequence length
            # labels = np.zeros(len(protein.sequence))
            # labels[interface_residues] = 1
            # protein.label = labels

            # Extract features (ESM, DSSP, etc.)
            # protein.extract_features()

            proteins.append(protein)

        except Exception as e:
            print(f"Error processing {pdb_file} chain {chain_id}: {e}")
            continue

    return proteins


def prepare_dataset(input_dir, output_dir, distance_cutoff=6.0,
                   min_interface_size=5, test_ratio=0.1):
    """
    Prepare pre-training dataset from PDB complexes.

    Args:
        input_dir: Directory containing PDB files
        output_dir: Output directory for processed dataset
        distance_cutoff: Interface distance threshold (Å)
        min_interface_size: Minimum interface residues
        test_ratio: Fraction of data for test set
    """
    os.makedirs(output_dir, exist_ok=True)

    extractor = ProteinInterfaceExtractor(distance_cutoff, min_interface_size)

    # Find all PDB files
    pdb_files = list(Path(input_dir).glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files")

    all_proteins = []

    # Process each PDB file
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        proteins = process_pdb_complex(pdb_file, extractor)
        all_proteins.extend(proteins)

    print(f"Extracted {len(all_proteins)} protein chains with interfaces")

    # Split into train/test
    n_test = int(len(all_proteins) * test_ratio)
    indices = np.random.permutation(len(all_proteins))

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    trainset = [all_proteins[i] for i in train_indices]
    testset = [all_proteins[i] for i in test_indices]

    # Save datasets
    with open(f'{output_dir}/train.pkl', 'wb') as f:
        pk.dump(trainset, f)

    with open(f'{output_dir}/test.pkl', 'wb') as f:
        pk.dump(testset, f)

    # Create cross-validation indices (for 10-fold CV)
    cv_indices = np.random.permutation(len(trainset))
    np.save(f'{output_dir}/cross-validation.npy', cv_indices)

    print(f"\nDataset saved to {output_dir}")
    print(f"  Train: {len(trainset)} proteins")
    print(f"  Test:  {len(testset)} proteins")
    print(f"  Cross-validation indices: {len(cv_indices)}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare protein-protein binding site dataset for pre-training"
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing PDB complex files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed dataset')
    parser.add_argument('--distance_cutoff', type=float, default=6.0,
                       help='Distance cutoff (Å) for interface residues (default: 6.0)')
    parser.add_argument('--min_interface_size', type=int, default=5,
                       help='Minimum number of interface residues (default: 5)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Fraction of data for test set (default: 0.1)')
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random seed (default: 2022)')

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 80)
    print("Preparing Protein-Protein Binding Site Dataset")
    print("=" * 80)
    print(f"Input directory:     {args.input_dir}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Distance cutoff:     {args.distance_cutoff} Å")
    print(f"Min interface size:  {args.min_interface_size} residues")
    print(f"Test ratio:          {args.test_ratio}")
    print("=" * 80)

    prepare_dataset(
        args.input_dir,
        args.output_dir,
        args.distance_cutoff,
        args.min_interface_size,
        args.test_ratio
    )

    print("\n" + "=" * 80)
    print("Dataset preparation completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Verify the dataset:")
    print(f"   ls -lh {args.output_dir}")
    print("2. Run pre-training:")
    print("   sbatch run_pretrain_stage.sh")
    print("=" * 80)


if __name__ == '__main__':
    main()

# Get all fingerprints for given molecule smiles list
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import MACCSkeys
import pubchempy as pcp
import numpy as np
from sklearn.impute import SimpleImputer
from skfp.fingerprints import PubChemFingerprint, AtomPairFingerprint, TopologicalTorsionFingerprint, AutocorrFingerprint
from skfp.fingerprints import AvalonFingerprint


class FingerprintProcessor:
    def __init__(self):
        # Initialize any required settings or libraries
        pass

    def smiles_to_mol(self, smiles_list):
        """Convert SMILES to RDKit Mol objects."""
        return [Chem.MolFromSmiles(smile) for smile in smiles_list]

    def get_rdkit_fingerprints(self, smiles_list):
        """Generate RDKit 2D fingerprints."""
        mols = self.smiles_to_mol(smiles_list)
        return [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols if mol is not None]

    def get_substructure_fingerprints(self, smiles_list):
        """Generate substructure fingerprints."""
        mols = self.smiles_to_mol(smiles_list)
        return [rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol) for mol in mols if mol is not None]

    def MACCSkeys_fp(self, smiles_list):
        """Generate PubChem fingerprints."""
        mols = self.smiles_to_mol(smiles_list)
        return [MACCSkeys.GenMACCSKeys(mol) for mol in mols if mol is not None]
    
    def pubchem_fp(self, smiles_list):
        """Generate PubChem fingerprints."""
        fp = PubChemFingerprint()
        return fp.transform(smiles_list)
    #hashed fingerprint
    def atompair_fp(self, smiles_list):
        """Generate AtomPair fingerprints."""
        fp = AtomPairFingerprint()
        return fp.transform(smiles_list)
    #hashed fingerprint, fragments are computed based on atom environments
    def Avlon_fp(self, smiles_list):
        """Generate Avlon fingerprints."""
        fp = AvalonFingerprint()
        return fp.transform(smiles_list)
    
    def topologicaltorsion_fp(self, smiles_list):
        """Generate TopologicalTorsion fingerprints."""
        fp = TopologicalTorsionFingerprint()
        return fp.transform(smiles_list)
    
    def autocorr_fp(self, smiles_list):
        """Generate Autocorr fingerprints."""
        fp = AutocorrFingerprint()
        return fp.transform(smiles_list)

    #get substructure fingerprints
    #def substurcture_fp(self, smiles_list):



    def process_data(self, smiles_list, fingerprint_type='rdkit'):
        """Process data to get specified type of fingerprints."""
        if fingerprint_type == 'rdkit':
            return self.get_rdkit_fingerprints(smiles_list)
        elif fingerprint_type == 'substructure':
            return self.get_substructure_fingerprints(smiles_list)
        elif fingerprint_type == 'pubchem':
            return self.get_pubchem_fingerprints(smiles_list)
        else:
            raise ValueError("Unsupported fingerprint type")

# Example usage
'''
smiles_data = ['CCO', 'CC', 'CCC']
processor = FingerprintProcessor()
rdkit_fps = processor.process_data(smiles_data, 'rdkit')
print(rdkit_fps)
'''

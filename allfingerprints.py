# Get all fingerprints for given molecule smiles list
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import MACCSkeys
import pubchempy as pcp
import numpy as np
from sklearn.impute import SimpleImputer
from skfp.fingerprints import PubChemFingerprint, AtomPairFingerprint, TopologicalTorsionFingerprint, AutocorrFingerprint, PharmacophoreFingerprint
from skfp.fingerprints import AvalonFingerprint, ECFPFingerprint, E3FPFingerprint, LaggnerFingerprint, MACCSFingerprint, MHFPFingerprint, PhysiochemicalPropertiesFingerprint
from skfp.fingerprints import MQNsFingerprint
from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
from rdkit import DataStructs

# Fingerprints are classified based on binary, count, real-valued features

class FingerprintProcessor:
    def __init__(self):
        # Initialize any required settings or libraries
        pass

    def smiles_to_mol(self, smiles_list):
        """Convert SMILES to RDKit Mol objects."""
        return [Chem.MolFromSmiles(smile) for smile in smiles_list]

    # binary fingerprints: Morgan, MACCSkeys, TopologicalTorsion, AtomPair
    def Morgan_fp(self, smiles_list):
        mols = self.smiles_to_mol(smiles_list)
        return [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols if mol is not None]
    # binary, circular substractures
    def ECFP_fp(self, smiles_list):
        fp =ECFPFingerprint()
        return fp.transform(smiles_list)
    # binary fingerprints, fragments are computed based on "shells", spherical areas around each atom in the 3D conformation of a molecule
    def E3FP_fp(self, smiles_list):
        fp = E3FPFingerprint()
        mol_from_smiles = MolFromSmilesTransformer()
        mols = mol_from_smiles.transform(smiles)
        conf_gen = ConformerGenerator()
        mols = conf_gen.transform(mols)
        return fp.transform(mols)
    #substructure fingerprints, known as SubstructureFingerprint in CDK,307 predefined substructures
    def laggner_fp(self, smiles_list):
        fp = LaggnerFingerprint()
        return fp.transform(smiles_list)
    #substructure, count-based fingerprints
    def MACCS_fp(self, smiles_list):
        fp =MACCSFingerprint()
        return fp.transform(smiles_list)
    #substructure, hashed fingerprints
    def MHFPF_fp(self, smiles_list):
        fp =MHFPFingerprint()
        return fp.transform(smiles_list)
    #count of 42 simple structural features, Molecular Quantum Numbers (MQNs) fingerprint
    def MQNs_fp(self, smiles_list):
        fp =MQNsFingerprint()
        return fp.transform(smiles_list)
    # Pharmacophore fingerprint
    def pharma_fp(self, smiles_list):
        fp = PharmacophoreFingerprint()
        return fp.transform(smiles_list)
    # physiochemical properties fingerprints
    def physio_fp(self, smiles_list):
        fp = PhysiochemicalPropertiesFingerprint()
        return fp.transform(smiles_list)
    
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
    
    def tt_fp(self, smiles_list):
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
        


    def concat_fp(self, smiles_list):
        """Concatenate different types of fingerprints into a single feature vector per molecule."""
        # Generate fingerprints
        rdkit_fps = self.get_rdkit_fingerprints(smiles_list)
        substructure_fps = self.get_substructure_fingerprints(smiles_list)
        maccs_fps = self.MACCSkeys_fp(smiles_list)

        concatenated_fps = []
        for rdkit_fp, substructure_fp, maccs_fp in zip(rdkit_fps, substructure_fps, maccs_fps):
            # Convert RDKit ExplicitBitVect to list of integers
            list_rdkit_fp = [int(bit) for bit in DataStructs.BitVectToText(rdkit_fp)]
            list_substructure_fp = [int(bit) for bit in DataStructs.BitVectToText(substructure_fp)]
            list_maccs_fp = [int(bit) for bit in DataStructs.BitVectToText(maccs_fp)]
            
            # Concatenate the fingerprints
            concatenated_fp = list_rdkit_fp + list_substructure_fp + list_maccs_fp
            concatenated_fps.append(concatenated_fp)
        
        return concatenated_fps

# Example usage
'''
smiles_data = ['CCO', 'CC', 'CCC']
processor = FingerprintProcessor()
rdkit_fps = processor.process_data(smiles_data, 'rdkit')
print(rdkit_fps)
'''

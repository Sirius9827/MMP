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
from sklearn.preprocessing import StandardScaler
from padelpy import from_smiles

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
    
    def padel_fp(self, smiles_list):
        fp = from_smiles(smiles_list.tolist(), fingerprints=True, descriptors=True, threads=4)
        return fp

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
        pubchem_fps = self.pubchem_fp(smiles_list)
        # morgan_fps = self.Morgan_fp(smiles_list)
        #substructure_fps = self.get_substructure_fingerprints(smiles_list)
        # maccs_fps = self.MACCSkeys_fp(smiles_list)
        laggner_fps = self.laggner_fp(smiles_list)
        # fcep_fps = self.ECFP_fp(smiles_list)
        # padel_fps = self.padel_fp(smiles_list)

        # # check data_type
        # print(type(pubchem_fps))
        # print(type(morgan_fps))
        # #print(type(substructure_fps))
        # print(type(maccs_fps))
        # print(type(laggner_fps))
              

        concatenated_fps = []
        for pubchem_fp, laggner_fp in zip(pubchem_fps,laggner_fps):
            # Convert RDKit ExplicitBitVect to list of integers
            # np_morgan_fp = np.array(morgan_fp)
            #np_substructure_fp = np.array(substructure_fp)
            # np_maccs_fp = np.array(maccs_fp)
            # np_fcep_fp = np.array(fcep_fp)
            # np_padel_fp = np.array(padel_fp)
            
            # Concatenate the fingerprints
            concatenated_fp = np.concatenate([pubchem_fp, laggner_fp])
            concatenated_fps.append(concatenated_fp)

            # Convert concatenated_fps to a NumPy array
            # concatenated_fps_array = np.array(concatenated_fps)

            # # Instantiate the StandardScaler
            # scaler = StandardScaler()

            # # Fit the scaler to the data and transform the data
            # normalized_fps = scaler.fit_transform(concatenated_fps_array)
                    
        return concatenated_fps

# Example usage
'''
smiles_data = ['CCO', 'CC', 'CCC']
processor = FingerprintProcessor()
fps = processor.cancat_fp(smiles_data)
print(fps)
'''

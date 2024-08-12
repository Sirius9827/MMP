# Get all fingerprints for given molecule smiles list
from multiprocessing import Pool
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
from descriptastorus.descriptors import rdNormalizedDescriptors
from descriptors import rdDescriptors
# Fingerprints are classified based on binary, count, real-valued features


def generate_orthogonal_matrix(M):
    # define the dimension of the matrix as M: fingerprint length
    # Create a random matrix
    random_matrix = np.random.randn(M, M)
    # Perform QR decomposition
    Q, R = np.linalg.qr(random_matrix)
    return Q

class FingerprintProcessor:
    def __init__(self):
        # Initialize any required settings or libraries
        pass

    def rdkit2D_dp(self, smiles_list):
        """Generate RDKit 2D normalized descriptors."""
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        _, descriptor_lists = generator.processSmiles(smiles_list)
        descriptors_only = [desc[1:] for desc in descriptor_lists]
        # Convert the list of descriptor lists into a 2D NumPy array
        features_array = np.array(descriptors_only)
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        imputed_features_array = imputer.fit_transform(features_array)
        return imputed_features_array
        # with Pool(args.n_jobs) as pool:  
        #     features_map = pool.imap(generator.process, smiless)  
        #     arr = np.array(list(features_map))  
        # Extract descriptors, excluding the first boolean value from each list


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
        mols = mol_from_smiles.transform(smiles_list)
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
        laggner_fps = self.laggner_fp(smiles_list)
        rdkit2d_dps = self.rdkit2D_dp(smiles_list)

        concatenated_fps = []
        for pubchem_fp, laggner_fp, rdkit2d_dp in zip(pubchem_fps,laggner_fps, rdkit2d_dps):
            # Convert RDKit ExplicitBitVect to list of integers
            np_rdkit2d_dp = np.array(rdkit2d_dp)
            # print(type(np_rdkit2d_dp))
            # Concatenate the fingerprints
            concatenated_fp = np.concatenate([pubchem_fp, laggner_fp, np_rdkit2d_dp])
            concatenated_fps.append(concatenated_fp)

            # Convert concatenated_fps to a NumPy array
            # concatenated_fps_array = np.array(concatenated_fps)

            # # Instantiate the StandardScaler
            # scaler = StandardScaler()

            # # Fit the scaler to the data and transform the data
            # normalized_fps = scaler.fit_transform(concatenated_fps_array)
                    
        return concatenated_fps
    
    def orth_fp(self, smiles_list):
        """Generate orthogonal fingerprints."""
        # Generate fingerprints
        pubchem_fps = self.pubchem_fp(smiles_list)
        laggner_fps = self.laggner_fp(smiles_list)
        
        fps = []
        for pubchem_fp, laggner_fp in zip(pubchem_fps, laggner_fps):
            fp = np.concatenate([pubchem_fp, laggner_fp])
            fps.append(fp)
        fps = np.array(fps)
        print(fps.shape)
        ortho_matrix = generate_orthogonal_matrix(fps.shape[1])
        ortho_fps = np.dot(fps, ortho_matrix)

        # ortho_matrix = generate_orthogonal_matrix(laggner_fps.shape[1])
        # laggner_fps = np.dot(laggner_fps, ortho_matrix)
        rdkit2d_dps = self.rdkit2D_dp(smiles_list)

        concatenated_fps = []
        for ortho_fp, rdkit2d_dp in zip(ortho_fps, rdkit2d_dps):
            # Convert RDKit ExplicitBitVect to list of integers
            np_rdkit2d_dp = np.array(rdkit2d_dp)
            # print(type(np_rdkit2d_dp))
            # Concatenate the fingerprints
            concatenated_fp = np.concatenate([ortho_fp, np_rdkit2d_dp])
            concatenated_fps.append(concatenated_fp)

        return concatenated_fps


# Example usage
'''
smiles_data = ['CCO', 'CC', 'CCC']
processor = FingerprintProcessor()
fps = processor.cancat_fp(smiles_data)
print(fps)
'''

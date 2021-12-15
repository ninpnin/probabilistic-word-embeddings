import tensorflow as tf
import pickle
from probabilistic_word_embeddings.utils import dict_to_sparse

def _read_pairs(fname, d, N, suffix1="", suffix2="", cross=False):
    f = open(fname, "r")
    
    tensor_dict = {}
    for line in f:
        l = line.split()
        a, b = l[0], l[1].lower()
        a, b = a + suffix1, b + suffix2        
        if a in d and b in d and (d[a], d[b]) not in tensor_dict:
            ix1, ix2 = d[a], d[b]
            
            index1 = (ix1, ix2)
            index2 = (ix2, ix1)
            index3 = (ix1, ix1)
            index4 = (ix2, ix2)
            
            tensor_dict[index1] = -1.
            tensor_dict[index2] = -1.
            
            tensor_dict[index3] = 1. + tensor_dict.get(index3, 0.)
            tensor_dict[index4] = 1. + tensor_dict.get(index4, 0.)
    
    return dict_to_sparse(tensor_dict, (N,N))
    
def _cross_pairs(fname, d, N, suffix1="", suffix2="", cross=False):
    f = open(fname, "r")
    
    tensor_dict = {}
    
    for line in f:
        l = line.split()
        a, b = l[0], l[1].lower()
        a, b = a + suffix1, b + suffix2
        
        if a in d and b in d:
            ix1, ix2 = d[a], d[b]
            
            index1 = (ix1, ix2 + N)
            index2 = (ix2 + N, ix1)
            index3 = (ix1 + N, ix2)
            index4 = (ix2, ix1 + N)
            off_diagonal = [index1, index2, index3, index4]
            
            index5 = (ix1, ix1)
            index6 = (ix2, ix2)
            index7 = (ix1 + N, ix1 + N)
            index8 = (ix2 + N, ix2 + N)
            diagonal = [index5, index6, index7, index8]
            # Off-diagonal entries -1
            for index in off_diagonal:
                tensor_dict[index] = -1. + tensor_dict.get(index, 0.)
            
            # Diagonal entries 1
            for index in diagonal:
                tensor_dict[index] = 1. + tensor_dict.get(index, 0.)
    
    return dict_to_sparse(tensor_dict, (2*N,2*N))

def _create_crosslingual_laplacian(args):
    filename = args.fpath
    langcodes = filename.split("/")[-1].split(".")[0].split("_")[-2:]
    
    lan1, lan2 = "_" + langcodes[0], "_" + langcodes[1]
    
    dict_path = "fits/dictionary.pkl"
    d = pickle.load(open(dict_path, "rb"))
    V = max(d.values()) + 1

    sparse_tensor = _read_pairs(filename, d, 2 * V, suffix1=lan1, suffix2=lan2)
    return sparse_tensor

def _create_monolingual_laplacian(fpath, dict_path):
    filename = fpath
    dict_path = "fits/dictionary.pkl"
    d = pickle.load(open(dict_path, "rb"))
    V = max(d.values()) + 1
    sparse_tensor = _cross_pairs(filename, d, V)
    return sparse_tensor

def create_laplacian(data_type="monolingual", fpath=None, dpath=None):
    if data_type == "monolingual":
        _create_monolingual_laplacian(fpath, dpath)
    elif data_type == "crosslingual":
        _create_crosslingual_laplacian(fpath, dpath)
    else:
        return ValueError("Laplacian type: " + str(data_type) + "not recognized.")

if __name__ == "__main__":
    import argparse
    info = "Create laplacian matrix from side information graph."
    parser = argparse.ArgumentParser(description=info, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', "--type", default="monolingual",  help="Type of the data: monolingual or crosslingual.")
    parser.add_argument("--sipath", required=True, help="Path to the side info file.")
    parser.add_argument("--dpath", required=True, help="Path to the dictionary file.")
    args = parser.parse_args()

    if args.type == "monolingual":
        create_laplacian(data_type="monolingual", fpath=args.sipath, dpath=dpath)
    elif args.type == "crosslingual":
        create_laplacian(data_type="crosslingual", fpath=args.sipath, dpath=dpath)

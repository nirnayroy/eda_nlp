# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *
import pandas as pd

#arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input_file_path", required=True, type=str, help="input dataframe of unaugmented data")
ap.add_argument("--chunk_size", required=False, type=int, help="number of samples from input file to augment")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
args = ap.parse_args()

chunk_size = 10000
if args.chunk_size is not None:
    chunk_size = args.chunk_size
#number of augmented sentences to generate per original sentence
num_aug = 1 #default
if args.num_aug is not None:
    num_aug = args.num_aug

#how much to replace each word by synonyms
alpha_sr = 0.1#default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

#how much to insert new words that are synonyms
alpha_ri = 0.1#default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

#how much to swap words
alpha_rs = 0.1#default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

#how much to delete words
alpha_rd = 0.1#default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')

#generate more data with standard augmentation
def gen_eda(file_path, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    train_df = pd.read_csv(file_path, header=None)
    train_df = train_df[0:chunk_size]
    for i in range(1, len(train_df)):
        label = train_df[0][i]
        sentence = train_df[1][i]
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        train_df.append([label, aug_sentences])
    # concatenate
    
    # shuffle
    train_df.sample(frac=1).reset_index(drop=True)
    # save csv
    train_df.to_csv('augmented_data.csv')
    print("sucess!!")

#main function
if __name__ == "__main__":
    
    #generate augmented sentences and output into a new file
    gen_eda(args.input_file_path, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)

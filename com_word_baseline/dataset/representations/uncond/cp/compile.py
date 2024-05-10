import os
import json
import pickle
import numpy as np

# Define constants
TEST_AMOUNT = 50
WINDOW_SIZE = 512
GROUP_SIZE = 7
MAX_LEN = WINDOW_SIZE * GROUP_SIZE
COMPILE_TARGET = 'linear'
print('[config] MAX_LEN:', MAX_LEN)

# Function to traverse directories
def traverse_dir(root_dir, extension=('mid', 'MID'), amount=None, str_=None, is_pure=False, verbose=False, is_sort=False, is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1:] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext) + 1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list

if __name__ == '__main__':
    # Define paths
    path_root = '/Users/atharvasawant/Downloads/Echo-Deep-Learning-24/midi_rep_cp'
    path_indir = os.path.join(path_root, 'words')

    # Load dictionary
    path_dictionary = os.path.join(path_root, 'dictionary.pkl')
    event2word, word2event = pickle.load(open(path_dictionary, 'rb'))

    # Load all word files
    wordfiles = traverse_dir(path_indir, extension=('npy'))
    n_files = len(wordfiles)

    # Initialize lists
    x_list = []
    y_list = []
    mask_list = []
    seq_len_list = []
    num_groups_list = []
    name_list = []

    # Process each file
    for fidx in range(n_files):
        print('--[{}/{}]-----'.format(fidx, n_files))
        file = wordfiles[fidx]
        words = np.load(file)
        num_words = len(words)
        eos_arr = words[-1][None, ...]

        if num_words >= MAX_LEN - 2:
            print(' [!] too long:', num_words)
            continue

        # Arrange input-output
        x = words[:-1].copy()
        y = words[1:].copy()
        seq_len = len(x)
        print(' > seq_len:', seq_len)

        # Pad with eos
        pad = np.tile(eos_arr, (MAX_LEN - seq_len, 1))
        x = np.concatenate([x, pad], axis=0)
        y = np.concatenate([y, pad], axis=0)
        mask = np.concatenate([np.ones(seq_len), np.zeros(MAX_LEN - seq_len)])

        # Collect data
        x_list.append(x)
        y_list.append(y)
        mask_list.append(mask)
        seq_len_list.append(seq_len)
        num_groups_list.append(int(np.ceil(seq_len / WINDOW_SIZE)))
        name_list.append(file)

    # Sort by sequence length (descending)
    zipped = sorted(zip(seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list), key=lambda x: -x[0])
    seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list = zip(*zipped)

    print('\n\n[Finished]')
    print(' compile target:', COMPILE_TARGET)
    if COMPILE_TARGET == 'XL':
        # Reshape for XL target
        x_final = np.array(x_list).reshape(len(x_list), GROUP_SIZE, WINDOW_SIZE, -1)
        y_final = np.array(y_list).reshape(len(x_list), GROUP_SIZE, WINDOW_SIZE, -1)
        mask_final = np.array(mask_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
    elif COMPILE_TARGET == 'linear':
        # Use linear target
        x_final = np.array(x_list)
        y_final = np.array(y_list)
        mask_final = np.array(mask_list)
    else:
        raise ValueError('Unknown target:', COMPILE_TARGET)

    # Check dimensions
    num_samples = len(seq_len_list)
    print(' > count:', num_samples)
    print(' > x_final shape:', x_final.shape)
    print(' > y_final shape:', y_final.shape)
    print(' > mask_final shape:', mask_final.shape)

    # Split train/test indices
    validation_songs = json.load(open('/Users/atharvasawant/Downloads/Echo-Deep-Learning-24/baseline/dataset/representations/uncond/validation_songs.json', 'r'))
    train_idx = []
    test_idx = []

    # Validation filename map
    fn2idx_map = {
        'fn2idx': dict(),
        'idx2fn': dict(),
    }

    # Run split
    valid_cnt = 0
    for nidx, n in enumerate(name_list):
        flag = True
        for fn in validation_songs:
            if fn in n:
                test_idx.append(nidx)
                flag = False
                fn2idx_map['fn2idx'][fn] = valid_cnt
                fn2idx_map['idx2fn'][valid_cnt] = fn
                valid_cnt += 1
                break
        if flag:
            train_idx.append(nidx)

    # Convert to numpy arrays
    test_idx = np.array(test_idx, dtype=np.int64)
    train_idx = np.array(train_idx, dtype=np.int64)

    # Save validation map
    path_fn2idx_map = os.path.join(path_root, '/Users/atharvasawant/Downloads/Echo-Deep-Learning-24/baseline/dataset/valid_fn2idx_map.json')
    with open(path_fn2idx_map, 'w') as f:
        json.dump(fn2idx_map, f)

    # Save train data
    path_train = os.path.join(path_root, 'train_data_{}'.format(COMPILE_TARGET))
    path_train += '.npz'
    np.savez(path_train,
             x=x_final[train_idx],
             y=y_final[train_idx],
             mask=mask_final[train_idx],
             seq_len=np.array(seq_len_list)[train_idx],
             num_groups=np.array(num_groups_list)[train_idx])

    # Save test data
    path_test = os.path.join(path_root, 'test_data_{}'.format(COMPILE_TARGET))
    path_test += '.npz'
    np.savez(path_test,
             x=x_final[test_idx],
             y=y_final[test_idx],
             mask=mask_final[test_idx],
             seq_len=np.array(seq_len_list)[test_idx],
             num_groups=np.array(num_groups_list)[test_idx])

    print('---')
    print(' > train x shape:', x_final[train_idx].shape)
    print(' > test x shape:', x_final[test_idx].shape)

import os
import urllib.request
import gzip
import shutil
import hashlib
import h5py
from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve
from torch.utils import data
import numpy as onp

class DatasetNumpy(data.Dataset):
    """ Numpy based dataset generator. """
    def __init__(self, spikes, labels, name, target_dim):
        self.nb_steps = 100
        self.nb_units = 700
        self.max_time = 1.4 
        self.spikes = spikes
        self.labels = labels
        self.name = name

        self.firing_times = self.spikes['times']
        self.units_fired  = self.spikes['units']    
        self.num_samples = self.firing_times.shape[0]
        self.time_bins = onp.linspace(0, self.max_time, num=self.nb_steps)

        self.input  = onp.zeros(
            (self.num_samples, self.nb_steps, self.nb_units), dtype=onp.uint8)
        self.output = onp.array(self.labels, dtype=onp.uint8)

        self.load_spikes()
        self.reduce_inp_dimensions(target_dim=target_dim, axis=2)

    def __len__(self):
        return self.num_samples

    def load_spikes(self):
        for idx in range(self.num_samples):
            times = onp.digitize(self.firing_times[idx], self.time_bins)
            units = self.units_fired[idx] 
            self.input[idx, times, units] = 1

    def reduce_inp_dimensions(self, target_dim, axis):
        """
        Reduces the input dimensions by sampling over its channel axis.

        This method allows interfacing to only a few neuron tiles available on
        hardware. Moreover, by concatenating generated samples, it increases 
        the training dataset size as a data augmentation technique.
        """
        sample_ind = int(onp.ceil(self.nb_units / target_dim))
        index = [onp.arange(i, 700, sample_ind) for i in range(sample_ind)]
        reshaped = [
            onp.take(self.input, index[i], axis) for i in range(sample_ind)
        ]
        reshaped = [
            onp.pad(
                reshaped[i], 
                [(0, 0), (0, 0), (0, int(target_dim - reshaped[i].shape[2]))],
                mode='constant'
            ) for i in range(sample_ind)
        ]
        reshaped = onp.concatenate(reshaped, axis=0)
        self.input = reshaped
        self.output = onp.tile(self.output, sample_ind)
        self.num_samples = reshaped.shape[0]


    def __getitem__(self, idx):
        inputs, outputs = self.__data_generation(idx)
        return inputs, outputs


    def __data_generation(self, idx):
        output = None
        if self.name == 'shd':
            output = self.output[idx]
        elif self.name == 'ssc':
            output = self.output[idx] + 20
        return self.input[idx], output


def numpy_collate(batch):
  if isinstance(batch[0], onp.ndarray):
    return onp.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return onp.array(batch)


class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        batch_sampler=batch_sampler, num_workers=num_workers,
        collate_fn=numpy_collate, pin_memory=pin_memory,
        drop_last=drop_last, timeout=timeout,
        worker_init_fn=worker_init_fn)


def get_audio_dataset(cache_dir, cache_subdir, dataset_name):
    # The remote directory with the data files
    base_url = "https://zenkelab.org/datasets"

    # Retrieve MD5 hashes from remote
    response = urllib.request.urlopen("%s/md5sums.txt"%base_url)
    data = response.read() 
    lines = data.decode('utf-8').split("\n")
    file_hashes = { 
        line.split()[1]:line.split()[0] 
        for line in lines if len(line.split())==2 
    }

    # Download the Spiking Heidelberg Digits (SHD) dataset
    if dataset_name == 'shd':
        files = [ "shd_train.h5.gz", "shd_test.h5.gz"]
    elif dataset_name == 'ssc':
        files = [ "ssc_train.h5.gz", "ssc_test.h5.gz"]
    elif dataset_name == 'all':
        files = [ 
            "shd_train.h5.gz", "shd_test.h5.gz", 
            "ssc_train.h5.gz", "ssc_test.h5.gz"
        ]
        
    for fn in files:
        origin = "%s/%s"%(base_url,fn)
        hdf5_file_path = get_and_gunzip(
            origin, fn, md5hash=file_hashes[fn], 
            cache_dir=cache_dir, cache_subdir=cache_subdir)
        print(f"Available at: {hdf5_file_path}")


def get_and_gunzip(origin, filename, md5hash=None, 
                   cache_dir=None, cache_subdir=None):
    gz_file_path = get_file(
        filename, origin, md5_hash=md5hash, 
        cache_dir=cache_dir, cache_subdir=cache_subdir
    )
    hdf5_file_path = gz_file_path[:-3]
    if (not os.path.isfile(hdf5_file_path) or 
        os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path)):
        print("Decompressing %s"%gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_file(fname,
             origin,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.data-cache')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.data-cache')
    datadir = os.path.join(datadir_base, cache_subdir)

    # Create directories if they don't exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
    # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' + file_hash +
                      ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)

    return fpath


def get_numpy_datasets(dataset_name, n_inp):
    """ Retrieves and prepares numpy datasets based on the specified dataset name. """
    cache_dir = os.path.expanduser("~/tmp/.data-cache")
    cache_subdir = "audiospikes"
    get_audio_dataset(cache_dir, cache_subdir, dataset_name)

    train_ds, test_ds = [], []
    if dataset_name in ['shd', 'all']:
        train_shd_path = os.path.join(cache_dir, cache_subdir, 'shd_train.h5')
        test_shd_path = os.path.join(cache_dir, cache_subdir, 'shd_test.h5')

        print(f"Attempting to open: {train_shd_path}")
        with h5py.File(train_shd_path, 'r') as train_shd_file:
            shd_train_ds = DatasetNumpy(
                train_shd_file['spikes'], train_shd_file['labels'], 
                name='shd', target_dim=n_inp
            )
        print(f"Attempting to open: {test_shd_path}")
        with h5py.File(test_shd_path, 'r') as test_shd_file:
            shd_test_ds = DatasetNumpy(
                test_shd_file['spikes'], test_shd_file['labels'], 
                name='shd', target_dim=n_inp
            )

        train_ds.append(shd_train_ds)
        test_ds.append(shd_test_ds)

    elif dataset_name in ['ssc', 'all']:
        train_ssc_path = os.path.join(cache_dir, cache_subdir, 'ssc_train.h5')
        test_ssc_path = os.path.join(cache_dir, cache_subdir, 'ssc_test.h5')

        with h5py.File(train_ssc_path, 'r') as train_ssc_file:
            ssc_train_ds = DatasetNumpy(
                train_ssc_file['spikes'], train_ssc_file['labels'], 
                name='ssc', target_dim=n_inp
            )
        with h5py.File(test_ssc_path, 'r') as test_ssc_file:
            ssc_test_ds = DatasetNumpy(
                test_ssc_file['spikes'], test_ssc_file['labels'], 
                name='ssc', target_dim=n_inp
            )

        train_ds.append(ssc_train_ds)
        test_ds.append(ssc_test_ds)

    return train_ds, test_ds
